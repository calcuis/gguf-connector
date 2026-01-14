
import os, shutil
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from safetensors.torch import safe_open, save_file

TYPE_PARSERS = {
    "INT8": lambda v: int(v),
    "INT16": lambda v: int(v),
    "INT32": lambda v: int(v),
    "INT64": lambda v: int(v),
    "UINT8": lambda v: int(v) if int(v) >= 0 else None,
    "UINT16": lambda v: int(v) if int(v) >= 0 else None,
    "UINT32": lambda v: int(v) if int(v) >= 0 else None,
    "UINT64": lambda v: int(v) if int(v) >= 0 else None,
    "FLOAT32": lambda v: float(v),
    "FLOAT64": lambda v: float(v),
    "BOOL": lambda v: v.lower() in ("true", "1", "yes", "false", "0", "no"),
    "STRING": lambda v: v,
}

BOOL_TRUE = {"true", "1", "yes"}
BOOL_FALSE = {"false", "0", "no"}

def validate_value(value_type, raw_value):
    try:
        if value_type == "BOOL":
            v = raw_value.lower()
            if v in BOOL_TRUE:
                return "true"
            if v in BOOL_FALSE:
                return "false"
            raise ValueError
        parsed = TYPE_PARSERS[value_type](raw_value)
        return str(parsed)
    except Exception:
        raise ValueError(f"Invalid value for type {value_type}")

class SafeTensorsEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SafeTensors Metadata Editor")
        icon = tk.PhotoImage(file = os.path.join(os.path.dirname(__file__), "logo.png"))
        self.iconphoto(False, icon)

        self.file_path = None
        self.metadata = {}
        self.tensors = {}

        self.create_ui()

    def create_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=5)

        ttk.Button(top, text="Open .safetensors", command=self.open_file).pack(side="left")

        self.file_label = ttk.Label(top, text="No file selected")
        self.file_label.pack(side="left", padx=10)

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(mid, columns=("key", "value"), show="headings")
        self.tree.heading("key", text="Key")
        self.tree.heading("value", text="Value")
        self.tree.pack(fill="both", expand=True)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=5)

        ttk.Button(bottom, text="Add / Edit", command=self.add_edit).pack(side="left")
        ttk.Button(bottom, text="Delete", command=self.delete_key).pack(side="left", padx=5)
        ttk.Button(bottom, text="Save", command=self.save_file).pack(side="right")

    def open_file(self):
        path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            filetypes=[("SafeTensors", "*.safetensors")]
        )
        if not path:
            return

        self.file_path = path
        self.file_label.config(text=os.path.basename(path))

        with safe_open(path, framework="pt") as f:
            self.metadata = dict(f.metadata() or {})
            # self.tensors = {k: f.get_tensor(k) for k in f.keys()}
            self.tensors = {k: f.get_tensor(k).clone() for k in f.keys()}

        self.refresh_tree()

    def refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for k, v in sorted(self.metadata.items()):
            self.tree.insert("", "end", values=(k, v))

    def add_edit(self):
        dialog = tk.Toplevel(self)
        dialog.title("Add / Edit Metadata")
        dialog.geometry("400x220")

        ttk.Label(dialog, text="Key").pack()
        key_entry = ttk.Entry(dialog)
        key_entry.pack(fill="x", padx=10)

        ttk.Label(dialog, text="Type").pack()
        type_box = ttk.Combobox(
            dialog,
            values=list(TYPE_PARSERS.keys()),
            state="readonly"
        )
        type_box.set("STRING")
        type_box.pack(fill="x", padx=10)

        ttk.Label(dialog, text="Value").pack()
        value_entry = ttk.Entry(dialog)
        value_entry.pack(fill="x", padx=10)

        selected = self.tree.selection()
        if selected:
            k, v = self.tree.item(selected[0])["values"]
            key_entry.insert(0, k)
            value_entry.insert(0, v)

        def apply():
            key = key_entry.get().strip()
            raw_value = value_entry.get().strip()
            value_type = type_box.get()

            if not key:
                messagebox.showerror("Error", "Key cannot be empty")
                return

            try:
                value = validate_value(value_type, raw_value)
            except ValueError as e:
                messagebox.showerror("Invalid Value", str(e))
                return

            self.metadata[key] = value
            self.refresh_tree()
            dialog.destroy()

        ttk.Button(dialog, text="Apply", command=apply).pack(pady=10)

    def delete_key(self):
        selected = self.tree.selection()
        if not selected:
            return
        key = self.tree.item(selected[0])["values"][0]
        if messagebox.askyesno("Confirm", f"Delete metadata key '{key}'?"):
            self.metadata.pop(key, None)
            self.refresh_tree()

    def save_file(self):
        if not self.file_path:
            return

        tmp_path = self.file_path + ".tmp"

        save_file(
            self.tensors,
            tmp_path,
            metadata=self.metadata
        )

        shutil.move(tmp_path, self.file_path)
        messagebox.showinfo("Saved", "Metadata saved successfully")

app = SafeTensorsEditor()
app.mainloop()
