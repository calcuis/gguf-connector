
from .const import GGML_QUANT_VERSION, LlamaFileType
from .reader import GGUFReader
from .writer import GGUFWriter
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def get_arch_str(reader):
    field = reader.get_field("general.architecture")
    return str(field.parts[field.data[-1]], encoding="utf-8")

def get_file_type(reader):
    field = reader.get_field("general.file_type")
    ft = int(field.parts[field.data[-1]])
    return LlamaFileType(ft)

def remove_tensors(input_path, tensor_names_to_remove, output_path):
    with open(input_path, "rb") as f:
        reader = GGUFReader(f)
        arch = get_arch_str(reader)
        file_type = get_file_type(reader)

        writer = GGUFWriter(path=None, arch=arch)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        writer.add_file_type(file_type)

        for tensor in reader.tensors:
            if tensor.name in tensor_names_to_remove:
                continue
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)

        with open(output_path, "wb"):
            writer.write_header_to_file(path=output_path)
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
            writer.close()

class GGUFRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GGUF Tensor Remover")
        icon = tk.PhotoImage(file = os.path.join(os.path.dirname(__file__), "logo.png"))
        self.root.iconphoto(False, icon)
        self.reader = None
        self.input_file = None
        self.file_label = tk.Label(root, text="No file loaded.")
        self.file_label.pack(pady=5)
        tk.Button(root, text="Load GGUF File", command=self.load_file).pack(pady=5)
        # Frame for treeview and scrollbar
        frame = tk.Frame(root)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        # Treeview for tensor listing
        self.tree = ttk.Treeview(frame, columns=("name", "shape", "dtype"), show="headings", selectmode="extended")
        self.tree.heading("name", text="Tensor Name")
        self.tree.heading("shape", text="Shape")
        self.tree.heading("dtype", text="DType")
        self.tree.column("name", width=250)
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill=tk.BOTH, expand=True)
        scrollbar.pack(side="right", fill=tk.Y)
        self.remove_button = tk.Button(root, text="Remove Selected Tensors", command=self.remove_selected)
        self.remove_button.pack(pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("GGUF files", "*.gguf")])
        if not file_path:
            return
        self.input_file = file_path
        self.file_label.config(text=f"Loaded: {os.path.basename(file_path)}")
        self.populate_tensor_list()

    def populate_tensor_list(self):
        self.tree.delete(*self.tree.get_children())
        with open(self.input_file, "rb") as f:
            self.reader = GGUFReader(f)
            for tensor in self.reader.tensors:
                self.tree.insert("", "end", iid=tensor.name,
                                 values=(tensor.name, str(tensor.shape), str(tensor.tensor_type)))

    def remove_selected(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select one or more tensors to remove.")
            return
        tensor_names = [self.tree.item(i)["values"][0] for i in selected_items]
        confirm = messagebox.askyesno("Confirm Batch Deletion",
                                      f"Remove the following {len(tensor_names)} tensor(s)?\n\n" +
                                      "\n".join(tensor_names[:10]) +
                                      ("\n..." if len(tensor_names) > 10 else ""))
        if not confirm:
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".gguf",
            filetypes=[("GGUF files", "*.gguf")],
            initialfile="output.gguf",
            title="Save Output GGUF As"
        )

        if not output_path:
            return

        remove_tensors(self.input_file, tensor_names, output_path)
        messagebox.showinfo("Success", f"{len(tensor_names)} tensor(s) removed.\nSaved as: {output_path}")
        self.populate_tensor_list()
