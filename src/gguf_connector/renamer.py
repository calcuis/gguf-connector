
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

class GGUFEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GGUF Tensor Renamer")
        icon = tk.PhotoImage(file = os.path.join(os.path.dirname(__file__), "logo.png"))
        self.root.iconphoto(False, icon)
        self.tensor_entries = []
        self.reader = None
        self.input_file = None
        self.build_ui()

    def build_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        # File selection
        ttk.Button(frame, text="Load GGUF File", command=self.load_file).pack(pady=(0, 10))
        # Batch options
        batch_frame = ttk.LabelFrame(frame, text="Batch Rename Options")
        batch_frame.pack(pady=5, fill="x")
        # Prefix
        prefix_frame = ttk.Frame(batch_frame)
        prefix_frame.pack(pady=2, fill="x")
        ttk.Label(prefix_frame, text="Prefix:").pack(side="left")
        self.prefix_entry = ttk.Entry(prefix_frame, width=30)
        self.prefix_entry.pack(side="left", padx=5)
        ttk.Button(prefix_frame, text="Apply", command=self.apply_prefix).pack(side="left")
        # Suffix
        suffix_frame = ttk.Frame(batch_frame)
        suffix_frame.pack(pady=2, fill="x")
        ttk.Label(suffix_frame, text="Suffix:").pack(side="left")
        self.suffix_entry = ttk.Entry(suffix_frame, width=30)
        self.suffix_entry.pack(side="left", padx=5)
        ttk.Button(suffix_frame, text="Apply", command=self.apply_suffix).pack(side="left")
        # Search and Replace
        sr_frame = ttk.Frame(batch_frame)
        sr_frame.pack(pady=2, fill="x")
        ttk.Label(sr_frame, text="Search:").pack(side="left")
        self.search_entry = ttk.Entry(sr_frame, width=20)
        self.search_entry.pack(side="left", padx=5)
        ttk.Label(sr_frame, text="Replace:").pack(side="left")
        self.replace_entry = ttk.Entry(sr_frame, width=20)
        self.replace_entry.pack(side="left", padx=5)
        ttk.Button(sr_frame, text="Apply", command=self.apply_search_replace).pack(side="left")
        # Scrollable tensor list
        self.canvas = tk.Canvas(frame, height=300)
        self.scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.tensor_frame = ttk.Frame(self.canvas)
        self.tensor_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.tensor_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        # Save button
        ttk.Button(frame, text="Save As...", command=self.save_file).pack(pady=10)

    def load_file(self):
        self.input_file = filedialog.askopenfilename(filetypes=[("GGUF Files", "*.gguf")])
        if not self.input_file:
            return
        with open(self.input_file, "rb") as f:
            self.reader = GGUFReader(f)
        for widget in self.tensor_frame.winfo_children():
            widget.destroy()
        self.tensor_entries.clear()
        for idx, tensor in enumerate(self.reader.tensors):
            ttk.Label(self.tensor_frame, text=f"{idx+1}.").grid(row=idx, column=0, padx=5, sticky="e")
            entry = ttk.Entry(self.tensor_frame, width=60)
            entry.insert(0, tensor.name)
            entry.grid(row=idx, column=1, padx=5, pady=2, sticky="w")
            self.tensor_entries.append((tensor.name, entry))

    def apply_prefix(self):
        prefix = self.prefix_entry.get()
        if not prefix:
            messagebox.showwarning("Input Required", "Please enter a prefix.")
            return
        for _, entry in self.tensor_entries:
            current_value = entry.get()
            if not current_value.startswith(prefix):
                entry.delete(0, tk.END)
                entry.insert(0, prefix + current_value)

    def apply_suffix(self):
        suffix = self.suffix_entry.get()
        if not suffix:
            messagebox.showwarning("Input Required", "Please enter a suffix.")
            return
        for _, entry in self.tensor_entries:
            current_value = entry.get()
            if not current_value.endswith(suffix):
                entry.delete(0, tk.END)
                entry.insert(0, current_value + suffix)

    def apply_search_replace(self):
        search = self.search_entry.get()
        replace = self.replace_entry.get()
        if not search:
            messagebox.showwarning("Input Required", "Please enter a search string.")
            return
        for _, entry in self.tensor_entries:
            current_value = entry.get()
            new_value = current_value.replace(search, replace)
            entry.delete(0, tk.END)
            entry.insert(0, new_value)

    def save_file(self):
        if not self.input_file or not self.reader:
            messagebox.showerror("Error", "No GGUF file loaded.")
            return
        output_path = filedialog.asksaveasfilename(defaultextension=".gguf", initialfile="output.gguf", filetypes=[("GGUF Files", "*.gguf")])
        if not output_path:
            return
        arch = get_arch_str(self.reader)
        file_type = get_file_type(self.reader)
        writer = GGUFWriter(path=None, arch=arch)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        writer.add_file_type(file_type)
        rename_map = {old: entry.get() for old, entry in self.tensor_entries}
        for tensor in self.reader.tensors:
            name = rename_map.get(tensor.name, tensor.name)
            writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type)
        with open(output_path, "wb"):
            writer.write_header_to_file(path=output_path)
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
            writer.close()
        messagebox.showinfo("Success", f"File saved as {os.path.basename(output_path)}")
