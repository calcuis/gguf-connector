
from .const import GGML_QUANT_VERSION, LlamaFileType
from .reader import GGUFReader
from .writer import GGUFWriter
from tkinter import filedialog, messagebox
import tkinter as tk
import os

def get_arch_str(reader):
    field = reader.get_field("general.architecture")
    return str(field.parts[field.data[-1]], encoding="utf-8")

def get_file_type(reader):
    field = reader.get_field("general.file_type")
    ft = int(field.parts[field.data[-1]])
    return LlamaFileType(ft)

class GGUFExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GGUF Tensor Extractor")
        icon = tk.PhotoImage(file = os.path.join(os.path.dirname(__file__), "logo.png"))
        self.root.iconphoto(False, icon)
        self.file_path = None
        self.reader = None
        # Load File Button
        self.load_button = tk.Button(root, text="Load GGUF File", command=self.load_file)
        self.load_button.pack(pady=5)
        # Scrollable Listbox for Tensors
        frame = tk.Frame(root)
        frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.tensor_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, width=60, height=20)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=self.tensor_listbox.yview)
        self.tensor_listbox.config(yscrollcommand=scrollbar.set)
        self.tensor_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        self.extract_selected_button = tk.Button(button_frame, text="Extract Selected Tensors", command=self.extract_selected)
        self.extract_selected_button.pack(side="left", padx=10)
        self.extract_unselected_button = tk.Button(button_frame, text="Extract Unselected Tensors", command=self.extract_unselected)
        self.extract_unselected_button.pack(side="left", padx=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("GGUF files", "*.gguf")])
        if not file_path:
            return
        try:
            with open(file_path, "rb") as f:
                self.reader = GGUFReader(f)
            self.file_path = file_path
            self.tensor_listbox.delete(0, tk.END)
            for idx, tensor in enumerate(self.reader.tensors):
                info = f"{tensor.name} — shape: {tensor.shape}, dtype: {tensor.tensor_type}"
                self.tensor_listbox.insert(tk.END, info)
            messagebox.showinfo("Success", f"Loaded {len(self.reader.tensors)} tensors from file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read GGUF file:\n{e}")

    def extract_tensors(self, indices_to_extract, output_title):
        if not self.reader or not self.file_path:
            messagebox.showwarning("Warning", "Please load a GGUF file first.")
            return
        if not indices_to_extract:
            messagebox.showwarning("Warning", f"No tensors to {output_title.lower()}.")
            return
        save_path = filedialog.asksaveasfilename(
            title=f"Save {output_title} File As",
            defaultextension=".gguf",
            initialfile="output.gguf",
            filetypes=[("GGUF files", "*.gguf")]
        )
        if not save_path:
            return
        try:
            arch = get_arch_str(self.reader)
            file_type = get_file_type(self.reader)
            writer = GGUFWriter(path=None, arch=arch)
            writer.add_quantization_version(GGML_QUANT_VERSION)
            writer.add_file_type(file_type)
            for idx in indices_to_extract:
                tensor = self.reader.tensors[idx]
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            with open(save_path, "wb"):
                writer.write_header_to_file(path=save_path)
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file(progress=True)
                writer.close()
            messagebox.showinfo("Success", f"{output_title} saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write file:\n{e}")

    def extract_selected(self):
        selected_indices = self.tensor_listbox.curselection()
        self.extract_tensors(selected_indices, "Selected Tensors")

    def extract_unselected(self):
        all_indices = set(range(self.tensor_listbox.size()))
        selected_indices = set(self.tensor_listbox.curselection())
        unselected_indices = sorted(all_indices - selected_indices)
        self.extract_tensors(unselected_indices, "Unselected Tensors")
