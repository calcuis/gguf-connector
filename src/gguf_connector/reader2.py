import struct

class ReaderError(Exception):
    """Exception raised for errors in GGUF reader"""

class GGUFReader:
    GGUF_FORMAT = b"GGUF"
    VALUE_FORMATS = {
        0: "B",  # UINT8
        1: "b",  # INT8
        2: "H",  # UINT16
        3: "h",  # INT16
        4: "I",  # UINT32
        5: "i",  # INT32
        6: "f",  # FLOAT32
        7: "?",  # BOOL
        10: "Q",  # UINT64
        11: "q",  # INT64
        12: "d",  # FLOAT64
    }
    TENSOR_TYPES = {

        0: "GGML_TYPE_F32",
        1: "GGML_TYPE_F16",#            = 1   # except 1d tensors
        2: "GGML_TYPE_Q4_0",#           = 2   # except 1d tensors
        3: "GGML_TYPE_Q4_1",#           = 3   # except 1d tensors
        4: "GGML_TYPE_Q4_1_SOME_F16",#  = 4   # tok_embeddings.weight and output.weight are F16
        # GGML_TYPE_Q4_2                = 5   # support has been removed
        # GGML_TYPE_Q4_3                = 6   # support has been removed
        7: "GGML_TYPE_Q8_0",#           = 7   # except 1d tensors
        8: "GGML_TYPE_Q5_0",#           = 8   # except 1d tensors
        9: "GGML_TYPE_Q5_1",#           = 9   # except 1d tensors
        10: "GGML_TYPE_Q2_K",#          = 10  # except 1d tensors
        11: "GGML_TYPE_Q3_K_S",#        = 11  # except 1d tensors
        12: "GGML_TYPE_Q3_K_M",#        = 12  # except 1d tensors
        13: "GGML_TYPE_Q3_K_L",#        = 13  # except 1d tensors
        14: "GGML_TYPE_Q4_K_S",#        = 14  # except 1d tensors
        15: "GGML_TYPE_Q4_K_M",#        = 15  # except 1d tensors
        16: "GGML_TYPE_Q5_K_S",#        = 16  # except 1d tensors
        17: "GGML_TYPE_Q5_K_M",#        = 17  # except 1d tensors
        18: "GGML_TYPE_Q6_K",#          = 18  # except 1d tensors
        19: "GGML_TYPE_IQ2_XXS",#       = 19  # except 1d tensors
        20: "GGML_TYPE_IQ2_XS",#        = 20  # except 1d tensors
        21: "GGML_TYPE_Q2_K_S",#        = 21  # except 1d tensors
        22: "GGML_TYPE_IQ3_XS",#        = 22  # except 1d tensors
        23: "GGML_TYPE_IQ3_XXS",#       = 23  # except 1d tensors
        24: "GGML_TYPE_IQ1_S",#         = 24  # except 1d tensors
        25: "GGML_TYPE_IQ4_NL",#        = 25  # except 1d tensors
        26: "GGML_TYPE_IQ3_S",#         = 26  # except 1d tensors
        27: "GGML_TYPE_IQ3_M",#         = 27  # except 1d tensors
        28: "GGML_TYPE_IQ2_S",#         = 28  # except 1d tensors
        29: "GGML_TYPE_IQ2_M",#         = 29  # except 1d tensors
        30: "GGML_TYPE_IQ4_XS",#        = 30  # except 1d tensors
        31: "GGML_TYPE_IQ1_M",#         = 31  # except 1d tensors
        32: "GGML_TYPE_BF16",#          = 32  # except 1d tensors
    }

    def __init__(self, file_path):
        """Initialize the GGUF reader"""
        self.file_path = file_path
        self.version = None
        self.format = None
        self.tensors_info = None
        self.metadata = None
        self.alignment = None

    def read(self):
        """Read the GGUF file."""
        with open(self.file_path, "rb") as f:
            # Check the file type
            self.format = f.read(4)
            if self.format != self.GGUF_FORMAT:
                raise ReaderError("Invalid format")
            # Check the version
            self.version = struct.unpack("I", f.read(4))[0]
            # if self.version != 3:
            #     raise ReaderError("Unsupported version")
            tensor_count = struct.unpack("Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("Q", f.read(8))[0]
            self.metadata = {}
            for _ in range(metadata_kv_count):
                key, value = self._read_metadata_kv(f)
                self.metadata[key] = value
            self.alignment = self.metadata.get("general.alignment", 1)
            self.tensors_info = []
            for _ in range(tensor_count):
                tensor_info = self._read_tensor_info(f)
                self.tensors_info.append(tensor_info)

    def _read_string(self, f):
        """Read a string from the file"""
        length = struct.unpack("Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    def _read_metadata_kv(self, f):
        """Read a metadata key-value pair from the file"""
        key = self._read_string(f)
        value_type = struct.unpack("I", f.read(4))[0]
        value = self._read_value(f, value_type)
        return key, value

    def _read_value(self, f, value_type):
        """Read a value of the given type from the file"""
        if value_type in self.VALUE_FORMATS:
            return struct.unpack(
                self.VALUE_FORMATS[value_type],
                f.read(struct.calcsize(self.VALUE_FORMATS[value_type])),
            )[0]
        if value_type == 8:  # STRING
            return self._read_string(f)
        if value_type == 9:  # ARRAY
            array_type = struct.unpack("I", f.read(4))[0]
            array_len = struct.unpack("Q", f.read(8))[0]
            return [self._read_value(f, array_type) for _ in range(array_len)]
        raise ReaderError("Unsupported value type")

    def _read_tensor_info(self, f):
        """Read tensor information from the file"""
        name = self._read_string(f)
        n_dimensions = struct.unpack("I", f.read(4))[0]
        dimensions = struct.unpack(f"{n_dimensions}Q", f.read(8 * n_dimensions))
        tensor_type = struct.unpack("I", f.read(4))[0]
        offset = struct.unpack("Q", f.read(8))[0]
        return {
            "name": name,
            "n_dimensions": n_dimensions,
            "dimensions": dimensions,
            "type": tensor_type,
            "offset": offset,
        }

    def load_tensors(self):
        """Load the tensors from the file"""
        tensors = []
        with open(self.file_path, "rb") as f:
            for tensor_info in self.tensors_info:
                f.seek(tensor_info["offset"])
                tensor_data = f.read(tensor_info["n_dimensions"])
                if (f.tell() % self.alignment) != 0:
                    f.read(self.alignment - (f.tell() % self.alignment))
                tensors.append(tensor_data)
        return tensors

    def print(self):
        """Print the file details"""
        print(f"Version: {self.version}")
        print(f"Format: {self.format}")
        print("Tensors Info:")
        for tensor_info in self.tensors_info:
            print(
                f"  Name: {tensor_info['name']},\tShape: {tensor_info['dimensions']},"
                f"\tType: {self.TENSOR_TYPES[tensor_info['type']]},"
                f"\tOffset: {tensor_info['offset']}"
            )
        print("Metadata:")
        for key, value in self.metadata.items():
            if isinstance(value, list) and len(value) > 50:
                print(f"  {key}: {value[:50]}... ({len(value) - 50} more elements)")
            else:
                print(f"  {key}: {value}")
