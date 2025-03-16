
import torch # optional (if you want this tool; pip install torch)
from .reader import GGUFReader, GGUFValueType

def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        if len(field.types) != 1 or field.types[0] != GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]])
    else:
        raise TypeError(f"Unknown field type {field_type}")

def get_list_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        return tuple(str(field.parts[part_idx], encoding="utf-8") for part_idx in field.data)
    elif field_type in [int, float, bool]:
        return tuple(field_type(field.parts[part_idx][0]) for part_idx in field.data)
    else:
        raise TypeError(f"Unknown field type {field_type}")

def gguf_tokenizer_loader(path, temb_shape):
    print(f"Attempting to rebuild sentencepiece tokenizer from metadata..")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("protobuf is required; pip install protobuf")
    spm = model.ModelProto()
    reader = GGUFReader(path)
    spm.trainer_spec.model_type == 1
    spm.normalizer_spec.add_dummy_prefix = get_field(reader, "tokenizer.ggml.add_space_prefix", bool)
    spm.normalizer_spec.remove_extra_whitespaces = get_field(reader, "tokenizer.ggml.remove_extra_whitespaces", bool)
    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)
    for idx, (token, score, toktype) in enumerate(zip(tokens, scores, toktypes)):
        piece = spm.SentencePiece()
        piece.piece = token
        piece.score = score
        piece.type = toktype
        spm.pieces.append(piece)
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.vocab_size = len(tokens)
    spm.trainer_spec.max_sentence_length = 4096
    spm.trainer_spec.eos_id = get_field(reader, "tokenizer.ggml.eos_token_id", int)
    spm.trainer_spec.pad_id = get_field(reader, "tokenizer.ggml.padding_token_id", int)
    print(f"Rebuilt tokenizer successfully with vocab size of {len(spm.pieces)}")
    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))
