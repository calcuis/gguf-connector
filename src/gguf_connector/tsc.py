import os
os.system("git clone https://huggingface.co/cantonese-dev/cantonese-1.2-0.6b")

filename = "ttsc.py"
content = """
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterator
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
from gguf_connector import writer as gw
from gguf_connector import const as gt
from gguf_connector import quant
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
class CantoneseTTSConverter:
    TENSOR_MAP = {
        "talker.model.codec_embedding.weight": "talker.codec_embd.weight",
        "talker.model.text_embedding.weight": "talker.text_embd.weight",
        "talker.codec_head.weight": "talker.codec_head.weight",
        "talker.model.norm.weight": "talker.output_norm.weight",
        "talker.text_projection.linear_fc1.weight": "talker.text_proj.fc1.weight",
        "talker.text_projection.linear_fc1.bias": "talker.text_proj.fc1.bias",
        "talker.text_projection.linear_fc2.weight": "talker.text_proj.fc2.weight",
        "talker.text_projection.linear_fc2.bias": "talker.text_proj.fc2.bias",
        "talker.code_predictor.model.norm.weight": "code_pred.output_norm.weight",
        "talker.code_predictor.small_to_mtp_projection.weight": "code_pred.mtp_proj.weight",
        "talker.code_predictor.small_to_mtp_projection.bias": "code_pred.mtp_proj.bias",
        "speaker_encoder.blocks.0.conv.weight": "spk_enc.conv0.weight",
        "speaker_encoder.blocks.0.conv.bias": "spk_enc.conv0.bias",
        "speaker_encoder.asp.conv.weight": "spk_enc.asp.conv.weight",
        "speaker_encoder.asp.conv.bias": "spk_enc.asp.conv.bias",
        "speaker_encoder.asp.tdnn.conv.weight": "spk_enc.asp.tdnn.weight",
        "speaker_encoder.asp.tdnn.conv.bias": "spk_enc.asp.tdnn.bias",
        "speaker_encoder.mfa.conv.weight": "spk_enc.mfa.weight",
        "speaker_encoder.mfa.conv.bias": "spk_enc.mfa.bias",
        "speaker_encoder.fc.weight": "spk_enc.fc.weight",
        "speaker_encoder.fc.bias": "spk_enc.fc.bias",
    }
    TALKER_LAYER_PATTERNS = [
        (r"talker\.model\.layers\.(\d+)\.input_layernorm\.weight", "talker.blk.{}.attn_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight", "talker.blk.{}.attn_q.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight", "talker.blk.{}.attn_k.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight", "talker.blk.{}.attn_v.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight", "talker.blk.{}.attn_output.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight", "talker.blk.{}.attn_q_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight", "talker.blk.{}.attn_k_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "talker.blk.{}.ffn_norm.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight", "talker.blk.{}.ffn_gate.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.up_proj\.weight", "talker.blk.{}.ffn_up.weight"),
        (r"talker\.model\.layers\.(\d+)\.mlp\.down_proj\.weight", "talker.blk.{}.ffn_down.weight"),
    ]
    CODE_PREDICTOR_LAYER_PATTERNS = [
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.input_layernorm\.weight", "code_pred.blk.{}.attn_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight", "code_pred.blk.{}.attn_q.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight", "code_pred.blk.{}.attn_k.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight", "code_pred.blk.{}.attn_v.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight", "code_pred.blk.{}.attn_output.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight", "code_pred.blk.{}.attn_q_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight", "code_pred.blk.{}.attn_k_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "code_pred.blk.{}.ffn_norm.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight", "code_pred.blk.{}.ffn_gate.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.up_proj\.weight", "code_pred.blk.{}.ffn_up.weight"),
        (r"talker\.code_predictor\.model\.layers\.(\d+)\.mlp\.down_proj\.weight", "code_pred.blk.{}.ffn_down.weight"),
    ]
    CODE_PREDICTOR_CODEBOOK_PATTERNS = [
        (r"talker\.code_predictor\.model\.codec_embedding\.(\d+)\.weight", "code_pred.codec_embd.{}.weight"),
        (r"talker\.code_predictor\.lm_head\.(\d+)\.weight", "code_pred.lm_head.{}.weight"),
    ]
    SPEAKER_ENCODER_PATTERNS = [
        (r"speaker_encoder\.blocks\.(\d+)\.res2net_block\.blocks\.(\d+)\.conv\.weight", "spk_enc.blk.{}.res2net.{}.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.res2net_block\.blocks\.(\d+)\.conv\.bias", "spk_enc.blk.{}.res2net.{}.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv1\.weight", "spk_enc.blk.{}.se.conv1.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv1\.bias", "spk_enc.blk.{}.se.conv1.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv2\.weight", "spk_enc.blk.{}.se.conv2.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.se_block\.conv2\.bias", "spk_enc.blk.{}.se.conv2.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn1\.conv\.weight", "spk_enc.blk.{}.tdnn1.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn1\.conv\.bias", "spk_enc.blk.{}.tdnn1.bias"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn2\.conv\.weight", "spk_enc.blk.{}.tdnn2.weight"),
        (r"speaker_encoder\.blocks\.(\d+)\.tdnn2\.conv\.bias", "spk_enc.blk.{}.tdnn2.bias"),
    ]
    def __init__(
        self,
        input_dir: Path,
        output_path: Path,
        output_type: str = "f16",
    ):
        self.input_dir = input_dir
        self.output_path = output_path
        self.output_type = output_type
        self.config = self._load_config()
        self._extract_params()
    def _load_config(self) -> dict[str, Any]:
        config_path = self.input_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    def _load_generation_config(self) -> dict[str, Any]:
        generation_config_path = self.input_dir / "generation_config.json"
        if not generation_config_path.exists():
            return {}
        with open(generation_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    def _extract_params(self) -> None:
        talker_config = self.config.get("talker_config", {})
        code_predictor_config = talker_config.get("code_predictor_config", {})
        speaker_encoder_config = self.config.get("speaker_encoder_config")
        speaker_encoder_config = speaker_encoder_config if isinstance(speaker_encoder_config, dict) else None
        self.tts_model_type = str(self.config.get("tts_model_type", "base")).lower()
        self.tts_model_size = str(self.config.get("tts_model_size", "0b6")).lower()
        self.tokenizer_type = str(self.config.get("tokenizer_type", "tts_tokenizer_12hz"))
        self.generation_config = self._load_generation_config()
        raw_spk_id_map = talker_config.get("spk_id", {})
        self.spk_id_map = raw_spk_id_map if isinstance(raw_spk_id_map, dict) else {}
        self.im_start_token_id = self.config.get("im_start_token_id", 151644)
        self.im_end_token_id = self.config.get("im_end_token_id", 151645)
        self.tts_pad_token_id = self.config.get("tts_pad_token_id", 151671)
        self.tts_bos_token_id = self.config.get("tts_bos_token_id", 151672)
        self.tts_eos_token_id = self.config.get("tts_eos_token_id", 151673)
        raw_supports_instruction = self.config.get("supports_instruction")
        if isinstance(raw_supports_instruction, bool):
            self.supports_instruction = raw_supports_instruction
        elif isinstance(raw_supports_instruction, int):
            self.supports_instruction = raw_supports_instruction != 0
        elif isinstance(raw_supports_instruction, str):
            self.supports_instruction = raw_supports_instruction.strip().lower() in {"1", "true", "yes", "on"}
        elif self.tts_model_type == "voice_design":
            self.supports_instruction = True
        elif self.tts_model_type == "custom_voice":
            self.supports_instruction = self.tts_model_size == "1b7"
        else:
            self.supports_instruction = False
        self.hidden_size = talker_config.get("hidden_size", 1024)
        self.intermediate_size = talker_config.get("intermediate_size", 3072)
        self.num_hidden_layers = talker_config.get("num_hidden_layers", 28)
        self.num_attention_heads = talker_config.get("num_attention_heads", 16)
        self.num_kv_heads = talker_config.get("num_key_value_heads", 8)
        self.head_dim = talker_config.get("head_dim", 128)
        self.vocab_size = talker_config.get("vocab_size", 3072)  # codec vocab
        self.text_vocab_size = talker_config.get("text_vocab_size", 151936)
        self.text_hidden_size = talker_config.get("text_hidden_size", 2048)
        self.num_code_groups = talker_config.get("num_code_groups", 16)
        self.rms_norm_eps = talker_config.get("rms_norm_eps", 1e-6)
        self.rope_theta = talker_config.get("rope_theta", 1000000)
        self.context_length = talker_config.get("max_position_embeddings", 32768)
        self.position_id_per_seconds = talker_config.get("position_id_per_seconds", 13)
        rope_scaling = talker_config.get("rope_scaling", {})
        self.mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])
        self.mrope_interleaved = bool(rope_scaling.get("interleaved", False))
        self.code_predictor_num_layers = code_predictor_config.get("num_hidden_layers", 5)
        self.code_predictor_vocab_size = code_predictor_config.get("vocab_size", 2048)
        self.code_predictor_hidden_size = code_predictor_config.get("hidden_size", self.hidden_size)
        self.code_predictor_intermediate_size = code_predictor_config.get(
            "intermediate_size", self.intermediate_size
        )
        self.code_predictor_num_attention_heads = code_predictor_config.get(
            "num_attention_heads", self.num_attention_heads
        )
        self.code_predictor_num_kv_heads = code_predictor_config.get(
            "num_key_value_heads", self.num_kv_heads
        )
        self.code_predictor_head_dim = code_predictor_config.get("head_dim", self.head_dim)
        self.code_predictor_rms_norm_eps = code_predictor_config.get("rms_norm_eps", self.rms_norm_eps)
        self.code_predictor_rope_theta = code_predictor_config.get("rope_theta", self.rope_theta)
        self.code_predictor_context_length = code_predictor_config.get("max_position_embeddings", 65536)
        self.has_speaker_encoder_config = speaker_encoder_config is not None
        speaker_cfg = speaker_encoder_config or {}
        self.speaker_enc_dim = speaker_cfg.get("enc_dim", 1024)
        self.speaker_sample_rate = speaker_cfg.get("sample_rate", 24000)
        self.codec_pad_id = talker_config.get("codec_pad_id", 2148)
        self.codec_bos_id = talker_config.get("codec_bos_id", 2149)
        self.codec_eos_id = talker_config.get("codec_eos_token_id", 2150)
        self.codec_think_id = talker_config.get("codec_think_id", 2154)
        self.codec_nothink_id = talker_config.get("codec_nothink_id", 2155)
        self.codec_think_bos_id = talker_config.get("codec_think_bos_id", 2156)
        self.codec_think_eos_id = talker_config.get("codec_think_eos_id", 2157)
        raw_language_map = talker_config.get("codec_language_id", {})
        self.codec_language_id = raw_language_map if isinstance(raw_language_map, dict) else {}
        if self.tts_model_size == "1b7":
            self.model_name = f"Cantonese-12Hz-1.7B-{self.tts_model_type}"
        else:
            self.model_name = f"Cantonese-12Hz-0.6B-{self.tts_model_type}"
    def _map_tensor_name(self, hf_name: str) -> str | None:
        if hf_name in self.TENSOR_MAP:
            return self.TENSOR_MAP[hf_name]
        for pattern, template in self.TALKER_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                layer_idx = match.group(1)
                return template.format(layer_idx)
        for pattern, template in self.CODE_PREDICTOR_LAYER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                layer_idx = match.group(1)
                return template.format(layer_idx)
        for pattern, template in self.CODE_PREDICTOR_CODEBOOK_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                codebook_idx = match.group(1)
                return template.format(codebook_idx)
        for pattern, template in self.SPEAKER_ENCODER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return template.format(groups[0], groups[1])
                else:
                    return template.format(groups[0])
        return None
    def _get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        safetensor_files = list(self.input_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {self.input_dir}")
        for sf_path in sorted(safetensor_files):
            logger.info(f"Loading tensors from {sf_path.name}")
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)
    def _should_quantize(self, tensor_name: str) -> bool:
        if tensor_name == "spk_enc.fc.weight":
            return False
        return True
    def _convert_dtype(self, tensor: torch.Tensor, tensor_name: str = "") -> tuple[np.ndarray, gw.GGMLQuantizationType]:
        if tensor.dtype == torch.bfloat16:
            data = tensor.float().numpy()
        else:
            data = tensor.numpy()
        n_dims = len(data.shape)
        if n_dims == 3 and "weight" in tensor_name:
            logger.info(f"Conv1d weight {tensor_name}: shape {data.shape} [OC,IC,K] - GGUF will reverse to [K,IC,OC]")
        if tensor_name == "spk_enc.fc.weight":
            return data.astype(np.float32), gw.GGMLQuantizationType.F32
        if n_dims <= 1:
            return data.astype(np.float32), gw.GGMLQuantizationType.F32
        if self.output_type == "f32":
            return data.astype(np.float32), gw.GGMLQuantizationType.F32
        elif self.output_type == "f16":
            return data.astype(np.float16), gw.GGMLQuantizationType.F16
        elif self.output_type == "q8_0":
            if not self._should_quantize(tensor_name):
                logger.debug(f"Keeping {tensor_name} in F16 (not quantizing)")
                return data.astype(np.float16), gw.GGMLQuantizationType.F16
            data = data.astype(np.float32)
            try:
                quantized = quant.quantize(data, gw.GGMLQuantizationType.Q8_0)
                return quantized, gw.GGMLQuantizationType.Q8_0
            except Exception as e:
                logger.warning(f"Q8_0 quantization failed for {tensor_name}: {e}, falling back to F16")
                return data.astype(np.float16), gw.GGMLQuantizationType.F16
        else:
            return data.astype(np.float16), gw.GGMLQuantizationType.F16
    def _load_tokenizer(self) -> tuple[list[str], list[int], list[str]]:
        vocab_path = self.input_dir / "vocab.json"
        merges_path = self.input_dir / "merges.txt"
        tokenizer_config_path = self.input_dir / "tokenizer_config.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        added_tokens = {}
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                tokenizer_config = json.load(f)
            added_tokens = tokenizer_config.get("added_tokens_decoder", {})
        max_id = max(vocab_dict.values())
        for token_id in added_tokens.keys():
            max_id = max(max_id, int(token_id))
        tokens: list[str | None] = [None] * (max_id + 1)
        toktypes = [gt.TokenType.NORMAL] * (max_id + 1)
        for token, token_id in vocab_dict.items():
            tokens[token_id] = token
        for token_id_str, token_info in added_tokens.items():
            token_id = int(token_id_str)
            tokens[token_id] = token_info["content"]
            toktypes[token_id] = gt.TokenType.USER_DEFINED
        for token_id, token in enumerate(tokens):
            if token is None:
                tokens[token_id] = f"<|unused-{token_id}|>"
                toktypes[token_id] = gt.TokenType.UNUSED
        merges = []
        if merges_path.exists():
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        merges.append(line)
        return [token for token in tokens if token is not None], toktypes, merges
    def convert(self) -> None:
        logger.info(f"Converting {self.model_name} to GGUF format")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Output type: {self.output_type}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        arch = "pig"
        writer = gw.GGUFWriter(path=None, arch=arch)
        self._add_metadata(writer)
        self._add_tokenizer(writer)
        tensor_count = 0
        skipped_count = 0
        logger.info("Processing tensors...")
        for hf_name, tensor in tqdm(list(self._get_tensors()), desc="Converting"):
            ggml_name = self._map_tensor_name(hf_name)
            if ggml_name is None:
                logger.warning(f"Skipping unmapped tensor: {hf_name}")
                skipped_count += 1
                continue
            data, dtype = self._convert_dtype(tensor, ggml_name)
            writer.add_tensor(ggml_name, data, raw_dtype=dtype)
            tensor_count += 1
            logger.debug(f"  {hf_name} -> {ggml_name} [{dtype.name}] {data.shape}")
        logger.info(f"Converted {tensor_count} tensors, skipped {skipped_count}")
        logger.info(f"Writing GGUF file to {self.output_path}")
        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        logger.info("Conversion complete!")
    def _add_metadata(self, writer: gw.GGUFWriter) -> None:
        arch = "cantonese"
        writer.add_name(self.model_name)
        if self.output_type == "f32":
            ftype = "F32"
        elif self.output_type == "f16":
            ftype = "F16"
        elif self.output_type == "q8_0":
            ftype = "Q8_0"
        else:
            ftype = "F16"
        writer.add_string("general.file_type", ftype)
        writer.add_quantization_version(gt.GGML_QUANT_VERSION)
        writer.add_string(f"{arch}.tokenizer_type", self.tokenizer_type)
        writer.add_string(f"{arch}.model_size", self.tts_model_size)
        writer.add_string(f"{arch}.model_type", self.tts_model_type)
        writer.add_uint32(f"{arch}.num_code_groups", self.num_code_groups)
        writer.add_uint32(f"{arch}.talker.embedding_length", self.hidden_size)
        writer.add_uint32(f"{arch}.talker.feed_forward_length", self.intermediate_size)
        writer.add_uint32(f"{arch}.talker.block_count", self.num_hidden_layers)
        writer.add_uint32(f"{arch}.talker.attention.head_count", self.num_attention_heads)
        writer.add_uint32(f"{arch}.talker.attention.head_count_kv", self.num_kv_heads)
        writer.add_uint32(f"{arch}.talker.attention.key_length", self.head_dim)
        writer.add_uint32(f"{arch}.talker.vocab_size", self.vocab_size)
        writer.add_uint32(f"{arch}.talker.text_vocab_size", self.text_vocab_size)
        writer.add_uint32(f"{arch}.talker.text_hidden_size", self.text_hidden_size)
        writer.add_uint32(f"{arch}.talker.context_length", self.context_length)
        writer.add_float32(f"{arch}.talker.rope.freq_base", float(self.rope_theta))
        writer.add_float32(f"{arch}.talker.attention.layer_norm_rms_epsilon", float(self.rms_norm_eps))
        writer.add_uint32(f"{arch}.talker.position_id_per_seconds", self.position_id_per_seconds)
        writer.add_array(f"{arch}.talker.rope.mrope_section", self.mrope_section)
        writer.add_bool(f"{arch}.talker.mrope_interleaved", self.mrope_interleaved)
        writer.add_uint32(f"{arch}.code_pred.embedding_length", self.code_predictor_hidden_size)
        writer.add_uint32(f"{arch}.code_pred.feed_forward_length", self.code_predictor_intermediate_size)
        writer.add_uint32(f"{arch}.code_pred.block_count", self.code_predictor_num_layers)
        writer.add_uint32(f"{arch}.code_pred.attention.head_count", self.code_predictor_num_attention_heads)
        writer.add_uint32(f"{arch}.code_pred.attention.head_count_kv", self.code_predictor_num_kv_heads)
        writer.add_uint32(f"{arch}.code_pred.attention.key_length", self.code_predictor_head_dim)
        writer.add_uint32(f"{arch}.code_pred.vocab_size", self.code_predictor_vocab_size)
        writer.add_uint32(f"{arch}.code_pred.context_length", self.code_predictor_context_length)
        writer.add_float32(f"{arch}.code_pred.rope.freq_base", self.code_predictor_rope_theta)
        writer.add_float32(
            f"{arch}.code_pred.attention.layer_norm_rms_epsilon", self.code_predictor_rms_norm_eps
        )
        if self.has_speaker_encoder_config:
            writer.add_uint32(f"{arch}.spk_enc.embedding_length", self.speaker_enc_dim)
            writer.add_uint32(f"{arch}.spk_enc.sample_rate", self.speaker_sample_rate)
        writer.add_uint32(f"{arch}.codec.pad_id", self.codec_pad_id)
        writer.add_uint32(f"{arch}.codec.bos_id", self.codec_bos_id)
        writer.add_uint32(f"{arch}.codec.eos_id", self.codec_eos_id)
        writer.add_uint32(f"{arch}.codec.think_id", self.codec_think_id)
        writer.add_uint32(f"{arch}.codec.nothink_id", self.codec_nothink_id)
        writer.add_uint32(f"{arch}.codec.think_bos_id", self.codec_think_bos_id)
        writer.add_uint32(f"{arch}.codec.think_eos_id", self.codec_think_eos_id)
        language_names = list(self.codec_language_id.keys())
        language_ids = [int(self.codec_language_id[name]) for name in language_names]
        writer.add_array(f"{arch}.codec.language_names", language_names)
        writer.add_array(f"{arch}.codec.language_ids", language_ids)
        writer.add_uint32(f"{arch}.text.im_start_id", self.im_start_token_id)
        writer.add_uint32(f"{arch}.text.im_end_id", self.im_end_token_id)
        writer.add_uint32(f"{arch}.text.tts_pad_id", self.tts_pad_token_id)
        writer.add_uint32(f"{arch}.text.tts_bos_id", self.tts_bos_token_id)
        writer.add_uint32(f"{arch}.text.tts_eos_id", self.tts_eos_token_id)
        speaker_items: list[tuple[str, int, str]] = []
        raw_dialect_map = self.config.get("talker_config", {}).get("spk_is_dialect", {})
        dialect_map = raw_dialect_map if isinstance(raw_dialect_map, dict) else {}
        for k, v in self.spk_id_map.items():
            name = str(k).strip()
            if not name:
                continue
            try:
                spk_id = int(v)
            except (TypeError, ValueError):
                continue
            if spk_id < 0:
                continue
            dialect = dialect_map.get(k) or ""
            dialect = dialect if isinstance(dialect, str) else ""
            speaker_items.append((name, spk_id, dialect))
        if speaker_items:
            writer.add_array(f"{arch}.codec.speaker_names", [name for name, _, _ in speaker_items])
            writer.add_array(f"{arch}.codec.speaker_ids", [spk_id for _, spk_id, _ in speaker_items])
            writer.add_array(f"{arch}.codec.speaker_dialects", [dialect for _, _, dialect in speaker_items])
        gen_cfg = self.generation_config
        if gen_cfg:
            if "do_sample" in gen_cfg:
                writer.add_bool("generation.do_sample", bool(gen_cfg["do_sample"]))
            if "top_k" in gen_cfg:
                writer.add_uint32("generation.top_k", int(gen_cfg["top_k"]))
            if "top_p" in gen_cfg:
                writer.add_float32("generation.top_p", float(gen_cfg["top_p"]))
            if "temperature" in gen_cfg:
                writer.add_float32("generation.temperature", float(gen_cfg["temperature"]))
            if "repetition_penalty" in gen_cfg:
                writer.add_float32("generation.repetition_penalty", float(gen_cfg["repetition_penalty"]))
            if "subtalker_dosample" in gen_cfg:
                writer.add_bool("generation.subtalker_do_sample", bool(gen_cfg["subtalker_dosample"]))
            if "subtalker_top_k" in gen_cfg:
                writer.add_uint32("generation.subtalker_top_k", int(gen_cfg["subtalker_top_k"]))
            if "subtalker_top_p" in gen_cfg:
                writer.add_float32("generation.subtalker_top_p", float(gen_cfg["subtalker_top_p"]))
            if "subtalker_temperature" in gen_cfg:
                writer.add_float32("generation.subtalker_temperature", float(gen_cfg["subtalker_temperature"]))
            if "max_new_tokens" in gen_cfg:
                writer.add_uint32("generation.max_new_tokens", int(gen_cfg["max_new_tokens"]))
        logger.info("Added model metadata")
    def _add_tokenizer(self, writer: gw.GGUFWriter) -> None:
        tokens, toktypes, merges = self._load_tokenizer()
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(tokens)
        writer.add_token_types(toktypes)
        if merges:
            writer.add_token_merges(merges)
        writer.add_uint32("tokenizer.ggml.eos_token_id", 151643)
        logger.info(f"Added tokenizer with {len(tokens)} tokens and {len(merges)} merges")
def main():
    parser = argparse.ArgumentParser(
        description="Convert TTS talker checkpoints to GGUF format"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to HuggingFace model directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["f16", "f32", "q8_0"],
        default="f16",
        help="Output data type (default: f16). q8_0 provides ~50%% size reduction, q4_k provides ~70%% size reduction."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    converter = CantoneseTTSConverter(
        input_dir=args.input,
        output_path=args.output,
        output_type=args.type,
    )
    converter.convert()
if __name__ == "__main__":
    main()
"""

with open(filename, "w") as f:
    f.write(content)

os.system(f"python {filename}  --input cantonese-1.2-0.6b --output cantonese-1.2-0.6b-f16.gguf --type f16")
