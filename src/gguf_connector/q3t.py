import os
os.system("git clone https://huggingface.co/callgg/tts-tokenizer-12hz")

filename = "q3tts.py"
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
class Qwen3TTSTokenizerConverter:
    TENSOR_MAP = {
        "encoder.downsample.conv.weight": "tok_enc.downsample.weight",
        "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight": "tok_enc.vq_acoustic.input_proj.weight",
        "encoder.quantizer.acoustic_residual_vector_quantizer.output_proj.weight": "tok_enc.vq_acoustic.output_proj.weight",
        "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight": "tok_enc.vq_semantic.input_proj.weight",
        "encoder.quantizer.semantic_residual_vector_quantizer.output_proj.weight": "tok_enc.vq_semantic.output_proj.weight",
        "decoder.pre_conv.conv.bias": "tok_dec.pre_conv.bias",
        "decoder.pre_conv.conv.weight": "tok_dec.pre_conv.weight",
        "decoder.pre_transformer.input_proj.bias": "tok_dec.pre_tfm.input_proj.bias",
        "decoder.pre_transformer.input_proj.weight": "tok_dec.pre_tfm.input_proj.weight",
        "decoder.pre_transformer.output_proj.bias": "tok_dec.pre_tfm.output_proj.bias",
        "decoder.pre_transformer.output_proj.weight": "tok_dec.pre_tfm.output_proj.weight",
        "decoder.pre_transformer.norm.weight": "tok_dec.pre_tfm.norm.weight",
        "decoder.quantizer.rvq_first.output_proj.weight": "tok_dec.vq_first.output_proj.weight",
        "decoder.quantizer.rvq_rest.output_proj.weight": "tok_dec.vq_rest.output_proj.weight",
        "decoder.decoder.0.conv.weight": "tok_dec.dec.0.conv.weight",
        "decoder.decoder.0.conv.bias": "tok_dec.dec.0.conv.bias",
        "decoder.decoder.5.alpha": "tok_dec.dec.5.snake.alpha",
        "decoder.decoder.5.beta": "tok_dec.dec.5.snake.beta",
        "decoder.decoder.6.conv.weight": "tok_dec.dec.6.conv.weight",
        "decoder.decoder.6.conv.bias": "tok_dec.dec.6.conv.bias",
    }
    ENCODER_PATTERNS = [
        (r"encoder\.encoder\.layers\.(\d+)\.conv\.weight", "tok_enc.conv.{}.weight"),
        (r"encoder\.encoder\.layers\.(\d+)\.conv\.bias", "tok_enc.conv.{}.bias"),
        (r"encoder\.encoder\.layers\.(\d+)\.block\.(\d+)\.conv\.weight", "tok_enc.res.{}.blk.{}.weight"),
        (r"encoder\.encoder\.layers\.(\d+)\.block\.(\d+)\.conv\.bias", "tok_enc.res.{}.blk.{}.bias"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.input_layernorm\.weight", "tok_enc.blk.{}.attn_norm.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.input_layernorm\.bias", "tok_enc.blk.{}.attn_norm.bias"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.post_attention_layernorm\.weight", "tok_enc.blk.{}.ffn_norm.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.post_attention_layernorm\.bias", "tok_enc.blk.{}.ffn_norm.bias"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.q_proj\.weight", "tok_enc.blk.{}.attn_q.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.k_proj\.weight", "tok_enc.blk.{}.attn_k.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.v_proj\.weight", "tok_enc.blk.{}.attn_v.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn\.o_proj\.weight", "tok_enc.blk.{}.attn_output.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.self_attn_layer_scale\.scale", "tok_enc.blk.{}.attn_scale"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp\.fc1\.weight", "tok_enc.blk.{}.ffn_up.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp\.fc2\.weight", "tok_enc.blk.{}.ffn_down.weight"),
        (r"encoder\.encoder_transformer\.layers\.(\d+)\.mlp_layer_scale\.scale", "tok_enc.blk.{}.ffn_scale"),
        (r"encoder\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.embed_sum", "tok_enc.vq_acoustic.{}.codebook"),
        (r"encoder\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.cluster_usage", "tok_enc.vq_acoustic.{}.usage"),
        (r"encoder\.quantizer\.acoustic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.initialized", "tok_enc.vq_acoustic.{}.initialized"),
        (r"encoder\.quantizer\.semantic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.embed_sum", "tok_enc.vq_semantic.{}.codebook"),
        (r"encoder\.quantizer\.semantic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.cluster_usage", "tok_enc.vq_semantic.{}.usage"),
        (r"encoder\.quantizer\.semantic_residual_vector_quantizer\.layers\.(\d+)\.codebook\.initialized", "tok_enc.vq_semantic.{}.initialized"),
    ]
    DECODER_PATTERNS = [
        (r"decoder\.decoder\.(\d+)\.block\.0\.alpha", "tok_dec.dec.{}.snake.alpha"),
        (r"decoder\.decoder\.(\d+)\.block\.0\.beta", "tok_dec.dec.{}.snake.beta"),
        (r"decoder\.decoder\.(\d+)\.block\.1\.conv\.weight", "tok_dec.dec.{}.conv_t.weight"),
        (r"decoder\.decoder\.(\d+)\.block\.1\.conv\.bias", "tok_dec.dec.{}.conv_t.bias"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act1\.alpha", "tok_dec.dec.{}.res.{}.act1.alpha"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act1\.beta", "tok_dec.dec.{}.res.{}.act1.beta"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act2\.alpha", "tok_dec.dec.{}.res.{}.act2.alpha"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.act2\.beta", "tok_dec.dec.{}.res.{}.act2.beta"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv1\.conv\.weight", "tok_dec.dec.{}.res.{}.conv1.weight"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv1\.conv\.bias", "tok_dec.dec.{}.res.{}.conv1.bias"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv2\.conv\.weight", "tok_dec.dec.{}.res.{}.conv2.weight"),
        (r"decoder\.decoder\.(\d+)\.block\.(\d+)\.conv2\.conv\.bias", "tok_dec.dec.{}.res.{}.conv2.bias"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.input_layernorm\.weight", "tok_dec.pre_tfm.blk.{}.attn_norm.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.post_attention_layernorm\.weight", "tok_dec.pre_tfm.blk.{}.ffn_norm.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.q_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_q.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.k_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_k.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.v_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_v.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn\.o_proj\.weight", "tok_dec.pre_tfm.blk.{}.attn_output.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.self_attn_layer_scale\.scale", "tok_dec.pre_tfm.blk.{}.attn_scale"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.gate_proj\.weight", "tok_dec.pre_tfm.blk.{}.ffn_gate.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.up_proj\.weight", "tok_dec.pre_tfm.blk.{}.ffn_up.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp\.down_proj\.weight", "tok_dec.pre_tfm.blk.{}.ffn_down.weight"),
        (r"decoder\.pre_transformer\.layers\.(\d+)\.mlp_layer_scale\.scale", "tok_dec.pre_tfm.blk.{}.ffn_scale"),
        (r"decoder\.quantizer\.rvq_first\.vq\.layers\.(\d+)\._codebook\.embedding_sum", "tok_dec.vq_first.{}.codebook"),
        (r"decoder\.quantizer\.rvq_first\.vq\.layers\.(\d+)\._codebook\.cluster_usage", "tok_dec.vq_first.{}.usage"),
        (r"decoder\.quantizer\.rvq_rest\.vq\.layers\.(\d+)\._codebook\.embedding_sum", "tok_dec.vq_rest.{}.codebook"),
        (r"decoder\.quantizer\.rvq_rest\.vq\.layers\.(\d+)\._codebook\.cluster_usage", "tok_dec.vq_rest.{}.usage"),
        (r"decoder\.upsample\.(\d+)\.0\.conv\.weight", "tok_dec.upsample.{}.conv.weight"),
        (r"decoder\.upsample\.(\d+)\.0\.conv\.bias", "tok_dec.upsample.{}.conv.bias"),
        (r"decoder\.upsample\.(\d+)\.1\.dwconv\.conv\.weight", "tok_dec.upsample.{}.dwconv.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.dwconv\.conv\.bias", "tok_dec.upsample.{}.dwconv.bias"),
        (r"decoder\.upsample\.(\d+)\.1\.gamma", "tok_dec.upsample.{}.gamma"),
        (r"decoder\.upsample\.(\d+)\.1\.norm\.weight", "tok_dec.upsample.{}.norm.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.norm\.bias", "tok_dec.upsample.{}.norm.bias"),
        (r"decoder\.upsample\.(\d+)\.1\.pwconv1\.weight", "tok_dec.upsample.{}.pwconv1.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.pwconv1\.bias", "tok_dec.upsample.{}.pwconv1.bias"),
        (r"decoder\.upsample\.(\d+)\.1\.pwconv2\.weight", "tok_dec.upsample.{}.pwconv2.weight"),
        (r"decoder\.upsample\.(\d+)\.1\.pwconv2\.bias", "tok_dec.upsample.{}.pwconv2.bias"),
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
        tokenizer_config_path = self.input_dir / "speech_tokenizer" / "config.json"
        config_path = tokenizer_config_path if tokenizer_config_path.exists() else self.input_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    def _extract_params(self) -> None:
        encoder_config = self.config.get("encoder_config", {})
        decoder_config = self.config.get("decoder_config", {})
        self.encoder_hidden_size = encoder_config.get("hidden_size", 512)
        self.encoder_num_layers = encoder_config.get("num_hidden_layers", 8)
        self.encoder_num_heads = encoder_config.get("num_attention_heads", 8)
        self.encoder_num_quantizers = encoder_config.get("num_quantizers", 32)
        self.encoder_valid_quantizers = self.config.get("encoder_valid_num_quantizers", 16)
        self.encoder_codebook_size = encoder_config.get("codebook_size", 2048)
        self.encoder_codebook_dim = encoder_config.get("codebook_dim", 256)
        self.decoder_hidden_size = decoder_config.get("hidden_size", 512)
        self.decoder_num_layers = decoder_config.get("num_hidden_layers", 8)
        self.decoder_num_heads = decoder_config.get("num_attention_heads", 16)
        self.decoder_latent_dim = decoder_config.get("latent_dim", 1024)
        self.decoder_codebook_size = decoder_config.get("codebook_size", 2048)
        self.decoder_codebook_dim = decoder_config.get("codebook_dim", 512)
        self.decoder_num_quantizers = decoder_config.get("num_quantizers", 16)
        self.decoder_semantic_codebook_size = decoder_config.get("semantic_codebook_size", 4096)
        self.sample_rate = self.config.get("input_sample_rate", 24000)
        self.frame_rate = encoder_config.get("_frame_rate", 12.5)
        self.upsample_rates = decoder_config.get("upsample_rates", [8, 5, 4, 3])
    def _map_tensor_name(self, hf_name: str) -> str | None:
        if hf_name in self.TENSOR_MAP:
            return self.TENSOR_MAP[hf_name]
        for pattern, template in self.ENCODER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    return template.format(groups[0])
                elif len(groups) == 2:
                    return template.format(groups[0], groups[1])
                return None
        for pattern, template in self.DECODER_PATTERNS:
            match = re.match(pattern, hf_name)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    return template.format(groups[0])
                elif len(groups) == 2:
                    return template.format(groups[0], str(int(groups[1]) - 2))
                return None
        return None
    def _get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        tokenizer_dir = self.input_dir / "speech_tokenizer"
        if tokenizer_dir.exists():
            safetensor_files = list(tokenizer_dir.glob("*.safetensors"))
        else:
            safetensor_files = list(self.input_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {self.input_dir} or {tokenizer_dir}")
        for sf_path in sorted(safetensor_files):
            logger.info(f"Loading tensors from {sf_path.name}")
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)
    def _convert_dtype(self, tensor: torch.Tensor, tensor_name: str = "") -> tuple[np.ndarray, gw.GGMLQuantizationType]:
        if tensor.dtype == torch.bfloat16:
            data = tensor.float().numpy()
        else:
            data = tensor.numpy()
        if tensor_name == "tok_dec.dec.6.conv.weight" and len(data.shape) == 3 and data.shape[0] == 1:
            data = data.squeeze(0)
        n_dims = len(data.shape)
        if tensor_name.startswith("tok_enc.vq_") or tensor_name.startswith("tok_dec.vq_"):
            return data.astype(np.float32), gw.GGMLQuantizationType.F32
        if n_dims <= 1:
            return data.astype(np.float32), gw.GGMLQuantizationType.F32
        if self.output_type == "f32":
            return data.astype(np.float32), gw.GGMLQuantizationType.F32
        elif self.output_type == "f16":
            return data.astype(np.float16), gw.GGMLQuantizationType.F16
        elif self.output_type == "q8_0":
            if any(x in tensor_name for x in ["_norm", "norm.", "scale", "alpha", "beta"]):
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
    def convert(self) -> None:
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Output type: {self.output_type}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        arch = "pig"
        writer = gw.GGUFWriter(path=None, arch=arch)
        self._add_metadata(writer)
        tensor_count = 0
        skipped_count = 0
        skipped_tensors = []
        codebook_pairs: dict[str, dict[str, torch.Tensor]] = {}
        encoder_acoustic_kept = self.encoder_valid_quantizers - self.config.get("encoder_config", {}).get("num_semantic_quantizers", 1)
        logger.info("Processing tensors...")
        all_tensors = list(self._get_tensors())
        for hf_name, tensor in all_tensors:
            if "embedding_sum" in hf_name or "embed_sum" in hf_name:
                sum_name = "embedding_sum" if "embedding_sum" in hf_name else "embed_sum"
                base_name = hf_name.replace(sum_name, "")
                if base_name not in codebook_pairs:
                    codebook_pairs[base_name] = {}
                codebook_pairs[base_name]["embedding_sum"] = tensor
            elif "cluster_usage" in hf_name:
                base_name = hf_name.replace("cluster_usage", "")
                if base_name not in codebook_pairs:
                    codebook_pairs[base_name] = {}
                codebook_pairs[base_name]["cluster_usage"] = tensor
        for hf_name, tensor in tqdm(all_tensors, desc="Converting"):
            ggml_name = self._map_tensor_name(hf_name)
            if ggml_name is None:
                skipped_tensors.append(hf_name)
                skipped_count += 1
                continue
            if ggml_name.endswith(".initialized"):
                skipped_count += 1
                continue
            acoustic_match = re.match(r"tok_enc\.vq_acoustic\.(\d+)\.codebook$", ggml_name)
            if acoustic_match and int(acoustic_match.group(1)) >= encoder_acoustic_kept:
                skipped_count += 1
                continue
            if "cluster_usage" in hf_name:
                skipped_count += 1
                continue
            if "embedding_sum" in hf_name or "embed_sum" in hf_name:
                sum_name = "embedding_sum" if "embedding_sum" in hf_name else "embed_sum"
                base_name = hf_name.replace(sum_name, "")
                if base_name in codebook_pairs and "cluster_usage" in codebook_pairs[base_name]:
                    embedding_sum = codebook_pairs[base_name]["embedding_sum"]
                    cluster_usage = codebook_pairs[base_name]["cluster_usage"]
                    tensor = embedding_sum / cluster_usage.clamp(min=1e-5).unsqueeze(1)
                    logger.debug(f"  Computing codebook from embedding_sum/cluster_usage for {hf_name}")
            data, dtype = self._convert_dtype(tensor, ggml_name)
            writer.add_tensor(ggml_name, data, raw_dtype=dtype)
            tensor_count += 1
            logger.debug(f"  {hf_name} -> {ggml_name} [{dtype.name}] {data.shape}")
        logger.info(f"Converted {tensor_count} tensors, skipped {skipped_count}")
        if skipped_tensors and logger.level <= logging.DEBUG:
            logger.debug("Skipped tensors:")
            for t in skipped_tensors:
                logger.debug(f"  {t}")
        logger.info(f"Writing GGUF file to {self.output_path}")
        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        logger.info("Conversion complete!")
    def _add_metadata(self, writer: gw.GGUFWriter) -> None:
        arch = "qwen3-tts-tokenizer"
        if self.output_type == "f32":
            ftype = 0
        elif self.output_type == "f16":
            ftype = 1
        elif self.output_type == "q8_0":
            ftype = 7
        else:
            ftype = 1
        writer.add_uint32("general.file_type", ftype)
        writer.add_quantization_version(gt.GGML_QUANT_VERSION)
        writer.add_uint32(f"{arch}.input_sample_rate", self.config.get("input_sample_rate", self.sample_rate))
        writer.add_uint32(f"{arch}.output_sample_rate", self.config.get("output_sample_rate", self.sample_rate))
        writer.add_uint32(f"{arch}.decode_upsample_rate", self.config.get("decode_upsample_rate", 1920))
        writer.add_uint32(f"{arch}.encode_downsample_rate", self.config.get("encode_downsample_rate", 1920))
        writer.add_uint32(f"{arch}.encoder_valid_num_quantizers", self.encoder_valid_quantizers)
        enc_cfg = self.config.get("encoder_config", {})
        writer.add_uint32(f"{arch}.encoder.num_filters", enc_cfg.get("num_filters", 32))
        writer.add_uint32(f"{arch}.encoder.kernel_size", enc_cfg.get("kernel_size", 7))
        writer.add_uint32(f"{arch}.encoder.last_kernel_size", enc_cfg.get("last_kernel_size", 7))
        writer.add_uint32(f"{arch}.encoder.residual_kernel_size", enc_cfg.get("residual_kernel_size", 3))
        writer.add_uint32(f"{arch}.encoder.num_residual_layers", enc_cfg.get("num_residual_layers", 1))
        writer.add_uint32(f"{arch}.encoder.dilation_growth_rate", enc_cfg.get("dilation_growth_rate", 2))
        writer.add_uint32(f"{arch}.encoder.compress", enc_cfg.get("compress", 2))
        writer.add_array(f"{arch}.encoder.upsampling_ratios", enc_cfg.get("upsampling_ratios", [8, 5, 4, 3]))
        writer.add_uint32(f"{arch}.encoder.hidden_size", self.encoder_hidden_size)
        writer.add_uint32(f"{arch}.encoder.intermediate_size", enc_cfg.get("intermediate_size", 2048))
        writer.add_uint32(f"{arch}.encoder.head_dim", enc_cfg.get("head_dim", 64))
        writer.add_uint32(f"{arch}.encoder.num_attention_heads", self.encoder_num_heads)
        writer.add_uint32(f"{arch}.encoder.num_key_value_heads", enc_cfg.get("num_key_value_heads", self.encoder_num_heads))
        writer.add_uint32(f"{arch}.encoder.num_hidden_layers", self.encoder_num_layers)
        writer.add_float32(f"{arch}.encoder.norm_eps", enc_cfg.get("norm_eps", enc_cfg.get("rms_norm_eps", 1e-5)))
        writer.add_float32(f"{arch}.encoder.rope_theta", float(enc_cfg.get("rope_theta", 10000.0)))
        writer.add_float32(
            f"{arch}.encoder.layer_scale_initial_scale",
            enc_cfg.get("layer_scale_initial_scale", 0.01),
        )
        writer.add_uint32(f"{arch}.encoder.num_quantizers", self.encoder_num_quantizers)
        writer.add_uint32(f"{arch}.encoder.codebook_dim", self.encoder_codebook_dim)
        writer.add_uint32(f"{arch}.encoder.codebook_size", self.encoder_codebook_size)
        writer.add_uint32(
            f"{arch}.encoder.num_semantic_quantizers",
            enc_cfg.get("num_semantic_quantizers", 1),
        )
        writer.add_uint32(
            f"{arch}.encoder.vector_quantization_hidden_dim",
            enc_cfg.get("vector_quantization_hidden_dimension", 256),
        )
        writer.add_uint32(f"{arch}.decoder.hidden_size", self.decoder_hidden_size)
        writer.add_uint32(f"{arch}.decoder.latent_dim", self.decoder_latent_dim)
        writer.add_uint32(f"{arch}.decoder.codebook_dim", self.decoder_codebook_dim)
        writer.add_uint32(f"{arch}.decoder.codebook_size", self.decoder_codebook_size)
        writer.add_uint32(f"{arch}.decoder.decoder_dim", self.config.get("decoder_config", {}).get("decoder_dim", 1536))
        writer.add_uint32(f"{arch}.decoder.intermediate_size", self.config.get("decoder_config", {}).get("intermediate_size", 1024))
        writer.add_uint32(f"{arch}.decoder.head_dim", self.config.get("decoder_config", {}).get("head_dim", 64))
        writer.add_uint32(f"{arch}.decoder.num_attention_heads", self.decoder_num_heads)
        writer.add_uint32(f"{arch}.decoder.num_key_value_heads", self.config.get("decoder_config", {}).get("num_key_value_heads", self.decoder_num_heads))
        writer.add_uint32(f"{arch}.decoder.num_hidden_layers", self.decoder_num_layers)
        writer.add_uint32(f"{arch}.decoder.num_quantizers", self.decoder_num_quantizers)
        writer.add_uint32(
            f"{arch}.decoder.num_semantic_quantizers",
            self.config.get("decoder_config", {}).get("num_semantic_quantizers", 1),
        )
        writer.add_float32(f"{arch}.decoder.rms_norm_eps", self.config.get("decoder_config", {}).get("rms_norm_eps", 1e-5))
        writer.add_float32(f"{arch}.decoder.rope_theta", float(self.config.get("decoder_config", {}).get("rope_theta", 10000.0)))
        writer.add_uint32(f"{arch}.decoder.sliding_window", self.config.get("decoder_config", {}).get("sliding_window", 72))
        writer.add_float32(
            f"{arch}.decoder.layer_scale_initial_scale",
            self.config.get("decoder_config", {}).get("layer_scale_initial_scale", 0.01),
        )
        writer.add_uint32(
            f"{arch}.decoder.vector_quantization_hidden_dim",
            self.config.get("decoder_config", {}).get("vector_quantization_hidden_dimension", 512),
        )
        writer.add_uint32(f"{arch}.decoder.codebook_dim_internal", self.encoder_codebook_dim)
        # Upsample rates as array
        writer.add_array(f"{arch}.decoder.upsample_rates", self.upsample_rates)
        writer.add_array(
            f"{arch}.decoder.upsampling_ratios",
            self.config.get("decoder_config", {}).get("upsampling_ratios", [2, 2]),
        )
        logger.info("Added model metadata")
def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS-Tokenizer-12Hz model to GGUF format"
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
        help="Output data type (default: f16)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    converter = Qwen3TTSTokenizerConverter(
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

os.system(f"python {filename}  --input tts-tokenizer-12hz --output qwen-tokenizer-12hz-f16.gguf --type f16")
