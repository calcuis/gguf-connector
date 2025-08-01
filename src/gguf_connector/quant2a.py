import torch # need torch to work; pip install torch
from .quant import dequantize as gq
from .reader import GGMLQuantizationType, GGML_QUANT_SIZES
from tqdm import tqdm
TORCH_COMPATIBLE_QTYPES = {None, GGMLQuantizationType.F32,
    GGMLQuantizationType.F16}
def is_torch_compatible(tensor):
    return tensor is None or getattr(tensor, 'tensor_type', None
        ) in TORCH_COMPATIBLE_QTYPES
def is_quantized(tensor):
    return not is_torch_compatible(tensor)
def dequantize(data, qtype, oshape, dtype=None):
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)
def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, 'tensor_type', None)
    oshape = getattr(tensor, 'tensor_shape', tensor.shape)
    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == 'target' else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(
            dtype)
    else:
        tqdm.write(
            f'Pushing back to numpy dequant (cpu offload) for qtype: {qtype}')
        new = gq(tensor.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(tensor.device, dtype=dtype)
def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1
        )
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d = blocks[:, :2].view(torch.float16).to(dtype)
    x = blocks[:, 2:].view(torch.int8)
    return d * x
def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d = blocks[:, :2].view(torch.float16).to(dtype)
    qh = blocks[:, 2:6]
    qs = blocks[:, 6:]
    qh = to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype
        =torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4
        ], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 15).reshape(n_blocks, -1)
    qs = (ql | qh << 4).to(torch.int8) - 16
    return d * qs
def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d = blocks[:, :2].view(torch.float16).to(dtype)
    qs = blocks[:, 2:]
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0,
        4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs
def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d = blocks[:, :2].view(torch.float16).to(dtype)
    m = blocks[:, 2:4].view(torch.float16).to(dtype)
    qh = blocks[:, 4:8]
    qs = blocks[:, 8:]
    qh = to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device,
        dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0,
        4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 15).reshape((n_blocks, -1))
    qs = ql | qh << 4
    return d * qs + m
def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d = blocks[:, :2].view(torch.float16).to(dtype)
    m = blocks[:, 2:4].view(torch.float16).to(dtype)
    qs = blocks[:, 4:]
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0,
        4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 15).reshape(n_blocks, -1)
    return d * qs + m
QK_K = 256
K_SCALE = 12
def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)
def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K //
        16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d
        .device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 15).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6],
        device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 3).reshape((n_blocks, -1, 32))
    q = (ql | qh << 4).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))
def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4],
        device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6
        ], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = lscales & 15 | (hscales & 3) << 4
    scales = scales.to(torch.int8) - 32
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6],
        device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in
        range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = qh.reshape((n_blocks, 16, QK_K // 16)) & 1 ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q).reshape((n_blocks, QK_K))
def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 15)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8
        ).reshape((1, 1, 4, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> shift & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))
def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 63, m_d & 15 | d >> 2 & 48], dim=-1)
    min = torch.cat([m & 63, m_d >> 4 | m >> 2 & 48], dim=-1)
    return sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8))
def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE, QK_K // 8
        )
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d
        .device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in
        range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 15).reshape((n_blocks, -1, 32))
    qh = (qh & 1).reshape((n_blocks, -1, 32))
    q = ql | qh << 4
    return (d * q - dm).reshape((n_blocks, QK_K))
def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d
        .device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))
def dequantize_blocks_TQ2_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    qs, d = split_block_dims(blocks, QK_K // 4)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6],
        device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs & 3).reshape((n_blocks, -1)) - 1
    return d * qs
def dequantize_blocks_TQ1_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    qs, qh, d = split_block_dims(blocks, (QK_K - 4 * QK_K // 64) // 5, QK_K //
        64)
    d = d.view(torch.float16).to(dtype)
    qs0, qs1 = qs[..., :32], qs[..., 32:]
    qs0 = qs0.reshape((n_blocks, -1, 1, 32)) * torch.tensor([1, 3, 9, 27, 
        81], device=d.device, dtype=torch.uint8).reshape((1, 1, 5, 1))
    qs0 = qs0.reshape((n_blocks, -1))
    qs1 = qs1.reshape((n_blocks, -1, 1, 16)) * torch.tensor([1, 3, 9, 27, 
        81], device=d.device, dtype=torch.uint8).reshape((1, 1, 5, 1))
    qs1 = qs1.reshape((n_blocks, -1))
    qh = qh.reshape((n_blocks, -1, 1, 4)) * torch.tensor([1, 3, 9, 27],
        device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = qh.reshape((n_blocks, -1))
    qs = torch.cat([qs0, qs1, qh], dim=-1)
    qs = (qs * 3 >> 8) - 1
    return d * qs
def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13,
        25, 38, 53, 69, 89, 113], dtype=torch.float32, device=blocks.device)
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0,
        4], device=blocks.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], 16), 2, qs)
    qs = qs.squeeze(-1).to(dtype)
    return d * qs
def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13,
        25, 38, 53, 69, 89, 113], dtype=torch.float32, device=blocks.device)
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = scales_h.view(torch.int16)
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> torch.tensor([0, 4],
        device=blocks.device, dtype=torch.uint8).reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, 1, -1)) >> torch.tensor([(2 * i) for
        i in range(QK_K // 32)], device=blocks.device, dtype=torch.uint8
        ).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 15
    scales_h = scales_h.reshape((n_blocks, -1)) & 3
    scales = (scales_l | scales_h << 4) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    shifts_q = torch.tensor([0, 4], device=blocks.device, dtype=torch.uint8
        ).reshape(1, 1, 2, 1)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shifts_q
    qs = (qs & 15).reshape((n_blocks, -1, 32)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], qs.shape[2],
        16), 3, qs)
    qs = qs.squeeze(-1).to(dtype)
    return (dl * qs).reshape(n_blocks, -1)
dequantize_functions = {GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0, GGMLQuantizationType
    .Q5_1: dequantize_blocks_Q5_1, GGMLQuantizationType.Q5_0:
    dequantize_blocks_Q5_0, GGMLQuantizationType.Q4_1:
    dequantize_blocks_Q4_1, GGMLQuantizationType.Q4_0:
    dequantize_blocks_Q4_0, GGMLQuantizationType.Q6_K:
    dequantize_blocks_Q6_K, GGMLQuantizationType.Q5_K:
    dequantize_blocks_Q5_K, GGMLQuantizationType.Q4_K:
    dequantize_blocks_Q4_K, GGMLQuantizationType.Q3_K:
    dequantize_blocks_Q3_K, GGMLQuantizationType.Q2_K:
    dequantize_blocks_Q2_K, GGMLQuantizationType.TQ2_0:
    dequantize_blocks_TQ2_0, GGMLQuantizationType.TQ1_0:
    dequantize_blocks_TQ1_0, GGMLQuantizationType.IQ4_NL:
    dequantize_blocks_IQ4_NL, GGMLQuantizationType.IQ4_XS:
    dequantize_blocks_IQ4_XS}
