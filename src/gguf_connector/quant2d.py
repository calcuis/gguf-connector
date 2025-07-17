
import torch # need torch to work; pip install torch
from .quant import dequantize as gq
from .reader import GGMLQuantizationType, GGML_QUANT_SIZES
from tqdm import tqdm

TORCH_COMPATIBLE_QTYPES = {None, GGMLQuantizationType.F32, GGMLQuantizationType.F16}

def is_torch_compatible(tensor):
    return tensor is None or getattr(tensor, 'tensor_type', None) in TORCH_COMPATIBLE_QTYPES

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
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:
        tqdm.write(f'Processing tensor: qtype: {qtype}, {oshape}') # slow mode
        new = gq(tensor.cpu().numpy(), qtype) # push to numpy for dequant (cpu offload)
        return torch.from_numpy(new).to(tensor.device, dtype=dtype)

# handle bf16
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

# s quant (legacy)
# 8-bit; w=q*block_scale
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d = blocks[:, :2].view(torch.float16).to(dtype)
    x = blocks[:, 2:].view(torch.int8)
    return x * d
# 4-bit; w=q*block_scale
def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d  = blocks[:,  :2].view(torch.float16).to(dtype)
    qs = blocks[:,  2:]
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs
# 4-bit; w=q*block_scale+block_min
def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d  = blocks[:,  :2].view(torch.float16).to(dtype)
    m  = blocks[:, 2:4].view(torch.float16).to(dtype)
    qs = blocks[:, 4: ]
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 15).reshape(n_blocks, -1)
    return d * qs + m

from .quant2b import to_uint32
# 5-bit; w=q*block_scale
def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d  = blocks[:,  :2].view(torch.float16).to(dtype)
    qh = blocks[:, 2:6]
    qs = blocks[:, 6: ]
    qh = to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 15).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs
# 5-bit; w=q*block_scale+block_min
def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d  = blocks[:,  :2].view(torch.float16).to(dtype)
    m  = blocks[:, 2:4].view(torch.float16).to(dtype)
    qh = blocks[:, 4:8]
    qs = blocks[:, 8: ]
    qh = to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 15).reshape((n_blocks, -1))
    qs = (ql | (qh << 4))
    return d * qs + m

# k quant
QK_K = 256
from .quant2b import split_block_dims
# 6-bit; w=q*block_scale(8-bit); 6.5625 bit/weight
def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 15).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6],
        device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 3).reshape((n_blocks, -1, 32))
    q = (ql | qh << 4).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))
# 3-bit; w=q*block_scale(6-bit); 3.4375 bit/weight
def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = lscales & 15 | (hscales & 3) << 4
    scales = scales.to(torch.int8) - 32
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = qh.reshape((n_blocks, 16, QK_K // 16)) & 1 ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q).reshape((n_blocks, QK_K))
# 2-bit; w=q*block_scale(4-bit)+block_min(4-bit); 2.625 bit/weight
def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 15)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> shift & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))

K_SCALE_SIZE = 12
from .quant2b import get_scale_min
# 5-bit; w=q*block_scale(6-bit)+block_min(6-bit); 5.5 bit/weight
def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 15).reshape((n_blocks, -1, 32))
    qh = (qh & 1).reshape((n_blocks, -1, 32))
    q = ql | qh << 4
    return (d * q - dm).reshape((n_blocks, QK_K))
# 4-bit; w=q*block_scale(6-bit)+block_min(6-bit); 4.5 bit/weight
def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))

# t quant
# 2-bit; runable, experimental/test purpose
def dequantize_blocks_TQ2_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    qs, d = split_block_dims(blocks, QK_K // 4)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6],
        device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs & 3).reshape((n_blocks, -1)) - 1
    return d * qs
# 1-bit; runable; for test purpose
def dequantize_blocks_TQ1_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    qs, qh, d = split_block_dims(blocks, (QK_K - 4 * QK_K // 64) // 5, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    qs0, qs1 = qs[..., :32], qs[..., 32:]
    qs0 = qs0.reshape((n_blocks, -1, 1, 32)) * torch.tensor(
        [1, 3, 9, 27, 81], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 5, 1))
    qs0 = qs0.reshape((n_blocks, -1))
    qs1 = qs1.reshape((n_blocks, -1, 1, 16)) * torch.tensor(
        [1, 3, 9, 27, 81], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 5, 1))
    qs1 = qs1.reshape((n_blocks, -1))
    qh = qh.reshape((n_blocks, -1, 1, 4)) * torch.tensor(
        [1, 3, 9, 27], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = qh.reshape((n_blocks, -1))
    qs = torch.cat([qs0, qs1, qh], dim=-1)
    qs = ((qs * 3) >> 8) - 1
    return d * qs

# i quant
# 4-bit; w=super_block_scale (iq4_nl)
def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=torch.float32, device=blocks.device
    )
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], 16), 2, qs)
    qs = qs.squeeze(-1).to(dtype)
    return d * qs
# 4-bit; w=super_block_scale (iq4_xs); 4.25 bit/weight
def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=torch.float32, device=blocks.device
    )
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = scales_h.view(torch.int16)
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8).reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, 1, -1)) >> torch.tensor(
        [2 * i for i in range(QK_K // 32)], device=blocks.device, dtype=torch.uint8).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)) & 0x03
    scales = (scales_l | (scales_h << 4)) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    shifts_q = torch.tensor([0, 4], device=blocks.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shifts_q
    qs = (qs & 15).reshape((n_blocks, -1, 32)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], qs.shape[2], 16), 3, qs)
    qs = qs.squeeze(-1).to(dtype)
    return (dl * qs).reshape(n_blocks, -1)

def load_grid_tensor(grid_shape, grid_hex):
    grid_bytes = torch.tensor(list(grid_hex))
    grid_words = grid_bytes.view(-1, 2).flip(1)
    grid = grid_words.contiguous().view(-1).to(torch.int16).view(*grid_shape)
    return grid
# 3-bit; w=super_block_scale (iq3_s); 3.44 bit/weight
def dequantize_blocks_IQ3_S(blocks, block_size, type_size, dtype=None):
    grid_shape = (512, 4)
    # grid_map = (0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f)
    grid_hex = (
        b"0000010002000500070010001100120014001600200021002500330040004200"
        b"4500470051005300600062007100740077000001010102010401100111011501"
        b"2001230127013101350144016101650172010002010205020702100213021602"
        b"2102250230023402420245024702510253027002730203031103150320032203"
        b"3103330336034403500352036703710375030004130417042104240432044004"
        b"4304510470040205040520052205260533054105450547056605730506061106"
        b"1306310652067106000702070407200722072607330750075407001001100210"
        b"0410101011101310151017102010221031103410361054105610611072100011"
        b"0111031106111011141121113011331141115011521170117611001212121512"
        b"1712201224123212401243125512601272120113041307131013131321132713"
        b"3013341341136213701303140514121414143114331442144614501454140115"
        b"1015131521153015321551152016241627164416461601170317101712172117"
        b"3517411762177017002001200320052007201020122014201620212023202720"
        b"3020322041204320452050205220672070207320752000210221102113211721"
        b"2221252131213421422151210122042207222122232230223722412253225722"
        b"7122742200230223052311232223242331233323422350236623012407242024"
        b"2324322435244124722475240425112522253725402553257025002602260726"
        b"2126552661260527112726273027432750270230113013301530173022303130"
        b"3330353042304430473051306330713001310331053114312131233140316031"
        b"7231763100321232203232323432503201331033143321332333273330334133"
        b"4333473355337333033411341634223431345234603464340135103512352535"
        b"3235443556357335163641360137033720372237353700400440124020402440"
        b"2740324041405040704002410741114113412241304135414341514155410142"
        b"0342104215422142334240425742624270420443114313432043224331433543"
        b"0044024424443744404471440545074521456245134634466046104715473047"
        b"4347514702501050145022504050445047505250665074500151035105511251"
        b"2151325172510052115223523052365253520253075310532753445351536553"
        b"7353015404542054325446541255265551555355425602570457225711601360"
        b"1560316033606060006120612761646112623462426255626262706200631463"
        b"2163406325644364626400650365346560650566406611671367007004700770"
        b"2070227036704070547062700271117124714371457101720472107216722172"
        b"3072517202733273357353730174057413742074507422754275027631760077"
    )
    n_blocks = blocks.shape[0]
    d, qs, qh, signs, scales = split_block_dims(blocks, 2, QK_K // 4, QK_K // 32, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    scales = scales.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8).reshape((1, 1, 2))
    scales = (scales & 15).reshape((n_blocks, -1))
    db = d * (1 + 2 * scales.to(dtype))
    db = db.reshape((n_blocks, -1, 1, 1))
    signs = signs.reshape((n_blocks, -1, 1)) >> torch.arange(
        8, device=blocks.device, dtype=torch.uint8).reshape((1, 1, 8))
    signs = signs & 1
    # signs = torch.where(signs == 0, torch.tensor(1.0, device=blocks.device, dtype=torch.uint8),
    #                     torch.tensor(-1.0, device=blocks.device, dtype=torch.uint8))
    signs = signs.reshape((n_blocks, -1, 4, 8))
    qh_shifts = torch.arange(8, device=blocks.device, dtype=torch.uint8).view(1, 1, 8)
    qh = (qh.reshape((n_blocks, -1, 1)) >> qh_shifts) & 1
    qh = qh.reshape((n_blocks, -1)).to(torch.int16)
    qs = qs.to(torch.int64) | (qh << 8)                 # shape: (n_blocks, 64)
    grid = load_grid_tensor(grid_shape, grid_hex)       # (512, 4)
    grid = grid.unsqueeze(0).expand(n_blocks, -1, -1)   # (n_blocks, 512, 4)
    # qs_exp = qs.unsqueeze(-1).expand(-1, -1, 4)         # (n_blocks, 64, 4)
    # grid = torch.gather(grid, dim=1, index=qs_exp)      # (n_blocks, 64, 4)
    # grid = grid.view(n_blocks, 64, 4, 1).expand(-1, -1, -1, 8)  # (n_blocks, 64, 4, 8)
    # signs = signs.view(n_blocks, 64, 4, 8)                     # Ensure matching shape
    # assert db.shape == grid.shape == signs.shape, f"{db.shape} != {grid.shape} != {signs.shape}"
    # return (db * grid * signs).reshape(n_blocks, -1)    # skip grid recently
    return (db * signs).reshape(n_blocks, -1)
# 3-bit; w=super_block_scale (iq3_xxs); 3.06 bit/weight
def dequantize_blocks_IQ3_XXS(blocks, block_size, type_size, dtype=None):
    grid_shape = (256, 4)
    # grid_map = (0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e)
    grid_hex = (
        b"0000020004001100130017002000220031004200730075000101030110011201"
        b"2101250130013201410154017001000202020402110220022202310233023702"
        b"5102570275020103070310031203250370031304370444045704730475040105"
        b"0705320552053506640610071407160743076107011003101010121021102310"
        b"3010321034104710501000110211111120112211011203121012121221123012"
        b"7212001302132013311346136613011405145014201524154615711505162217"
        b"4017002002201120132020202220262031204220012103210521102112212121"
        b"3021632167217021002202221122172220222222372240225522012310231423"
        b"7023742335245324032527254125742501270327162745270130103012302130"
        b"2330503065307230003102312031313144314631013203321032253252327232"
        b"1133333330344734723400350635223555351436363663363337603704401740"
        b"3540374053405740744120423742404260426642074345430444514464442545"
        b"4345704505471047124730471250415070500051065126515551145232527252"
        b"0253535310542354275472540255315550562457425724604460466064602161"
        b"6161176264623063366344640565526533660367216703700570077010703270"
        b"5270267140711272457252720073157333736073217441740075027524753076"
    )
    n_blocks = blocks.shape[0]
    d, qs, scales = split_block_dims(blocks, 2, QK_K // 4)
    d = d.view(torch.float16).to(dtype)
    scales = to_uint32(scales)
    db = d * (0.5 + (scales >> 28)) * 0.5
    db = db.reshape((n_blocks, -1, 1, 1))
    bit_shifts = torch.tensor([0, 7, 14, 21], device=blocks.device, dtype=torch.uint8).reshape((1, 1, 4))
    signs = scales.reshape((n_blocks, -1, 1)) >> bit_shifts # tbc: pull iq2_xxs.ksigns
    db = db.reshape((n_blocks, -1, 1, 1))
    signs = signs.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [i for i in range(8)], device=blocks.device, dtype=torch.uint8).reshape((1, 1, 8))
    signs = signs & 1
    db = db.reshape((n_blocks, -1, 1, 1))
    sign_shifts = torch.arange(8, device=blocks.device, dtype=torch.uint8).view(1, 1, 8)
    signs = signs.reshape((n_blocks, -1, 1)) >> sign_shifts
    signs = (signs & 1).float()
    signs = torch.where(signs == 0, torch.tensor(1.0, device=blocks.device), torch.tensor(-1.0, device=blocks.device))
    signs = signs.reshape(n_blocks, -1, 4, 8)
    qs = qs.reshape(n_blocks, -1, 1, 1)
    grid = load_grid_tensor(grid_shape, grid_hex)           # (256, 4)
    grid = grid.unsqueeze(0).expand(n_blocks, -1, -1)       # (n_blocks, 512, 4)
    # qs_exp = qs.unsqueeze(-1).expand(-1, -1, 4)             # (n_blocks, 64, 4)
    # grid = torch.gather(grid, dim=1, index=qs_exp)          # (n_blocks, 64, 4)
    # grid = grid.view(n_blocks, 64, 4, 1).expand(-1, -1, -1, 8)  # (n_blocks, 64, 4, 8)
    # grid = grid.reshape((n_blocks, -1, 4, 8))             # Ensure matching shape
    # assert db.shape == grid.shape == signs.shape, f"{db.shape} != {grid.shape} != {signs.shape}"
    # return (db * grid * signs).reshape((n_blocks, -1))    # skip grid recently
    return (db * signs).reshape(n_blocks, -1)

dequantize_functions = {
    GGMLQuantizationType.BF16:dequantize_blocks_BF16,
    GGMLQuantizationType.Q8_0:dequantize_blocks_Q8_0,
    GGMLQuantizationType.Q5_1:dequantize_blocks_Q5_1,
    GGMLQuantizationType.Q5_0:dequantize_blocks_Q5_0,
    GGMLQuantizationType.Q4_1:dequantize_blocks_Q4_1,
    GGMLQuantizationType.Q4_0:dequantize_blocks_Q4_0,
    GGMLQuantizationType.Q6_K:dequantize_blocks_Q6_K,
    GGMLQuantizationType.Q5_K:dequantize_blocks_Q5_K,
    GGMLQuantizationType.Q4_K:dequantize_blocks_Q4_K,
    GGMLQuantizationType.Q3_K:dequantize_blocks_Q3_K,
    GGMLQuantizationType.Q2_K:dequantize_blocks_Q2_K,
    GGMLQuantizationType.TQ2_0:dequantize_blocks_TQ2_0,
    GGMLQuantizationType.TQ1_0:dequantize_blocks_TQ1_0,
    GGMLQuantizationType.IQ4_NL:dequantize_blocks_IQ4_NL,
    GGMLQuantizationType.IQ4_XS:dequantize_blocks_IQ4_XS,
    GGMLQuantizationType.IQ3_S:dequantize_blocks_IQ3_S,
    GGMLQuantizationType.IQ3_XXS:dequantize_blocks_IQ3_XXS,
    }
