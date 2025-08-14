
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

from .quant5b import to_uint32
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
from .quant5b import split_block_dims
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
from .quant5b import get_scale_min
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
from .quant5b import load_grid_tensor
# 2-bit; w=super_block_scale (iq2_s); 2.5 bit/weight
def dequantize_blocks_IQ2_S(blocks, block_size, type_size, dtype=None):
    grid_shape = (1024, 8)
    grid_map = (0x08, 0x19, 0x2b)
    grid_hex = (
        b"00000200050008000a0011001400160019002000220025002800410044004600"
        b"490050005200550058006100640066006900800082008500880091009400a000"
        b"a500aa0001010401060109011001120115011801210124014001420145014801"
        b"510154015601590160016501680181018401900192019501a101a40100020202"
        b"050208021102140220022a02410244024602490250025502800285028a029402"
        b"a202010404040604090410041204150418042104240426042904400442044504"
        b"48044a0451045404560459046004620465048104840486048904900495049804"
        b"a104a40400050205050508050a05110514051605190520052505280541054405"
        b"46054905500552055505580561056405800582058505880591059405a0050106"
        b"0406060609061006150640064506480651065406600681068406900600080208"
        b"050808081108140816081908200825082a084108440846084908500852085508"
        b"580861086408800885089408aa08010904091009120915091809210940094509"
        b"480951095409600981099009000a110a140a220a280a2a0a500a990a01100410"
        b"0610091010101210151018102110241026104010421045104810511054105610"
        b"59106010621065106810811084108610901095109810a110a410001102110511"
        b"08110a1111111411161119112011221125112811411144114611491150115211"
        b"5511581161116411801182118511881191119411011204120912101215122112"
        b"2412401245125112541281128412901200140214051408141114141416141914"
        b"2014251428144114441446144914501452145514581461146414801482148514"
        b"881491149414a014011504150615091510151215151518152115241540154215"
        b"4515481551155415601581158415901500160516081611161416201641164416"
        b"50168016aa160118041806180918101815181818211840184218451848185118"
        b"541860188118841800190219051908191119141920194119441950196919a219"
        b"041a101a401a561a00200220052008201120142016201920202025202a204120"
        b"4420502052205520642080208a209420aa200121042110211221152121214021"
        b"4221452151215421602181218421902100220a22222228222a22442250228822"
        b"8a22a82201240424062409241024152418242124242440244224452448245124"
        b"5424602481248424902400250525082511251425202541254425502566258025"
        b"0126042610264026592600280528112814284128442850288a28aa2801290429"
        b"102995290a2a222a642a882a8a2a014004400640094010401240154018401a40"
        b"21402440264040404240454048404a4051405440564059406040624065408140"
        b"8440904095409840a140a4400041024105410841114114411641194120412241"
        b"2541414144414641494150415241554158416141644180418241854188419141"
        b"9441a04101420442104212421542184224424042454248425142544260428142"
        b"844200440244054408440a441144144416441944204422442544284441444444"
        b"46444944504452445544584461446444804482448544884491449444a0440145"
        b"0445064509451045124515451845214524454045424545454845514554456045"
        b"6a4581458445904500460246054608461146144620464146444650468046a546"
        b"0148044809481048124815481848214824484048424845484848514854486048"
        b"84489048004902490549084911491449204941494449504980499649014a044a"
        b"104a404a00500250055008501150145016501950205022502550285041504450"
        b"4650495050505250555058506150645080508250855088509150945001510451"
        b"0651095110511251155118512151245140514251455148515151545160518151"
        b"8451905100520552085211521452205241524452505269528052015404540654"
        b"0954105412541554185421542454405442544554485451545454605481548454"
        b"9054005502550555085511551455205541554455505580550156045610562656"
        b"405600580258055808581158145820584158445850585a588058015904591059"
        b"4059005a195a855aa85a01600460066010601260156018602160246040604560"
        b"4860516054606060846090600061026105610861116114612061416144615061"
        b"806199610462106240625662a162006405640864116414642064416444645064"
        b"806401650465106540654a656865926500669466016804681068656898680069"
        b"2a69426aa16a0080028005800880118014801980208025804180448050805280"
        b"5580588061808080858091809480018104810981108112811581188121812481"
        b"408142814581488151815481818184819081a981008205820a82118214824182"
        b"4482508201840484068409841084128415841884218440844284458448845184"
        b"5484608481848484908400850285058508851185148520854185448550858085"
        b"8a85018604861086298640860088058811881488418844885088a28801890489"
        b"40896589228a588a5a8a828aa28a019004900990109012901590189024904090"
        b"4290459048905190549060908190849090900091059111911491419144915091"
        b"5a910192049210924092a6920094029405940894119414942094419444945094"
        b"8094969401950495109540959895a19500964696649601980498109826984098"
        b"a998009949995299909a00a005a00aa014a022a02aa041a044a050a0a2a0aaa0"
        b"40a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4"
        b"a4a400a519a551a60aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa"
    )
    n_blocks = blocks.shape[0]
    d, qs, signs, qh, scales = split_block_dims(blocks, 2, QK_K // 8, QK_K // 8, QK_K // 32)
    d = d.view(torch.float16).to(dtype)
    scales = scales.reshape(n_blocks, -1, 1) >> torch.tensor([0, 4], dtype=torch.uint8, device=d.device).view(1, 1, 2)
    scales = (scales & 15).reshape(n_blocks, -1).to(dtype)
    db = d * (0.5 + scales) * 0.25
    db = db.view(n_blocks, -1, 1, 1)
    # signs = signs.reshape((n_blocks, -1, 1)) >> torch.arange(
    #     [i for i in range(8)], dtype=torch.uint8, device=d.device).reshape((1, 1, 8))
    signs = signs.reshape(n_blocks, -1, 1) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8).reshape(1, 1, 8)
    signs = (signs & 1).to(dtype)
    signs = torch.where(signs == 0, 1.0, -1.0)
    signs = signs.view(n_blocks, -1, 2, 8)
    qh = qh.view(n_blocks, -1, 1) >> torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=d.device).view(1, 1, 4)
    qs = qs.to(torch.int64) | ((qh & 3) << 8).reshape(n_blocks, -1)  # shape: (n_blocks, 64)
    grid = load_grid_tensor(grid_shape, grid_hex, grid_map, device=d.device)  # shape: (1, 1, 1024, 8)
    grid = grid.expand(n_blocks, 1, *grid_shape)                            # shape: (n_blocks, 1, 1024, 8)
    grid = grid.squeeze(1)  # remove 1-channel dimension → (n_blocks, 1024, 8)
    # version 0 #
    # grid = torch.take_along_dim(grid, (qs & 511).reshape((n_blocks, -1, 1, 1)), dim=-2)
    # grid = grid.reshape((n_blocks, -1, 2, 8))
    # version 1 #
    # gathered_grid = torch.gather(grid, dim=1, index=qs.unsqueeze(-1).expand(-1, -1, 4))  # (n_blocks, 64, 4)
    # gathered_grid = gathered_grid.unsqueeze(-1).expand(-1, -1, -1, 8)  # (n_blocks, 64, 4, 8)
    # db = db.expand(-1, 64, 4, 8)  # Match shapes
    # return (db * gathered_grid * signs).reshape(n_blocks, -1)
    # version 2 #
    # n_blocks, n_qs = qs.shape
    # gathered_grid = torch.gather(grid, dim=1, index=qs.unsqueeze(-1).expand(-1, -1, 4))  # (n_blocks, n_qs, 4)
    # gathered_grid = gathered_grid.unsqueeze(-1).expand(-1, -1, -1, 8)  # (n_blocks, n_qs, 4, 8)
    # db = db.expand(-1, n_qs, 4, 8) # adjustment for shapes
    # return (db * gathered_grid.to(dtype) * signs).reshape(n_blocks, -1) # skip grid for test
    return (db * signs).reshape(n_blocks, -1)

ksigns: bytes = (
    b"\x00\x81\x82\x03\x84\x05\x06\x87\x88\x09\x0a\x8b\x0c\x8d\x8e\x0f"
    b"\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f"
    b"\xa0\x21\x22\xa3\x24\xa5\xa6\x27\x28\xa9\xaa\x2b\xac\x2d\x2e\xaf"
    b"\x30\xb1\xb2\x33\xb4\x35\x36\xb7\xb8\x39\x3a\xbb\x3c\xbd\xbe\x3f"
    b"\xc0\x41\x42\xc3\x44\xc5\xc6\x47\x48\xc9\xca\x4b\xcc\x4d\x4e\xcf"
    b"\x50\xd1\xd2\x53\xd4\x55\x56\xd7\xd8\x59\x5a\xdb\x5c\xdd\xde\x5f"
    b"\x60\xe1\xe2\x63\xe4\x65\x66\xe7\xe8\x69\x6a\xeb\x6c\xed\xee\x6f"
    b"\xf0\x71\x72\xf3\x74\xf5\xf6\x77\x78\xf9\xfa\x7b\xfc\x7d\x7e\xff"
)
# 3-bit; w=super_block_scale (iq3_xxs); 3.06 bit/weight
def dequantize_blocks_IQ3_XXS(blocks, block_size, type_size, dtype=None):
    grid_shape = (256, 4)
    grid_map = (0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e)
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
    db = db.reshape(n_blocks, -1, 1, 1)
    bit_shifts = torch.tensor([0, 7, 14, 21], device=d.device, dtype=torch.uint8).reshape(1, 1, 4)
    signs = scales.reshape(n_blocks, -1, 1) >> bit_shifts
    # ksigns = torch.frombuffer(ksigns, dtype=torch.uint8).reshape((1, 1, 1, 128)) # runnable but trigger non-writable warning
    signs = signs.reshape(n_blocks, -1, 1) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8).reshape(1, 1, 8)
    signs = signs & 1
    sign_shifts = torch.arange(8, device=d.device, dtype=torch.uint8).view(1, 1, 8)
    signs = signs.reshape(n_blocks, -1, 1) >> sign_shifts
    signs = (signs & 1).float()
    signs = torch.where(signs == 0, torch.tensor(1.0, device=d.device), torch.tensor(-1.0, device=d.device))
    signs = signs.reshape(n_blocks, -1, 4, 8)
    qs = qs.reshape(n_blocks, -1, 1, 1)
    grid = load_grid_tensor(grid_shape, grid_hex, grid_map, device=d.device)   # (256, 4)
    grid = grid.expand(n_blocks, 1, *grid_shape)                               # shape: (n_blocks, 1, 256, 4)
    # version 2 (to be reviewed) #
    # grid = grid.unsqueeze(0).expand(n_blocks, -1, -1)       # (n_blocks, 256, 4)
    # qs_exp = qs.unsqueeze(-1).expand(-1, -1, 4)             # (n_blocks, 64, 4)
    # grid = torch.gather(grid, dim=1, index=qs_exp)          # (n_blocks, 64, 4)
    # grid = grid.view(n_blocks, 64, 4, 1).expand(-1, -1, -1, 8) # (n_blocks, 64, 4, 8)
    # version 1 #
    # grid = torch.take_along_dim(grid, qs.reshape((n_blocks, -1, 1, 1)), dim=-2) # dropped option
    # grid = grid.reshape((n_blocks, -1, 4, 8))             # Ensure matching shape
    # assert db.shape == grid.shape == signs.shape, f"{db.shape} != {grid.shape} != {signs.shape}"
    # return (db * grid * signs).reshape((n_blocks, -1))    # skip grid recently for speed test
    return (db * signs).reshape(n_blocks, -1)

# 3-bit; w=super_block_scale (iq3_s); 3.44 bit/weight
def dequantize_blocks_IQ3_S(blocks, block_size, type_size, dtype=None):
    grid_shape = 512, 4
    grid_map = (0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f)
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
    scales = scales.reshape(n_blocks, -1, 1) >> torch.tensor([0, 4], dtype=torch.uint8, device=d.device).view(1, 1, 2)
    scales = (scales & 15).reshape(n_blocks, -1).to(dtype)
    db = d * (1 + 2 * scales)
    db = db.view(n_blocks, -1, 1, 1)
    signs = signs.view(n_blocks, -1, 1) >> torch.arange(8, dtype=torch.uint8, device=d.device).view(1, 1, 8)
    signs = (signs & 1).to(dtype)
    signs = torch.where(signs == 0, 1.0, -1.0)
    signs = signs.view(n_blocks, -1, 4, 8)
    qh = qh.view(n_blocks, -1, 1) >> torch.arange(8, dtype=torch.uint8, device=d.device).view(1, 1, 8)
    qh = (qh & 1).view(n_blocks, -1).to(torch.int16)
    qs = qs.to(torch.int64) | (qh << 8)  # shape: (n_blocks, 64)
    grid = load_grid_tensor(grid_shape, grid_hex, grid_map, device=d.device)  # shape: (1, 1, 512, 4)
    grid = grid.expand(n_blocks, 1, *grid_shape)                            # shape: (n_blocks, 1, 512, 4)
    grid = grid.squeeze(1)  # remove 1-channel dimension → (n_blocks, 512, 4)
    # version 1 #
    # gathered_grid = torch.gather(grid, dim=1, index=qs.unsqueeze(-1).expand(-1, -1, 4))  # (n_blocks, 64, 4)
    # gathered_grid = gathered_grid.unsqueeze(-1).expand(-1, -1, -1, 8)  # (n_blocks, 64, 4, 8)
    # db = db.expand(-1, 64, 4, 8)  # Match shapes
    # return (db * gathered_grid * signs).reshape(n_blocks, -1)
    # version 2 #
    # n_blocks, n_qs = qs.shape
    # gathered_grid = torch.gather(grid, dim=1, index=qs.unsqueeze(-1).expand(-1, -1, 4))  # (n_blocks, n_qs, 4)
    # gathered_grid = gathered_grid.unsqueeze(-1).expand(-1, -1, -1, 8)  # (n_blocks, n_qs, 4, 8)
    # db = db.expand(-1, n_qs, 4, 8) # adjustment for shapes
    # return (db * gathered_grid.to(dtype) * signs).reshape(n_blocks, -1) # skip grid for test
    return (db * signs).reshape(n_blocks, -1)

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

# experimental: mxfp4
from .quant5b import e8m0_to_fp32_half
# subject to test further; mxfp4/mxfp4_moe
def dequantize_blocks_MXFP4(blocks, block_size, type_size, dtype=None):
    kvalues = torch.tensor(
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
        dtype=torch.float32, device=blocks.device
    )
    n_blocks = blocks.shape[0]
    e, qs = split_block_dims(blocks, 1)
    d = e8m0_to_fp32_half(e)
    qs = qs.reshape((n_blocks, 1, block_size // 2)) >> torch.tensor([0, 4], device=blocks.device, dtype=torch.uint8).reshape((1, 2, 1))
    qs = (qs & 15)
    kvalues = kvalues.view(1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], 16), 2, qs)
    qs = qs.squeeze(-1).reshape(n_blocks, block_size)
    return d * qs

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
    GGMLQuantizationType.IQ2_S:dequantize_blocks_IQ2_S,
    GGMLQuantizationType.IQ3_XXS:dequantize_blocks_IQ3_XXS,
    GGMLQuantizationType.IQ3_S:dequantize_blocks_IQ3_S,
    GGMLQuantizationType.IQ4_NL:dequantize_blocks_IQ4_NL,
    GGMLQuantizationType.IQ4_XS:dequantize_blocks_IQ4_XS,
    GGMLQuantizationType.MXFP4:dequantize_blocks_MXFP4,
    }
