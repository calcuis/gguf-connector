
import torch # need torch to work

# add split block by dims
def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)
# unit conversion (for 5_0, 5_1, etc.)
def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)
# calculate scale min (for 4_k, 5_k, tq1_0, tq_2_0, iq4_xs, iq4_nl, etc.)
def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))
# grid mapping (for iq3_s, iq3_xxs, etc.)
def load_grid_tensor(grid_shape, grid_hex, grid_map):
    grid_bytes = torch.tensor(list(grid_hex), dtype=torch.uint8)
    grid_words = grid_bytes.view(-1, 2).flip(1)
    grid = grid_words.contiguous().view(-1).to(torch.int16).view(*grid_shape)
    # i.e., map 0x01 to 0, 0x03 to 1, ... (1st to 0, 2nd to 1, etc.)
    grid_map_tensor = torch.tensor(grid_map, dtype=torch.int16)
    for mapped_value, original_value in enumerate(grid_map):
        grid[grid == original_value] = mapped_value
    return grid
