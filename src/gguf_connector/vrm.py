import torch # need torch to work

def get_gpu_vram(device: str | torch.device = "cuda", unit: str = "GiB") -> int:
    if isinstance(device, str):
        device = torch.device(device)
    assert unit in ("GiB", "MiB", "B")
    memory = torch.cuda.get_device_properties(device).total_memory
    if unit == "GiB":
        return memory // (1024**3)
    elif unit == "MiB":
        return memory // (1024**2)
    else:
        return memory

def get_affordable_precision(device="cuda"):
    if isinstance(device, str):
        device = torch.device(device)
    capability = torch.cuda.get_device_capability(0 if device.index is None else device.index)
    sm = f"{capability[0]}{capability[1]}"
    precision = "fp4" if sm == "120" else "int4"
    return precision
