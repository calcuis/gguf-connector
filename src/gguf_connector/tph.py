from pathlib import Path

def get_hf_cache_hub_path(gname,gproj,ghash):
    home_dir = Path.home()
    hf_cache_path = home_dir / ".cache" / "huggingface" / "hub" / f"models--{gname}--{gproj}" / "blobs" / ghash
    return str(hf_cache_path)
