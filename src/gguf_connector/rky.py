
unet_to_base_map = [
    ("_", "."),
    ("lora.unet", "base_model.model"),
    (".blocks", "_blocks"),
    ("lora.down", "lora_A"),
    ("lora.up", "lora_B"),
    ("txt.", "txt_"),
    ("img.", "img_"),
    ("final.", "final_")
]

base_to_unet_map = [
    ("base_model.model.", "lora_unet_"),
    (".", "_"),
    ("_lora_A_", ".lora_down."),
    ("_lora_B_", ".lora_up.")
]

def rename_key(key, rules):
    for search, replace in rules:
        key = key.replace(search, replace)
    return key
