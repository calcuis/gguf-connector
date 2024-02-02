import urllib.request
from os.path import basename
from tqdm import tqdm

def get_file_size(url):
    with urllib.request.urlopen(url) as response:
        size = int(response.headers['Content-Length'])
    return size

def clone_file(url):
    try:
        file_size = get_file_size(url)
        filename = basename(url)
        with urllib.request.urlopen(url) as response, \
            open(filename, 'wb') as file, \
            tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc=f'Downloading {filename}') as pbar:
            chunk_size = 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                file.write(chunk)
                pbar.update(len(chunk))
        print(f"\nFile downloaded successfully and saved as '{filename}' in the current directory.")
    except Exception as e:
        print(f"Error: {e}")

print("Please select a GGUF file to download:\n1. chat.gguf\n2. code.gguf\n3. medi.gguf")
choice = input("Enter your choice (1 to 3): ")

if choice=="1":
    clone_file("https://huggingface.co/calcuis/chat/resolve/main/chat.gguf")
elif choice=="2":
    clone_file("https://huggingface.co/calcuis/chat/resolve/main/code.gguf")
elif choice=="3":
    clone_file("https://huggingface.co/calcuis/chat/resolve/main/medi.gguf")
else:
    print("Not a valid number.")
