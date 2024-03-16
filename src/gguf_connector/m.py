import json, os.path, urllib.request
from rich.progress import Progress

def get_file_size(url):
    with urllib.request.urlopen(url) as response:
        size = int(response.headers['Content-Length'])
    return size

def format_size(size_bytes):
    return f"{size_bytes / (1024 * 1024):.2f} MB"

def clone_file(url):
    try:
        file_size = get_file_size(url)
        filename = os.path.basename(url)
        with Progress(transient=True) as progress:
            task = progress.add_task(f"Downloading {filename}", total=file_size)
            with urllib.request.urlopen(url) as response, open(filename, 'wb') as file:
                chunk_size = 1024
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    file.write(chunk)
                    downloaded += len(chunk)
                    progress.update(task, completed=downloaded, description=f"Downloading {filename} [green][{format_size(downloaded)} / {format_size(file_size)}]")
        print(f"File cloned successfully and saved as '{filename}'({format_size(file_size)}) in the current directory.")
    except Exception as e:
        print(f"Error: {e}")

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_names(data):
    for idx, entry in enumerate(data, start=1):
        print(f'{idx}. {entry["name"]}')

def handle_user_input(data):
    while True:
        user_choice = input(f"Enter your choice (1 to {len(data)}) or 'q' to quit: ")
        if user_choice.lower() == 'q':
            break
        try:
            index = int(user_choice)
            if 1 <= index <= len(data):
                source_url = data[index - 1]["url"]
                clone_file(source_url)
                break
            else:
                print("Invalid selection. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

print("Please select an option:\n1. choose a connector\n2. download model(s)")
choice = input("Enter your choice (1 to 2): ")

if choice=="1":
    from gguf_connector import menu
elif choice=="2":
    file_path = os.path.join(os.path.dirname(__file__), 'data.json')
    json_data = read_json_file(file_path)
    print("Please select a GGUF file to download:")
    extract_names(json_data)
    handle_user_input(json_data)
