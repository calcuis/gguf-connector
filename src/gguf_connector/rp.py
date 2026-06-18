from pathlib import Path

OLD = input("type the keyword which you want to replace: ")
NEW = input(f"input the new keyword to replace {OLD}: ")

root = Path.cwd()

def is_binary(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except:
        return True

# Replace text inside files
for path in root.rglob("*"):
    if not path.is_file():
        continue

    if is_binary(path):
        continue

    try:
        data = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Skip files that are not UTF-8
        continue
    except Exception:
        continue

    if OLD in data:
        path.write_text(data.replace(OLD, NEW), encoding="utf-8")
        print(f"Updated contents: {path}")

# Rename files (deepest first)
files = sorted(
    [p for p in root.rglob("*") if p.is_file()],
    key=lambda p: len(p.parts),
    reverse=True,
)

for path in files:
    if OLD in path.name:
        new_path = path.with_name(path.name.replace(OLD, NEW))
        path.rename(new_path)
        print(f"Renamed: {path} -> {new_path}")

# Rename directories (deepest first)
dirs = sorted(
    [p for p in root.rglob("*") if p.is_dir()],
    key=lambda p: len(p.parts),
    reverse=True,
)

for path in dirs:
    if OLD in path.name:
        new_path = path.with_name(path.name.replace(OLD, NEW))
        path.rename(new_path)
        print(f"Renamed: {path} -> {new_path}")
