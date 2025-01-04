import os, subprocess

def clone_github_repo(repo_url):
    try:
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        if os.path.exists(repo_name):
            print(f"Error: A folder named '{repo_name}' already exists in the current directory.")
            return
        print(f"Cloning repository '{repo_url}'...")
        subprocess.run(["git", "clone", repo_url], check=True)
        print(f"Repository '{repo_name}' cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to clone the repository. {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

repo_url = "https://github.com/calcuis/gguf"
clone_github_repo(repo_url)
