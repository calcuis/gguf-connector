
filename = "server.py"
content = """
from __future__ import annotations
import os, sys
import argparse
import uvicorn
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import (
    Settings,
    ServerSettings,
    ModelSettings,
    ConfigFileSettings,
)
from llama_cpp.server.cli import add_args_from_model, parse_model_from_args
def launch_server():
    description = "gguf server"
    parser = argparse.ArgumentParser(description=description)
    add_args_from_model(parser, Settings)
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to a config file to load.",
    )
    server_settings: ServerSettings | None = None
    model_settings: list[ModelSettings] = []
    args = parser.parse_args()
    try:
        config_file = os.environ.get("CONFIG_FILE", args.config_file)
        config_file = os.environ.get("CONFIG_FILE", args.config_file)
        if config_file:
            if not os.path.exists(config_file):
                raise ValueError(f"Config file {config_file} not found!")
            with open(config_file, "rb") as f:
                if config_file.endswith(".yaml") or config_file.endswith(".yml"):
                    import yaml
                    import json
                    config_file_settings = ConfigFileSettings.model_validate_json(
                        json.dumps(yaml.safe_load(f))
                    )
                else:
                    config_file_settings = ConfigFileSettings.model_validate_json(
                        f.read()
                    )
                server_settings = ServerSettings.model_validate(config_file_settings)
                model_settings = config_file_settings.models
        else:
            server_settings = parse_model_from_args(ServerSettings, args)
            model_settings = [parse_model_from_args(ModelSettings, args)]
    except Exception as e:
        print(e, file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    assert server_settings is not None
    assert model_settings is not None
    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings,
    )
    uvicorn.run(
        app,
        host=os.getenv("HOST", server_settings.host),
        port=int(os.getenv("PORT", server_settings.port)),
        ssl_keyfile=server_settings.ssl_keyfile,
        ssl_certfile=server_settings.ssl_certfile,
    )
launch_server()
"""

with open(filename, "w") as f:
    f.write(content)

import os
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]
if gguf_files:
    print('GGUF file(s) available. Select which one to use:')
    for index, file_name in enumerate(gguf_files, start=1):
        print(f'{index}. {file_name}')
    choice = input(f'Enter your choice (1 to {len(gguf_files)}): ')
    try:
        choice_index = int(choice) - 1
        selected_file = gguf_files[choice_index]
        print(f'Model file: {selected_file} is selected!')
        input_path = selected_file
        port = 1234
        os.system(f"python {filename} --port {port} --model {input_path}")
    except (ValueError, IndexError) as e:
        print(f"Invalid choice. Please enter a valid number. ({e})")
else:
    print('No GGUF files are available in the current directory.')
    input('--- Press ENTER To Exit ---')
