
filename = "coder.config.json"
content = """
{
  "coder": {
    "providers": [
      {
        "name": "lmstudio",
        "models": [
          "openai/gpt-oss-20b"
        ],
        "baseUrl": "http://localhost:1234/v1"
      }
    ]
  }
}
"""

with open(filename, "w") as f:
    f.write(content)

import os
os.system("coder")
