## GGUF connector

GGUF (GPT-Generated Unified Format) is a successor of GGML (GPT-Generated Model Language), it was released on August 21, 2023; by the way, GPT stands for Generative Pre-trained Transformer.

[<img src="https://raw.githubusercontent.com/calcuis/gguf-connector/master/gguf.gif" width="128" height="128">](https://github.com/calcuis/gguf-connector)
[![Static Badge](https://img.shields.io/badge/version-0.5.2-green?logo=github)](https://github.com/calcuis/gguf-connector/releases)
[![Static Badge](https://img.shields.io/badge/cgg-0.1.0-green?logo=github)](https://github.com/calcuis/cgg/releases)
[![Static Badge](https://img.shields.io/badge/package-0.1.3-green?logo=microsoft)](https://github.com/calcuis/chatgpt-model-selector/releases)

This package is a simple graphical user interface (GUI) application that uses the ctransformers or llama.cpp to interact with a chat model for generating responses.

Install the connector via pip (once only):
```
pip install gguf-connector
```
Update the connector (if previous version installed) by:
```
pip install gguf-connector --upgrade
```

With this version, you can interact straight with the GGUF file(s) available in the same directory by a simple command.
### Graphical User Interface (GUI)
Select model(s) with llama.cpp connector:
```
gguf cpp
```
Select model(s) with ctransformers:
```
gguf c
```
[<img src="https://raw.githubusercontent.com/calcuis/chatgpt-model-selector/master/demo.gif" width="350" height="280">](https://github.com/calcuis/chatgpt-model-selector/blob/main/demo.gif)
[<img src="https://raw.githubusercontent.com/calcuis/chatgpt-model-selector/master/demo1.gif" width="350" height="280">](https://github.com/calcuis/chatgpt-model-selector/blob/main/demo1.gif)

### Command Line Interface (CLI)
Select model(s) with ctransformers:
```
gguf g
```
Select model(s) with llama.cpp connector:
```
gguf gpp
```
### Connector selection menu
You can even choose which connector (either ctransformers or llama.cpp) to be used at the very beginning.
```
gguf m
```
#### Import as a module
Include the connector selection menu to your code by:
```
from gguf_connector import menu
```

[<img src="https://raw.githubusercontent.com/calcuis/gguf-connector/master/demo.gif" width="350" height="200">](https://github.com/calcuis/gguf-connector/blob/main/demo.gif)

For standalone version please refer to the repository in the reference list (below).
#### References
[model selector](https://github.com/calcuis/chatgpt-model-selector) (standalone version: [installable package](https://github.com/calcuis/chatgpt-model-selector/releases))

[cgg](https://github.com/calcuis/cgg) (cmd-based tool)
#### Resources
[ctransformers](https://github.com/marella/ctransformers)
[llama.cpp](https://github.com/ggerganov/llama.cpp)
