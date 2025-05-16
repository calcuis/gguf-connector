## GGUF connector

GGUF (GPT-Generated Unified Format) is a successor of GGML (GPT-Generated Model Language), it was released on August 21, 2023; by the way, GPT stands for Generative Pre-trained Transformer.

[<img src="https://raw.githubusercontent.com/calcuis/gguf-connector/master/gguf.gif" width="128" height="128">](https://github.com/calcuis/gguf-connector)
[![Static Badge](https://img.shields.io/badge/version-1.5.8-green?logo=github)](https://github.com/calcuis/gguf-connector/releases)
[![Static Badge](https://badgen.net/badge/pack/0.1.3/green?icon=windows)](https://github.com/calcuis/chatgpt-model-selector/releases)

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
Select model(s) with ctransformers:
```
ggc c
```
Select model(s) with llama.cpp connector (optional: need llama-cpp-python to work; get it [here](https://github.com/abetlen/llama-cpp-python/releases) or nightly [here](https://github.com/calcuis/llama-cpp-python/releases)):
```
ggc cpp
```
[<img src="https://raw.githubusercontent.com/calcuis/chatgpt-model-selector/master/demo.gif" width="350" height="280">](https://github.com/calcuis/chatgpt-model-selector/blob/main/demo.gif)
[<img src="https://raw.githubusercontent.com/calcuis/chatgpt-model-selector/master/demo1.gif" width="350" height="280">](https://github.com/calcuis/chatgpt-model-selector/blob/main/demo1.gif)

### Command Line Interface (CLI)
Select model(s) with ctransformers:
```
ggc g
```
Select model(s) with llama.cpp connector:
```
ggc gpp
```
Select model(s) with vision connector:
```
ggc v
```
Opt a clip handler then opt a vision model; prompt your picture link to process (see example [here](https://huggingface.co/calcuis/llava-gguf))
#### Metadata reader (CLI only)
Select model(s) with metadata reader:
```
ggc r
```
Select model(s) with metadata fast reader:
```
ggc r2
```
Select model(s) with tensor reader (optional: need torch to work; pip install torch):
```
ggc r3
```
#### PDF analyzor (beta feature on CLI recently)
Load PDF(s) into a model with ctransformers:
```
ggc cp
```
Load PDF(s) into a model with llama.cpp connector:
```
ggc pp
```
#### Speech recognizor (beta feature; accept WAV format recently)
Prompt WAV(s) into a model with ctransformers:
```
ggc cs
```
Prompt WAV(s) into a model with llama.cpp connector:
```
ggc ps
```
#### Speech recognizor (via Google api; online)
Prompt WAV(s) into a model with ctransformers:
```
ggc cg
```
Prompt WAV(s) into a model with llama.cpp connector:
```
ggc pg
```
#### Container
Launch to page/container:
```
ggc w
```
#### Divider
Divide gguf into different part(s) with a cutoff point (size):
```
ggc d2
```
#### Merger
Merge all gguf into one:
```
ggc m2
```
#### Merger (safetensors)
Merge all safetensors into one (optional: need torch to work; pip install torch):
```
ggc ma
```
#### Splitter (checkpoint)
Split checkpoint into components (optional: need torch to work; pip install torch):
```
ggc s
```
#### Quantizor
Quantize safetensors to fp8 (downscale; optional: need torch to work; pip install torch):
```
ggc q
```
Quantize safetensors to fp32 (upscale; optional: need torch to work; pip install torch):
```
ggc q2
```
#### Convertor
Convert safetensors to gguf (auto; optional: need torch to work; pip install torch):
```
ggc t
```
#### Convertor (alpha)
Convert safetensors to gguf (meta; optional: need torch to work; pip install torch):
```
ggc t1
```
#### Convertor (beta)
Convert safetensors to gguf (unlimited; optional: need torch to work; pip install torch):
```
ggc t2
```
#### Convertor (gamma)
Convert gguf to safetensors (reversible; optional: need torch to work; pip install torch):
```
ggc t3
```
#### Cutter
Get cutter for bf/f16 to q2-q8 quantization (see user guide [here](https://pypi.org/project/gguf-cutter)) by:
```
ggc u
```
#### Comfy
Download comfy pack (see user guide [here](https://pypi.org/project/gguf-comfy)) by:
```
ggc y
```
#### Node
Clone node (see user/setup guide [here](https://pypi.org/project/gguf-node)) by:
```
ggc n
```
#### Pack
Take pack (see user guide [here](https://pypi.org/project/gguf-pack)) by:
```
ggc p
```
#### PackPack
Take packpack (see user guide [here](https://pypi.org/project/framepack)) by:
```
ggc p2
```
#### FramePack
Take framepack (portable packpack) by:
```
ggc p1
```
#### Video 1 (image to video)
Activate backend and frontend by (optional: need torch, diffusers to work; pip install torch, diffusers):
```
ggc v1
```
#### Video 2 (text to video)
Activate backend and frontend by (optional: need torch, diffusers to work; pip install torch, diffusers):
```
ggc v2
```
#### Image 2 (text to image)
Activate backend and frontend by (optional: need torch, diffusers to work; pip install torch, diffusers):
```
ggc i2
```
#### Speech 2 (text to voice)
Activate backend and frontend by (optional: need diao to work; pip install diao):
```
ggc s2
```
### Menu
Enter the main menu for selecting a connector or getting pre-trained trial model(s).
```
ggc m
```
[<img src="https://raw.githubusercontent.com/calcuis/gguf-connector/master/demo1.gif" width="350" height="300">](https://github.com/calcuis/gguf-connector/blob/main/demo1.gif)

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
#### Article
[understanding gguf and the gguf-connector](https://medium.com/@whiteblanksheet/understanding-gguf-and-the-gguf-connector-a-comprehensive-guide-3b1fc0f938ba)
#### Website
[gguf.org](https://gguf.org) (you can download the frontend from [github](https://github.com/chatpig/gguf.github.io) and host it locally; the backend is ethereum blockchain)

[gguf.io](https://gguf.io) (i/o is a mirror of us; note: this web3 domain might be restricted access in some regions/by some service providers, if so, visit the one below instead, exactly the same)

[gguf.us](https://gguf.us)
