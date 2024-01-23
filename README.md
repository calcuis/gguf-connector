## GGUF Connector

GGUF (GPT-Generated Unified Format) is a successor of GGML (GPT-Generated Model Language), it was released on August 21, 2023; by the way, GPT stands for Generative Pre-trained Transformer.

This package is a simple graphical user interface (GUI) application that uses the ctransformers or llama.cpp to interact with a chat model for generating responses. You can include the connector by simply:

```
import gguf_connector 
```

Pull any (pre-trained model) GGUF file(s) along with the Python code and it will automatically be detected by the module.

[<img src="https://raw.githubusercontent.com/calcuis/chatgpt-model-selector/master/demo.gif" width="350" height="280">](https://github.com/calcuis/chatgpt-model-selector/blob/main/demo.gif)
[<img src="https://raw.githubusercontent.com/calcuis/chatgpt-model-selector/master/demo1.gif" width="350" height="280">](https://github.com/calcuis/chatgpt-model-selector/blob/main/demo1.gif)

With this version, you can even choose which connector (either ctransformers or llama.cpp) to be used at the very beginning.

### Reference
https://github.com/calcuis/chatgpt-model-selector
