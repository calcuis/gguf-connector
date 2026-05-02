import os
os.system("vllm serve callgg/minimax-m2.7 \ --host 0.0.0.0 --port 8000 \ --served-model-name minimax-m2.7 \ --trust-remote-code \ --max-model-len 75776 \ --gpu-memory-utilization 0.85 \ --kv-cache-dtype fp8 \ --load-format fastsafetensors \ --enable-auto-tool-choice \ --tool-call-parser minimax_m2 \ --reasoning-parser minimax_m2_append_think")
