numactl --cpunodebind 0 --membind 0 --physcpubind=0-31 deepsparse.benchmark hf:neuralmagic/Llama2-7b-chat-pruned50-quant-ds -x benchmark -b 1 
