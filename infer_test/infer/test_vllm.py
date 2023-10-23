from vllm import LLM
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
# import torch
import time


prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
prompts = ["The capital of France is"]  # Sample prompts.

load_start = time.time()
# llm = LLM (model="lmsys/vicuna-7b-v1.3")  # Create an LLM.
llm = LLM(
    model="/home/lc/models/codellama/CodeLlama-7b-Instruct-hf",
    tensor_parallel_size=2,
    dtype="float16")  # Create an LLM.

load_done = time.time()
outputs = llm.generate (prompts)  # Generate texts from the prompts.
gen_done = time.time()
print(outputs)
output = outputs[0].outputs[0].text

print("load_time: ", load_done - load_start)
print("gen_time: ", gen_done - load_done)
print("gen_speed: ", len(output) / (gen_done - load_done))
print("tokens: ", output, len(output))

# 推理速度
# test_data = load_dataset("json", data_files="/home/lc/data/THUDM/humaneval-x/data/python/data//humaneval.jsonl")
# train_data = test_data['train']

# for data in train_data:
    # print(data)


# 显存占用




