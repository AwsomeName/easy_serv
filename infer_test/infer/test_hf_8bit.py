# 正常推理测试，速度验证，显存验证
# bnb，量化，等等
# 按照HF中的说法，int8不会下降精度，但是速度会下降 20% 左右


# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained(
#   '/home/lc/models/codellama/CodeLlama-7b-Instruct-hf',
#   device_map='auto',
#   load_in_8bit=True)
#   max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB')



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 128
model_name = '/home/lc/models/codellama/CodeLlama-7b-Instruct-hf'

text = 'Hamburg is in which country?\n'
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors="pt").input_ids

free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

print("max_memory: ", max_memory)

n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}

model = AutoModelForCausalLM.from_pretrained(
  model_name,
  device_map='auto',
  load_in_8bit=True,
  max_memory=max_memory
)
generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))


