from llama_cpp import Llama
# llm = Llama(model_path="./models/7B/ggml-model.bin")
llm = Llama(model_path="/home/lc/code/ggml-model-q4_0.bin")
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
print(output)



# import llama_cpp
# import ctypes
# params = llama_cpp.llama_context_default_params()
# # use bytes for char * params
# ctx = llama_cpp.llama_init_from_file(b"./models/7b/ggml-model.bin", params)
# max_tokens = params.n_ctx
# # use ctypes arrays for array params
# tokens = (llama_cpp.llama_token * int(max_tokens))()
# n_tokens = llama_cpp.llama_tokenize(ctx, b"Q: Name the planets in the solar system? A: ", tokens, max_tokens, add_bos=llama_cpp.c_bool(True))
# llama_cpp.llama_free(ctx)

# python3 -m llama_cpp.server --model ~/code/ggml-model-q4_0.bin --main_gpu 0 --n_gpu_layers 35