# first_run.py —— 你的第一个大模型推理（CPU 可跑）
from transformers import AutoModelForCausalLM, AutoTokenizer

# 选一个很小的模型（约 0.5B，CPU 也扛得住）。第一次会自动下载。
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)   # 默认 CPU

# 1) 把你的问题套上对话模板，再 tokenize（对应 P.1.2）
messages = [{"role": "user", "content": "用一句话解释什么是大模型推理。"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")             # 文字 → token 数字

print("输入被切成了", inputs.input_ids.shape[1], "个 token")  # 看看 P.1.2 的 token

# 2) 让模型生成（这一步内部就是 prefill + decode，对应 P.1.4）
outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

# 3) 把输出的 token 拼回文字（detokenize）
reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("模型回复：", reply)
