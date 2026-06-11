import os

import torch
from gptqmodel import GPTQModel
from transformers import AutoTokenizer


MODEL_PATH = os.environ.get("MODEL_PATH", "../../../models/gemma-3-270m-it/")
QUANT_PATH = os.environ.get("QUANT_PATH", "../gemma-3-270m-it-gptq-int4")
QUESTION = os.environ.get("QUESTION", "请用中文介绍一下人工智能的发展历程。")
DEVICE = os.environ.get("DEVICE", "cuda:0")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def system_prompt(question: str) -> str:
    if contains_cjk(question):
        return "你是一个有用的AI助手，请使用和用户问题相同的语言回答。如果用户使用中文提问，请用简体中文回答；如果用户使用英文提问，请用英文回答。不要混合语言。"
    return "You are a helpful AI assistant. Answer in the same language as the user's question. If the user asks in Chinese, respond in Simplified Chinese; if the user asks in English, respond in English. Do not mix languages."


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

messages = [
    {
        "role": "system",
        "content": system_prompt(QUESTION),
    },
    {
        "role": "user",
        "content": QUESTION,
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(DEVICE)

print("input_ids", inputs["input_ids"].tolist(), inputs["input_ids"].shape)
print("input_ids.device", inputs["input_ids"].device)

model = GPTQModel.load(QUANT_PATH, device_map={"": DEVICE})

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.12,
        no_repeat_ngram_size=8,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
print("generated_ids_trimmed", generated_ids_trimmed)
output_text = tokenizer.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print("Generated response:", output_text[0])