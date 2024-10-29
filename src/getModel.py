from transformers import LlamaTokenizer, LlamaForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"  # 모델 이름 지정

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
