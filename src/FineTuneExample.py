from transformers import AutoTokenizer,LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from huggingface_hub import HfApi
from datasets import load_dataset

def tokenize_function(examples):
    tokens=tokenizer(examples['text'],return_tensors='pt',padding='max_length',truncation=True,max_length=512)
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

# model_name = "meta-llama/Llama-3.2-1B"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = "albert/albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
print("model loaded")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset['train']
val_data = dataset['validation']
tokenizer.pad_token = tokenizer.eos_token

tokenized_train_data = train_data.map(tokenize_function, batched=True)
tokenized_val_data = val_data.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="./llama-finetuned",  # 모델이 저장될 경로
    evaluation_strategy="epoch",     # 평가 전략 (매 에포크마다 평가)
    learning_rate=2e-5,              # 학습률
    per_device_train_batch_size=1,   # 배치 크기
    per_device_eval_batch_size=1,    # 평가 배치 크기
    num_train_epochs=1,              # 학습 에포크 수
    weight_decay=0.01,               # 가중치 감쇠
    gradient_accumulation_steps = 8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_val_data,
)

trainer.train()

# 모델 저장
trainer.save_model("./llama-finetuned")  # 모델이 저장될 경로
tokenizer.save_pretrained("./llama-finetuned")