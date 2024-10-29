from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def tokenize_function(examples):
    return tokenizer(examples["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)


model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰을 EOS 토큰으로 설정

model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = load_dataset("wikitext","wikitext-2-raw-v1", split="train")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Causal Language Model(생성 모델)로 사용
    inference_mode=False,          # 학습 모드
    r=8,                           # 저랭크 행렬의 랭크 설정
    lora_alpha=32,                 # LoRA의 alpha 값
    lora_dropout=0.1               # 드롭아웃 비율
)

lora_model = get_peft_model(model, lora_config)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./llama3.2-lora-finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=5e-5
)

# Trainer 설정
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# 모델 저장
trainer.save_model("./llama3.2-lora-finetuned")