from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
import torch
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader


def dpo_loss(preferred_logits, dispreferred_logits):
    # 손실 계산 (여기서는 간단한 로그우도를 기반으로)
    return -torch.mean(torch.log(torch.sigmoid(preferred_logits - dispreferred_logits)))

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("Anthropic/hh-rlhf",split="train")
num_epochs = 8

optimizer = AdamW(model.parameters(), lr=1e-4)
num_training_steps = len(dataset) * num_epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model.train()
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        inputs = tokenizer(batch["chosen"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        preferred_outputs = tokenizer(batch["chosen"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        dispreferred_outputs = tokenizer(batch["rejected"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inputs = {key: value for key, value in inputs.items() if key != "token_type_ids"}
        preferred_logits = model(**inputs, labels=preferred_outputs["input_ids"]).logits
        dispreferred_logits = model(**inputs, labels=dispreferred_outputs["input_ids"]).logits
        print(f"Preferred logits mean: {preferred_logits.mean().item()}")
        print(f"Dispreferred logits mean: {dispreferred_logits.mean().item()}")
        print(f"Logit difference: {torch.mean(preferred_logits - dispreferred_logits).item()}")


        loss = dpo_loss(preferred_logits, dispreferred_logits)

        # 역전파 및 최적화
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()


        print(f"Epoch {epoch}, Loss: {loss.item()}")    

model.save_pretrained("./dpo-finetuned-model")
tokenizer.save_pretrained("./dpo-finetuned-model")    
