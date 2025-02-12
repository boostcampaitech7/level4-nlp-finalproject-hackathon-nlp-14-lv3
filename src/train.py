import os
import datetime
import random
from typing import List

import torch
from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

seed = 25
max_length = 1_000
test_size = 0.1
num_train_epochs = 2

from dotenv import load_dotenv
load_dotenv()

random.seed(seed)
torch.manual_seed(seed)

model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,
    # device_map="auto",
    # torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = model.to("cuda")  # Ensure the model is fully on GPU
model.gradient_checkpointing_enable()
# Set model to training mode.
model.train()

dataset_path = os.path.join(os.getcwd(), "data/train.json")
raw_dataset = load_dataset("json", data_files=dataset_path, field="datasets")
full_dataset = raw_dataset["train"]

full_dataset = full_dataset.shuffle(seed=seed)
split_dataset = full_dataset.train_test_split(test_size=test_size, seed=seed)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

def format_retrieved_docs(docs: List[str], return_single_str=True):
    formatted_docs = []
    for doc in docs:
        text = f"""<context>Written By {doc['stockfirm_name']} at {doc['report_date']}, Content: {doc['page_content']}</context>"""
        formatted_docs.append(text)
    joined_formatted_docs = "\n".join(formatted_docs)
    if return_single_str:
        return joined_formatted_docs
    else:
        return formatted_docs, joined_formatted_docs

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a knowledgeable financial chatbot for Korean that assists users by synthesizing information from retrieved brokerage firm reports. When answering a user’s query, follow these steps:
Direct Answer: Provide a clear, concise, and factual response to the user's question, integrating relevant details from the retrieved reports.
Supporting Evidence: Reference key points from the retrieved content (such as report names, dates, and major findings) that support your answer.
Content Summary: At the end of your answer, include a brief summary that outlines:
The sources of the retrieved content (e.g., the names and dates of the reports)
The main points or insights extracted from those sources.
Limitations: Clearly state any limitations related to the retrieved content. For example, mention if the data might be outdated, incomplete, or if there are any inherent biases.
Disclaimer: End your response with a disclaimer stating that the information provided is for informational purposes only and should not be considered professional financial advice.

# Example input
<user>
Contexts: 
<context>CONTEXT 1 FOR ANSWERING QUESTION WELL</context>
<context>CONTEXT 2 FOR ANSWERING QUESTION WELL</context>
<context>CONTEXT 3 FOR ANSWERING QUESTION WELL</context>

Question: THE USER QUERY HERE

Answer:</user>

# Example output

YOUR ANSWER HERE
""",
        ),
        ("user", "<user>Context:\n{context}\nQuestion: {question}\nAnswer:</user>"),
    ]
)

def tokenize_function(example):
    # Build the prompt exactly as in your original code:
    # (You may adjust the system text if needed.)
    formatted_docs = format_retrieved_docs(example["context"])
    prompt_text = prompt.format(context=formatted_docs, question=example["question"])

    # The full text concatenates the prompt with the expected answer.
    full_text = prompt_text + example["answer"]
    
    # Tokenize the full text (with truncation as needed).
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length)
    
    # --- Mask out the prompt part from the loss ---
    # Tokenize the prompt only to determine its length.
    prompt_tokenized = tokenizer(prompt_text, truncation=True, max_length=max_length)
    prompt_length = len(prompt_tokenized["input_ids"])
    
    # Copy input_ids as labels and mask the prompt tokens with -100.
    labels = tokenized["input_ids"].copy()
    if prompt_length > len(labels):
        prompt_length = len(labels)
    labels[:prompt_length] = [-100] * prompt_length
    tokenized["labels"] = labels
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=False)
eval_dataset = eval_dataset.map(tokenize_function, batched=False)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

output_dir = os.path.join(os.getcwd(), "output", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    report_to=["wandb"],
    run_name="reader-training-run",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model"))
