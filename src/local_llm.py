import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_lg = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model_lg.eval()

hf_pipeline = pipeline(
    "text-generation",
    model=model_lg,
    tokenizer=tokenizer,
    temperature=0.1,
    max_new_tokens=8196,
    do_sample=True,
    return_full_text=False,
)
pipeline_lg = HuggingFacePipeline(pipeline=hf_pipeline)
