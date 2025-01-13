# from datasets import load_dataset

# dataset = load_dataset("klue/klue", "mrc", split="train[:100]")
# contexts = [ex["context"] for ex in dataset]

from pathlib import Path
from PyPDF2 import PdfReader

pdf_directory = Path("/data/ephemeral/home/level4-nlp-finalproject-hackathon-nlp-14-lv3/data/랩큐/네이버")
# pdf_directory = Path("/data/ephemeral/home/level4-nlp-finalproject-hackathon-nlp-14-lv3/data/랩큐")

# Recursively find all PDFs in subdirectories
contexts = []
for pdf_file in pdf_directory.rglob("*.pdf"):
    print(f"Processing: {pdf_file}")
    
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            contexts.append(text)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.create_documents(contexts)

embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

vector_store = FAISS.from_documents(split_docs, embeddings)
vector_store.save_local("faiss_klue_mrc")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_length=1024
))
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

query = "향후 5~10년간 기업이 집중할 전략은 무엇인가?"
response = qa_chain.run(query)
print(response)
