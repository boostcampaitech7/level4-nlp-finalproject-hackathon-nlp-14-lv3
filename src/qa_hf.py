# from datasets import load_dataset

# dataset = load_dataset("klue/klue", "mrc", split="train[:100]")
# contexts = [ex["context"] for ex in dataset]

from pathlib import Path
from PyPDF2 import PdfReader

pdf_directory = Path("/data/ephemeral/level4-nlp-finalproject-hackathon-nlp-14-lv3/data/랩큐/네이버")
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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
split_docs = text_splitter.create_documents(contexts)

# 1. 먼저 문서가 비어있지 않은지 확인
print(f"문서 개수: {len(split_docs)}")

# 2. 문서 내용 확인
if split_docs:
    print(f"첫 번째 문서 샘플: {split_docs[0]}")

# 3. 안전하게 벡터 스토어 생성
try:
    if not split_docs:
        raise ValueError("문서가 비어있습니다!")
        
    embedding_model = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    print("벡터 스토어 생성 완료!")
    
except Exception as e:
    print(f"에러 발생: {e}")
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
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_length=1024
))
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)

while True:
    query = input()
    if query == '':
        break
    result = qa_chain.invoke({"query": query})
    print("\n=== 검색된 문서들 ===")
    for i, doc in enumerate(result['source_documents']):
        print(f"\n문서 {i+1}:")
        print(f"내용: {doc.page_content}")
        print(f"메타데이터: {doc.metadata}")
    print("\n=== 응답 ===")
    print(result['result'])
