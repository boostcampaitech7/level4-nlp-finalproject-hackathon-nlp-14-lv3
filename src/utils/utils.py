import os
import re
from typing import List

import pandas as pd
from konlpy.tag import Mecab, Okt
from langchain_core.documents import Document
from sqlalchemy import RowMapping
from transformers import BertTokenizer

okt = Okt()
# mecab = Mecab()
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")


# 사용하지 않는 Document 주석 처리, 만일 사용할 때가 온다면 주석 제거 후 사용용
def create_documents(rows: List[RowMapping]):
    documents = []
    all_texts = get_info_and_texts(rows)
    for metadata, text in all_texts:
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def get_info_and_texts(texts: List[RowMapping]):
    texts_total = []
    for row in texts:
        paragraph_text = row["paragraph_text"]
        metadata = {
            "report_id": row["report_id"],
            "company_name": row["company_name"],
            "stockfirm_name": row["stockfirm_name"],
            "report_date": row["report_date"],
            "paragraph_id": row["paragraph_id"],
            "raw_text": paragraph_text,
        }
        texts_total.append((metadata, clean_korean_text(paragraph_text)))
    return texts_total


def clean_korean_text(text: str):
    # Example stopwords (customize for your domain)
    KOREAN_STOPWORDS = {
        "의",
        "가",
        "이",
        "은",
        "는",
        "엔",
        "에",
        "들",
        "와",
        "과",
        "도",
        "를",
        "으로",
        "자",
        "고",
        "수",
        "다",
        "로",
        "가서",
        "에선",
        "그리고",
        "하지만",
        "그래서",
        "그러나",
        "따라서",
        "즉",
        "때문에",
        "하여",  # etc.
    }

    # Remove special characters: keep only Korean (가-힣), English letters, numbers, and whitespace
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Morphological analysis with Okt
    # tokens = mecab.morphs(text, stem=True, norm=True)
    tokens = okt.morphs(text, stem=True, norm=True)
    # tokens = tokenizer.tokenize(text)
    # Remove stopwords
    filtered_tokens = [t for t in tokens if t not in KOREAN_STOPWORDS and t.strip()]

    # Re-join tokens
    cleaned_text = " ".join(filtered_tokens)
    return cleaned_text


def clean_text(text: str):
    return text.replace("\n", " ").replace("\r", "")
