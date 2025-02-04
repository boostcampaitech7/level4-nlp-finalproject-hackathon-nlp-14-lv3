import json
from collections import Counter

import openai


def tokenize(text):
    return text.lower().split()


# f1-score 계산
def calculate_f1_score(true_text, pred_text):
    true_tokens = tokenize(true_text)
    pred_tokens = tokenize(pred_text)
    true_counter = Counter(true_tokens)
    pred_counter = Counter(pred_tokens)
    common_tokens = list((true_counter & pred_counter).elements())
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(common_tokens) / len(true_tokens) if true_tokens else 0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1


# EM 계산
def calculate_exact_match(true_text, pred_text):
    return 1.0 if true_text.strip().lower() == pred_text.strip().lower() else 0.0


def evaluate_generation(groud_truth, generate_answer):
    precision, recall, f1 = calculate_f1_score(groud_truth, generate_answer)
    em = calculate_exact_match(groud_truth, generate_answer)
    return f1, em


def evaluate_rag(absolute_text, retrieved_docs):
    return absolute_text in retrieved_docs


### Geval
# RAG 평가
def evaluate_retrieval_geval(question, retrieved_docs, model="gpt-4o-mini"):
    prompt = f"""
    You are an AI assistant evaluating retrieval quality.
    Given the following question and retrieved documents, assign a relevance score from 1 to 5.
    
    Question: {question}
    Retrieved Documents:
    {retrieved_docs}
    
    Score (1-5) and brief explanation:
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful evaluation assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response["choices"][0]["message"]["content"]


# Generation 평가
def evaluate_generation_geval(
    question, retrieved_docs, generated_answer, model="gpt-4o-mini"
):
    prompt = f"""
    You are an AI assistant evaluating answer quality.
    Given the question, retrieved documents, and generated answer, assign an accuracy score from 1 to 5.
    
    Question: {question}
    Retrieved Documents:
    {retrieved_docs}
    
    Generated Answer:
    {generated_answer}
    
    Score (1-5) and brief explanation:
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful evaluation assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response["choices"][0]["message"]["content"]


def evaluate_rag_pipeline_geval(question, retrieved_docs, generated_answer):
    retrieval_score = evaluate_retrieval_geval(question, retrieved_docs)
    generation_score = evaluate_generation_geval(
        question, retrieved_docs, generated_answer
    )

    return {
        "retrieval_evaluation": retrieval_score,
        "generation_evaluation": generation_score,
    }


# 사용 예시 with f1, em
if __name__ == "__main__":
    results = []
    with open("C:\\test.json", "r") as f:
        data = json.load(f)
        # 우리가 생성된 결과들
        output_data = None
        for item, output in (zip(data["questions"], output_data),):
            question = item["question"]
            absolute_text = item["context"]
            ground_truth = item["answer"]
            # 밑에는 파일 구조에 맞게 수정 필요
            retrieved_docs = output["context"]
            generated_answer = output["answer"]
            rag_score = evaluate_rag(absolute_text, retrieved_docs)
            generation_score = evaluate_generation(ground_truth, generated_answer)
            print("rag score :", rag_score)
            print("eval score :", generation_score)
            results.append(
                {
                    "question": question,
                    "rag_score": rag_score,
                    "generation_score": generation_score,
                }
            )
