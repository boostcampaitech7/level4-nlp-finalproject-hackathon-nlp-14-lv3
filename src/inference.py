import datetime
import json
import os
import random
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.bm25retriever_with_scores import CustomizedOkapiBM25
from src.model import EvaluationOutput, GEvalResult, ServiceOutput, ValidationOutput
from src.text_embedding import EmbeddingModel  # 임베딩 모델 (bge-m3) 불러오기
from src.utils.load_engine import SqlEngine

load_dotenv()
random.seed(42)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please respond to the user's request only based on the given context.",
        ),
        ("user", "Context: {context}\nQuestion: {question}\nAnswer:"),
    ]
)
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()
chain = prompt | model | output_parser


prompt_geval = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an evaluation assistant tasked with assessing the quality of both the retrieval and generation stages of a QA system. Your evaluation is based on the following weighted criteria:

[Retrieval Evaluation - Total 20 points]
1. (5 points) Do any of the retrieved contexts show strong similarity to the Ground Truth?
2. (5 points) Do the retrieved contexts collectively capture essential information from the Ground Truth?
3. (4 points) Do the retrieved contexts sufficiently address the user’s question?
4. (3 points) Are all retrieved contexts relevant to the Ground Truth or the user’s query?
5. (3 points) Does the combined length and number of retrieved contexts remain reasonable without overwhelming the user with excessive or irrelevant details?

[Generation Evaluation - Total 30 points]
1. (5 points) Is the final answer clearly relevant to the question and reflective of the user’s intent?
2. (5 points) Is the answer factually correct and free from unsupported or inaccurate information?
3. (5 points) Does the answer include all essential points required by the question and the ground truth answer?
4. (5 points) Is the answer clear and concise, avoiding unnecessary repetition or ambiguity?
5. (3 points) Is the answer logically structured, consistent with the context, and free of contradictions?
6. (3 points) Does the answer provide sufficient detail for the question without being excessive?
7. (2 points) Does the answer provide proper citations or indications of the source when claims or data are referenced?
8. (1 point) Is the answer presented in a suitable format (list, table, short text, etc.) for the question?
9. (1 point) Does the answer offer any helpful extra insights or context that enrich the user’s understanding (without deviating from factual correctness)?

Below is the input data you need to evaluate:

- Query: <THE USER QUERY HERE>
- Retrieved Contexts: 
  1. <FIRST CONTEXT HERE>
  2. <SECOND CONTEXT HERE>
  ... (more if available)
- Generated Answer: <THE GENERATED ANSWER HERE>
- Ground Truth Answer: <THE GROUND TRUTH ANSWER HERE>

Your task:
1. Evaluate the retrieved contexts against the retrieval criteria and assign all 5 criteria scores as a list in order.
2. Evaluate the generated answer against the generation criteria and assign all 9 criteria scores as a list in order.
3. Return the scores in JSON format with the following keys: "retrieval_score" and "generation_score".

Example of expected output:

{{
  "retrieval_score": [4, 4, 4, 3, 3],
  "generation_score": [5, 5, 5, 5, 2, 1, 2, 1, 1]
}}

Please consider all the criteria provided and output only the JSON result.
""",
        ),
        (
            "user",
            "- Query: {question}\n- Retrieved Contexts: {contexts}\nGenerated Answer: {answer}\nGround Truth Answer: {expected_answer}",
        ),
    ]
)
chain_geval = prompt_geval | model | output_parser

embedding_model = EmbeddingModel()
embedding_model.load_model()

engine = SqlEngine()

sparse_retriever = CustomizedOkapiBM25()
sparse_retriever.load_retriever(engine, k=10)


def RAG(query: str, spr_limit: int = 10, dpr_limit: int = 3) -> str:
    """
    query_embedding: 문자열 query의 임베딩 결과

    paragraph_ids: BM25 기반 문자열 query와 유사도가 높은 문서들의 paragraph_id 리스트
    - BM25 문서 갯수는 sparse_retriever.load_retriever(engine, k=5)에서 k값 조절

    ordered_ids: paragraph_id를 재정렬한 리스트
    - BM25 유사도가 높은 문서들 중에서 query_embedding 벡터와 유사도가 가장 높은 문서 순서대로
    - dpr_limit 갯수만큼만 문서를 가져옴

    contexts: 원본 텍스트들의 리스트
    - paragraph_id 리스트에서 재정렬를 실행하고 이에 대응하는 원본 텍스트를 가져옴



    의문점)
    지금은 임베딩 벡터를 원본 텍스트에서 숫자만 제거한거로 만들고 있는데
    임베딩 하기전에 한국어 전처리로 단어제거(불용어 등), 특수문자 제거, stem 단어 변환 등을 하는게 더 좋을 것 같아요 -> 비교를 해보죠 저의들의 평가 데이터셋으로 제거하는 과정은 얼마걸리지 않으니깐

    """
    query_embedding = embedding_model.get_embedding(query)
    paragraph_ids = sparse_retriever.get_pids_from_sparse(query)
    ordered_ids = run_dense_retriever(query_embedding, paragraph_ids)
    contexts = sparse_retriever.get_contexts_by_ids(ordered_ids, dpr_limit)
    return contexts


def run_inference(query: str) -> ServiceOutput:
    query = query.strip()
    try:
        if not query:
            raise ValueError("query is empty or only contains whitespace.")

        contexts = RAG(query)
        context = format_retrieved_docs(contexts)
        response = chain.invoke({"question": query, "context": context})

        if not response or not response.strip():
            raise ValueError("The model returned an empty or invalid response.")

        return {"text": response}
    except Exception as e:
        print(f"Error during inference: {e}")
        return "An error occurred during inference. Please try again later."


def run_evaluation(query: str) -> EvaluationOutput:
    query = query.strip()
    try:
        if not query:
            raise ValueError("query is empty or only contains whitespace.")

        context = RAG(query)
        contexts, joined_context = format_retrieved_docs(
            context, return_single_str=False
        )
        response = chain.invoke({"question": query, "context": joined_context})

        if not response or not response.strip():
            raise ValueError("The model returned an empty or invalid response.")

        return {"context": contexts, "answer": response}
    except Exception as e:
        print(f"Error during inference: {e}")
        return "An error occurred during inference. Please try again later."


def compute_geval(
    query: str, retrieval_contexts: str, generated_answer: str, expected_answer: str
) -> dict:
    try:
        answer_geval = chain_geval.invoke(
            {
                "question": query,
                "contexts": retrieval_contexts,
                "answer": generated_answer,
                "expected_answer": expected_answer,
            }
        )
        # answer_geval = json.dumps({
        #     "retrieval_score": random.randint(0, 20),
        #     "generation_score": random.randint(0, 30)
        # }, indent=4)
        answer_geval = json.loads(answer_geval)
        retrieval_score, generation_score = (
            answer_geval["retrieval_score"],
            answer_geval["generation_score"],
        )
        total_score = retrieval_score + generation_score
        return {
            "retrieval_score": retrieval_score,
            "generation_score": generation_score,
            "total_score": total_score,
        }
    except:
        return {"retrieval_score": -20, "generation_score": -30, "total_score": -50}


def run_validation(
    train_test_ratio: str = 0.01,
    path_to_datasets: str = os.path.join(os.getcwd(), "data/datasets.json"),
) -> ValidationOutput:
    import evaluate
    from datasets import load_dataset

    ratio = float(train_test_ratio)

    dataset = load_dataset("json", data_files=path_to_datasets, field="qa_sets")
    full_ds = list(dataset["train"])
    random.shuffle(full_ds)
    split_idx = int(len(full_ds) * ratio)
    test_ds = full_ds[:split_idx]

    squad_metric = evaluate.load("squad")

    predictions, references, detailed_results = [], [], []

    for idx, row in enumerate(test_ds):
        question = row["question"]
        expected_answer = row["answer"]
        # expected_contexts = row["context"]
        # source = row["source"]

        context = RAG(question)
        contexts, joined_contexts = format_retrieved_docs(
            context, return_single_str=False
        )

        # answer = "This is sample answer"
        answer = chain.invoke({"question": question, "context": joined_contexts})

        predictions.append({"id": str(idx), "prediction_text": answer})

        references.append(
            {
                "id": str(idx),
                "answers": {"answer_start": [0], "text": [expected_answer]},
            }
        )

        eval_scores = compute_geval(
            question,
            "\n".join([f"  {i}. " + context for i, context in enumerate(contexts)]),
            answer,
            expected_answer,
        )
        retrieval_score = eval_scores["retrieval_score"]
        generation_score = eval_scores["generation_score"]
        total_score = sum(retrieval_score) + sum(generation_score)
        detailed_results.append(
            GEvalResult(
                query=question,
                retrieval_contexts=contexts,
                generated_answer=answer,
                expected_answer=expected_answer,
                retrieval_score=retrieval_score,
                generation_score=generation_score,
                total_score=total_score,
            )
        )

    # Compute overall SQuAD metrics.
    squad_results = squad_metric.compute(predictions=predictions, references=references)
    f1_score = squad_results.get("f1")
    exact_match = squad_results.get("exact_match")

    final_results = {
        "squad_f1": f1_score,
        "squad_exact_match": exact_match,
        "detailed_results": [result.dict() for result in detailed_results],
    }

    datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"output/{datetime_now}")
    os.makedirs(output_dir, exist_ok=True)
    validation_output_path = os.path.join(output_dir, "validation_results.json")
    with open(validation_output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print("Validation completed.")
    print(f"SQuAD F1: {f1_score:.2f}, Exact Match: {exact_match:.2f}")
    print("Detailed results saved to validation_results.json")


def format_retrieved_docs(docs: List[str], return_single_str=True):
    # RAG된 docs Formatting 진행
    formatted_docs = []
    for doc in docs:
        text = f"""Company: {doc['company_name']} Date: {doc['report_date']} Content: {doc['paragraph_text']}"""
        formatted_docs.append(text)
    joined_formatted_docs = "\n---\n".join(formatted_docs)
    if return_single_str:
        return joined_formatted_docs
    else:
        return formatted_docs, joined_formatted_docs


def run_dense_retriever(query_vector: List[float], paragraph_ids: List[str]):
    vector_array = f"[{','.join(str(x) for x in query_vector)}]"
    joined_ids = ", ".join(f"'{str(u)}'" for u in paragraph_ids)
    query = text(
        f"""
SELECT paragraph_id 
FROM embedding
WHERE paragraph_id in ({joined_ids})
ORDER BY clean_text_embedding_vector <-> '{vector_array}'::vector"""
    )
    results = engine.conn.execute(query)
    return [row for row in results.mappings()]


if __name__ == "__main__":
    run_validation()
    # while True:
    # query = input("질의를 입력해주세요: ")
    # result = run_inference(query)
    # print(result)
