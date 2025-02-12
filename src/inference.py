import datetime
import gc
import json
import os
import random
from typing import List

import torch
from dotenv import load_dotenv
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.bm25retriever_with_scores import CustomizedOkapiBM25
from src.model import EvaluationOutput, GEvalResult, ServiceOutput, ValidationOutput
from src.my_hosted_llm import MyHostedLLM
from src.rule_retriever import RuleRetriever
from src.text_embedding import EmbeddingModel
from src.utils.load_engine import SqlEngine
from src.utils.utils import filter_by_company

load_dotenv()
random.seed(25)

torch.cuda.empty_cache()

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
model_openai = ChatOpenAI(model="gpt-4o-mini")
model_ollama = MyHostedLLM(url=os.environ["OLLAMA_URL"])

output_parser = StrOutputParser()
chain_openai = prompt | model_openai | output_parser
chain_ollama = prompt | model_ollama | output_parser

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

Your task:
1. Evaluate the retrieved contexts against the retrieval criteria and assign all 5 criteria scores as a list in order.
2. Evaluate the generated answer against the generation criteria and assign all 9 criteria scores as a list in order.
3. Return the scores in JSON format with the following keys: "retrieval_score" and "generation_score".

Below is the example of input data you need to evaluate:

- Query: 카카오뱅크의 3분기 대출성장률과 NIM(순이자마진)은 어떻게 변화했나요?
- Retrieved Contexts: 
  0. <context>Written By 카카오뱅크 at 2024-11-06, Content: 그래프 설명:

도표 3: 카카오뱅크 분기별 대출 증가율 추이

1. 축 정보:
   - X축: 분기 (2021년 1분기부터 2024년 3분기)
   - Y축: 대출액 (조원) 및 증가율 (QoQ, %)

2. 추세 및 패턴:
   - 대출액은 2021년 1분기부터 2024년 3분기까지 꾸준히 증가하는 추세를 보입니다.
   - 증가율은 2021년 3분기에 최고점을 기록한 후 감소 추세를 보이면서 안정적인 수준에서 변동합니다.

3. 핵심 포인트:
   - 대출액은 2023년 이후 40조 원 이상으로 유지됩니다.
   - 증가율은 2021년 3분기 15.7%로 최고점, 이후 점진적으로 감소하여 2% 이하로 유지됩니다.

4. 통찰:
   - 카카오뱅크의 대출 규모는 지속적으로 증가하고 있으며, 증가율이 어느 정도 안정화되고 있음을 보여줍니다.

도표 4: 카카오뱅크 NIM 추이

1. 축 정보:
   - X축: 분기 (2021년 1분기부터 2024년 3분기)
   - Y축: NIM (%)

2. 추세 및 패턴:
   - NIM은 2021년부터 2023년 3분기까지 변동이 있으며, 2022년 2분기에 최고점을 기록하고 그 후 하락하는 경향을 보입니다.

3. 핵심 포인트:
   - NIM의 최고점은 2022년 2분기 2.83%로 나타나며 이후 지속적으로 감소하여 2024년 3분기에는 2.15%로 기록됩니다.

4. 통찰:
   - 카카오뱅크의 NIM은 2022년 말 이후 하락세로 접어들었으며, 이에 따른 이자 수익성 관리가 필요할 것으로 보입니다.</context>
  1. <context>Written By 카카오뱅크 at 2024-05-08, Content: 그래프가 여러 개 포함되어 있습니다. 각 그래프에 대한 분석을 아래에 제공합니다:

# 그림4: 카카오뱅크의 순이자마진
1. 축 정보:
   - X축: 분기 (1Q21 ~ 1Q24)
   - Y축: 백분율 (%), 범위 1.5 ~ 3.5
2. 추세 및 패턴:
   - NIM과 NIS 모두 3Q22에서 최고점을 찍고 이후 하락추세.
3. 핵심 포인트:
   - NIM 최고점: 약 3.0
   - NIS 최고점 이후 하락, 1Q24에서 NIM은 2.55, NIS는 2.18
4. 통찰:
   - 카카오뱅크의 순이자마진이 3Q22 이후 감소하고 있음.

# 그림5: 카카오뱅크의 분기별 여수신 금리
1. 축 정보:
   - X축: 분기 (1Q21 ~ 1Q24)
   - Y축: 백분율 (%), 범위 60 ~ 120
2. 추세 및 패턴:
   - 예대율은 안정적인 수준에서 약간의 변동.
   - 조정 예대율은 감소추세.
3. 핵심 포인트:
   - 예대율 1Q24: 106
   - 조정 예대율 1Q24: 78
4. 통찰:
   - 조정된 예대율의 하락은 리스크 관리나 대출 정책 변화 가능성을 시사.

# 그림6: 카카오뱅크의 대출총량 규모 및 증가율
1. 축 정보:
   - X축: 분기 (1Q22 ~ 1Q24)
   - Y축1: 조원
   - Y축2: 증가율 (% QoQ)
2. 추세 및 패턴:
   - 주택관리와 기타일반 대출의 각각 다른 증가율 패턴.
3. 핵심 포인트:
   - 3Q22 이후 주택관리와 기타일반 모두 증가율 하락.
4. 통찰</context>
  2. <context>Written By 카카오뱅크 at 2024-05-09, Content: # 
카카오뱅크 (323410.KS)  
이제 대출성장보다 플랫폼 실적이 관건  
카카오뱅크는 금년 대출성장 목표를 20%에서 10%대 초반으로 하향. 이제 성장주로서 밸류에이션 정당화를 위해 대출성장을 대신할 플랫폼 성과 입증이 필요.  
대출성장 둔화 불가피, 이제는 플랫폼 성장 입증 필요  
카카오뱅크 투자의견 Buy, 목표주가 32,000원 유지. 전일 카카오뱅크는 금년 대출성장 목표를 기존 20%에서 10%대 초반으로 하향 조정. 1분기 대출성장이 6.9%였기 때문에, 2~4분기는 사실상 분기별 1~2% 정도인 시중 은행 수준 대출 성장 예상. 반면 여신보다 높은 수신 증가율을 용인하면서, 예대율은 현재 78%에서 70%대 초반까지 하락 예정. 여신을 초과하는 수신 자금은 수익증권 운용을 통해 수익을 창출할 계획.  
아울러 사측은 향후 대출성장보다는 플랫폼 트래픽과 플랫폼/수수료 수익(비이자 이익) 강화에 집중하겠다는 입장. 가계부채 증가에 부담을 느끼는 금융당국 입장과 금융 플랫폼을 지향하는 카카오뱅크의 본질적인 목표를 고려한 선택.  
따라서 향후 관건 혹은 우려 요인은 매출(순이자+비이자)의 90% 이상을 차지하는 이자이익의 증가 둔화를 대신해 비이자이익, 플랫폼 트래픽이 얼마나 괄목할 만한 성장을 보일 수 있는가에 있음. 다행인 점은, 1분기 MAU, 수신 잔고 및 저원가성 예금 비중이 각각 전분기보다 9.1%, 12.3%, 1.3%p 개선되는 등 플랫폼 역량은 여전히 뛰어나다는 점임.  
카카오뱅크 1분기 순이익은 1,112억원으로 시장 컨센서스 부합. NIM은 2.18%로 다소 크게 하락했는데, 이는 예대율 하락과 대출/예금 리프라이싱 영향. 연체율(0.47%) 및 NPL비율(0.45%)은 시중 은행보다도 양호하게 관리되고 있음. </context>
- Generated Answer: 카카오뱅크의 3분기 대출 성장률은 감소 추세를 보이고 있으며, 2023년 3분기 이후 대출액이 40조 원 이상으로 유지되고 있습니다. 또한, NIM(순이자마진)은 2022년 2분기에 최고점을 기록한 후 계속 하락하여 2024년 3분기에는 2.15%로 기록되었습니다. 따라서 3분기 대출 성장률과 NIM 모두 감소한 상황입니다.
- Ground Truth Answer: 카카오뱅크의 3분기 대출성장률은 0.8%에 그쳤으며, NIM(순이자마진)은 2.15%로 2bp 하락하였습니다.

Example of expected output:

{{
  "retrieval_score": [5, 5, 4, 3, 2],
  "generation_score": [4, 5, 5, 5, 3, 2, 0, 1, 0]
}}

Below is another example of input data you need to evaluate:

- Query: LG화학의 2024년 예상 영업이익은 얼마인가요?
- Retrieved Contexts: 
  0. <context>Written By LG화학 at 2024-09-25, Content: 그래프 설명은 다음과 같습니다:

1. LG화학: LG에너지솔루션 지분 가치를 제외한 시가총액 추이
   - 축 정보: X축은 시간(년/월), Y축은 조원(시가총액 범위 -10 ~ 20).
   - 추세 및 패턴: 2022년 초부터 2024년 중반까지 시가총액이 지속적으로 감소하는 추세를 보임.
   - 핵심 포인트: 2022년 이후 평균이 7.9조원, 2024년 7월경 4.0조원까지 감소.
   - 통찰: LG에너지솔루션 지분 제외 시 LG화학의 시가총액 감소 경향을 보여줌.

2. LG화학: 첨단소재 부문 영업이익 추이
   - 축 정보: X축은 분기(연도/분기), Y축은 십억 원 단위의 영업이익 (-100 ~ 500).
   - 추세 및 패턴: 2022년부터 2024년까지 영업이익의 변화, 특히 2022년 3분기부터 급증 후 다시 감소하는 패턴.
   - 핵심 포인트: 2022년 3분기에 최고점인 416억 원, 이후 감소세.
   - 통찰: 영업이익의 변동성을 보여주며, 양극재 사업과 기타 부문의 기여도 차별화.

3. LG화학: Forward P/E
   - 축 정보: X축은 연도(2015 ~ 2024), Y축은 지수(원).
   - 추세 및 패턴: 2022년을 정점으로 이후 하강 곡선.
   - 핵심 포인트: 2022년 최고점 후 2024년까지 감소.
   - 통찰: 주가 수익 비율의 과거 변동성을 반영하여, 향후 변동성을 예상할 수 있음.

4. LG화학: Forward P/B
   - 축 정보: X축은 연도(2015 ~ 202</context>
  1. <context>Written By LG화학 at 2024-10-28, Content: # 
              I   LG화학                                                                                           2024년 10월 29일
자료: 회사 자료, 신한투자증권</context>
  2. <context>Written By LG화학 at 2024-10-28, Content: # 
 I                  LG화학                                                                                              2024년 10월 29일
- Generated Answer: 2024년 예상 영업이익에 대한 정보는 제공된 내용에 포함되어 있지 않습니다. 따라서 구체적인 수치를 제공할 수 없습니다. 추가적인 자료나 정보가 필요합니다.
- Ground Truth Answer: LG화학의 2024년 예상 영업이익은 5.5조 원입니다.</context>

Example of expected output:

{{
  "retrieval_score": [0, 1, 1, 1, 1],
  "generation_score": [3, 4, 2, 5, 3, 1, 0, 1, 1]
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
chain_geval_openai = prompt_geval | model_openai | output_parser

embedding_model = EmbeddingModel()
embedding_model.load_model()

engine = SqlEngine()

sparse_retriever = CustomizedOkapiBM25()
sparse_retriever.load_retriever(engine, k=30)

reordering = LongContextReorder()

rule_retriever = RuleRetriever()
rule_retriever.load_rule()


def RAG(query: str, limit: int = 5, dpr_limit: int = 3) -> str:
    """
    query_embedding: 문자열 query의 임베딩 결과

    paragraph_ids: BM25 기반 문자열 query와 유사도가 높은 문서들의 paragraph_id 리스트
    - BM25 문서 갯수는 sparse_retriever.load_retriever(engine, k=5)에서 k값 조절

    ordered_ids: paragraph_id를 재정렬한 리스트
    - BM25 유사도가 높은 문서들 중에서 query_embedding 벡터와 유사도가 가장 높은 문서 순서대로
    - dpr_limit 갯수만큼만 문서를 가져옴

    contexts: 원본 텍스트들의 리스트
    - paragraph_id 리스트에서 재정렬를 실행하고 이에 대응하는 원본 텍스트를 가져옴
    """
    companies, rule_retrieved = rule_retriever.run_retrieval(query)
    # companies, years, rule_retrieved = rule_retriever.run_retrieval(query)

    contexts_spr = sparse_retriever.run_sparse_retrieval(query)
    contexts_spr = filter_by_company(contexts_spr, companies)
    # contexts_spr = filter_by_year(contexts_spr, companies)

    query_embedding = embedding_model.get_embedding(query)
    paragraph_ids = [doc.metadata["paragraph_id"] for doc in contexts_spr]
    contexts_dpr = embedding_model.run_dense_retrieval(
        engine, query_embedding, paragraph_ids
    )[:dpr_limit]
    contexts_dpr = filter_by_company(contexts_dpr, companies)

    contexts_reordered = reordering.transform_documents(contexts_dpr)

    return contexts_reordered


def run_inference(query: str) -> ServiceOutput:
    query = query.strip()
    try:
        if not query:
            raise ValueError("query is empty or only contains whitespace.")

        contexts = RAG(query)
        context = format_retrieved_docs(contexts)
        if len(context) == 0:
            return ServiceOutput({"text": "관련 문서를 찾지 못했습니다."})
        else:
            response = chain_openai.invoke({"question": query, "context": context})

        if not response or not response.strip():
            raise ValueError("The model returned an empty or invalid response.")

        # return {"text": response}
        return ServiceOutput(text=response)
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
        response = chain_openai.invoke({"question": query, "context": joined_context})

        if not response or not response.strip():
            raise ValueError("The model returned an empty or invalid response.")

        return {"context": contexts, "answer": response}
    except Exception as e:
        print(f"Error during inference: {e}")
        return "An error occurred during inference. Please try again later."


def compute_geval(
    chain2_geval: RunnableSerializable[dict, str],
    query: str,
    retrieval_contexts: str,
    generated_answer: str,
    expected_answer: str,
) -> dict:
    trivial_lg = ["\n{\n", "\n}\n"]
    openai_answer_length_max = 150
    try:
        while True:
            answer_geval = chain2_geval.invoke(
                {
                    "question": query,
                    "contexts": retrieval_contexts,
                    "answer": generated_answer,
                    "expected_answer": expected_answer,
                }
            )

            if len(answer_geval) > openai_answer_length_max:
                # This is LG model or weired response
                answer_geval = (
                    "{"
                    + answer_geval.split(trivial_lg[0])[1]
                    .split(trivial_lg[1])[0]
                    .strip()
                    + "}"
                )

            if answer_geval.find("```json") != -1:
                """
                Case of '```json\n{\n  "retrieval_score": [4, 5, 3, 3, 2],\n  "generation_score": [5, 4, 4, 5, 3, 2, 2, 1, 1]\n}\n```'
                """
                answer_geval = answer_geval.split("```json")[1].split("```")[0].strip()
            answer_geval = json.loads(answer_geval)
            retrieval_score, generation_score = (
                answer_geval["retrieval_score"],
                answer_geval["generation_score"],
            )

            if len(retrieval_score) == 5 and len(generation_score) == 9:
                retrieval_sum, generation_sum = sum(retrieval_score), sum(
                    generation_score
                )
                if (
                    retrieval_sum >= 0
                    and retrieval_sum <= 20
                    and generation_sum >= 0
                    and generation_sum <= 30
                ):
                    return {
                        "retrieval_score": retrieval_score,
                        "generation_score": generation_score,
                    }
    except torch.OutOfMemoryError:
        print(f"OutOFMemoryError {e=}, {type(e)=}")
        return {
            "retrieval_score": [-1, -1, -1, -1, -1],
            "generation_score": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        }
    except Exception as e:
        print(f"Unexpected {e=}, {type(e)=}")
        # 에러 발생시 실행결과 저장
        # json.dump({"detailed_results": [result.dict() for result in detailed_results]}, open("temp2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        return {
            "retrieval_score": [-1, -1, -1, -1, -1],
            "generation_score": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        }


def run_validation(
    chain1_rag: RunnableSerializable[dict, str],
    chain2_geval: RunnableSerializable[dict, str],
    train_test_ratio: str = 1,
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
        try:
            question = row["question"]
            expected_answer = row["answer"]

            context_docs = RAG(question)
            contexts, joined_contexts = format_retrieved_docs(
                context_docs, return_single_str=False
            )

            answer = chain1_rag.invoke(
                {"question": question, "context": joined_contexts}
            )
            # Sometimes the response may contain multiple markers; try to extract the answer.
            if "</user>" in answer:
                parts = answer.split("</user>")
                if len(parts) > 2:
                    answer = parts[2].strip()
                else:
                    answer = answer.strip()
            else:
                answer = answer.strip()

            predictions.append({"id": str(idx), "prediction_text": answer})
            references.append(
                {
                    "id": str(idx),
                    "answers": {"answer_start": [0], "text": [expected_answer]},
                }
            )

            eval_scores = compute_geval(
                chain2_geval,
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
        except Exception as e:
            print(f"Error in evaluation loop at index {idx}: {e}")
        finally:
            # Clear memory after each iteration to help avoid OOM issues.
            torch.cuda.empty_cache()
            gc.collect()

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
        text = f"""<context>Written By {doc.metadata['stockfirm_name']} at {doc.metadata['report_date']}, Content: {doc.page_content}</context>"""
        formatted_docs.append(text)
    joined_formatted_docs = "\n".join(formatted_docs)
    if return_single_str:
        return joined_formatted_docs
    else:
        return formatted_docs, joined_formatted_docs


if __name__ == "__main__":
    # run_validation(chain1_rag=chain_lg, chain2_geval=chain_geval_lg)
    # run_validation(chain1_rag=chain_openai, chain2_geval=chain_geval_openai)
    # run_validation(chain1_rag=chain_ollama, chain2_geval=chain_geval_openai)
    run_validation_without_retriever(
        chain1_rag=chain_openai_no_retriever,
        chain2_geval=chain_geval_openai_no_retriever,
    )
    # while True:
    #     query = input("질의를 입력해주세요: ")
    #     result = run_inference(query, chain=chain_ollama)
    #     print(result)
