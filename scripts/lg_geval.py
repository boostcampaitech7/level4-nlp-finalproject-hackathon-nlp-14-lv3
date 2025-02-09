import datetime
import json
import os

import evaluate

from src.inference import chain_geval_openai, compute_geval
from src.model import GEvalResult

with open("output/lg_lg.json", "r") as f:
    results = json.load(f)["detailed_results"]
    predictions, references, detailed_results = [], [], []

    for idx, result in enumerate(results):
        question = result["query"]
        contexts = result["retrieval_contexts"]
        answer = result["generated_answer"]
        expected_answer = result["expected_answer"]
        predictions.append({"id": str(idx), "prediction_text": answer})
        references.append(
            {
                "id": str(idx),
                "answers": {"answer_start": [0], "text": [expected_answer]},
            }
        )

        eval_scores = compute_geval(
            chain_geval_openai,
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
    squad_metric = evaluate.load("squad")
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
