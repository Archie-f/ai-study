import json

from config import GOLDEN_QA_PATH
from rag_notes.retrieval import search_with_rerank
from verify.verify_rerank_eval import get_retrieval_index_and_reranker


ANSWERABLE_IDS = ["q003", "q016", "q028", "q033", "q038", "q041", "q046", "q048"]

UNANSWERABLE_OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread at home?",
    "Who won the 2018 football World Cup?",
    "What is the boiling point of water at high altitude?",
]

UNANSWERABLE_NEAR_TOPIC = [
    "How does LoRA fine-tuning reduce the number of trained parameters?",
    "What is HyDE (hypothetical document embeddings) and when does it help?",
    "How do I shard a vector database across several machines?",
    "How do I deploy a containerised app to Kubernetes?",
]


def main():
    with open(GOLDEN_QA_PATH, "r") as f:
        golden_qa = json.load(f)


    queries = {
        "answerable": [golden["question"] for golden in golden_qa if golden["id"] in ANSWERABLE_IDS],
        "off-topic": UNANSWERABLE_OFF_TOPIC,
        "near-topic": UNANSWERABLE_NEAR_TOPIC
    }
    retrieval_index, reranker = get_retrieval_index_and_reranker()

    results = []
    for cat, qa_list in queries.items():
        result = [search_with_rerank(retrieval_index, reranker, q, mode="hybrid") for q in qa_list]
        cat_results = [(cat, res[0]) for res in result]
        results.extend(cat_results)

    heading = f"{'Category':<15} {'Highest Score':<15} {'Heading':<15}"
    body = [f"{r[0]:<15} {r[1][1]:<15.2f} {r[1][2].heading or '-':<15}" for r in results]
    body_text = "\n".join(body)
    table = "\n".join([heading, body_text])
    print(table)

    q048 = next(e for e in golden_qa if e["id"] == "q048")
    results = search_with_rerank(retrieval_index, reranker, q048["question"], top_k=20)

    for rank, (chunk_id, score, chunk) in enumerate(results, start=1):
        is_relevant = q048["answer_key"] in chunk.text
        print(rank, f"{score:.2f}", is_relevant, chunk.heading)

    res = search_with_rerank(retrieval_index, reranker, "What is the capital of France?", top_k=1)
    print(res[0][2].text[:200])


if "__main__" == __name__:
    main()
