import json
from collections import defaultdict

from config import EVAL_CORPUS_PATH, PERSIST_PATH, GOLDEN_QA_PATH
from rag_notes.adapters import ChromaVectorStore
from rag_notes.bm25_index import bm25_search
from rag_notes.embedder import load_embedding_model
from rag_notes.hybrid_search import build_rank_map
from rag_notes.pipeline import make_default_chunker
from rag_notes.retrieval import build_retrieval_index
from rag_notes.retrieval_metrics import find_relevant_chunk_ids
from verify.verify_first_stage_recall import first_rank

CONFIGS = [
    # (name, depth, cut, weights)
    ("baseline d20 c20", 20, 20, [1.0, 1.0]),
    ("depth 100, cut 20", 100, 20, [1.0, 1.0]),
    ("depth 100, cut 30", 100, 30, [1.0, 1.0]),
    ("depth 100, cut 40", 100, 40, [1.0, 1.0]),
    ("depth 100, dense x2", 100, 20, [2.0, 1.0]),
    ("depth 100, dense x3", 100, 20, [3.0, 1.0]),
]


def weighted_rrf_merge(
        rank_maps: list[dict[str, int]],
        weights: list[float],
        k: int = 60,
) -> list[tuple[str, float]]:
    """Merge id -> rank maps like rrf_merge, but each ranker counts weight times as much.

    Args:
        rank_maps: one dict per ranker, each id -> 1-based rank
        weights: one weight per ranker, same order as rank_maps; must all be > 0
        k: RRF damping constant
    Returns:
        (chunk_id, score) tuples, highest score first; equal scores ordered by
        chunk_id ascending. Score = sum(weight / (k + rank)) over the rankers
        that contain the chunk.
    Raises:
        ValueError: if rank_maps and weights have different lengths, or a weight is <= 0
    """
    if len(rank_maps) != len(weights):
        raise ValueError("rank_maps and weights must have the same length.")

    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be strictly greater than 0.")

    scores = defaultdict(float)
    for rank_map, weight in zip(rank_maps, weights):
        for chunk_id, rank in rank_map.items():
            scores[chunk_id] += weight / (k + rank)
    sorted_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    return sorted_scores


def build_candidates(
        dense_ids: list[str],
        sparse_ids: list[str],
        depth: int,
        cut: int,
        weights: list[float] | None = None,
) -> list[str]:
    """Pick the candidate ids that would go to the reranker.

    Takes the first `depth` ids of each ranked list, merges them with
    weighted_rrf_merge (weights default to [1.0, 1.0]: dense, sparse),
    and returns the first `cut` ids of the merged list.
    """
    #Check depth and cut values are more than 0
    if depth <= 0 or cut <= 0:
        raise ValueError(f"'depth' and 'cut' must be strictly greater than zero. But found depth: {depth} and cut: {cut}")

    #Set default weights (Default to [1.0, 1.0])
    if weights is None:
        weights = [1.0, 1.0]

    #Take the first `depth` ids of each ranked list
    first_depth_number_of_dense = dense_ids[:depth]
    first_depth_number_of_sparse = sparse_ids[:depth]

    #Create id -> 1-based dict for the first `depth` number ranks
    dense_map = build_rank_map(first_depth_number_of_dense)
    sparse_map = build_rank_map(first_depth_number_of_sparse)

    #Weighted merge the two dicts
    merged_results = weighted_rrf_merge(
        rank_maps=[dense_map, sparse_map],
        weights=weights
    )

    #Return the chunk_ids of the first 'cut' number of merged results
    candidate_ids = [chunk_id for chunk_id, score in merged_results]
    return candidate_ids[:cut]


def main() -> None:
    """Count, per config, the questions whose right chunk is among the candidates.

    Fetches dense and BM25 lists (depth 100) once per question, then runs every
    config on the same lists. Prints one row per config: questions rescued,
    average candidates per question, and the ids of the questions still missed.
    """
    #Build the index
    model = load_embedding_model()
    vector_store = ChromaVectorStore(PERSIST_PATH)
    chunker = make_default_chunker(model)
    retrieval_index = build_retrieval_index(
        notes_root=EVAL_CORPUS_PATH,
        vector_store=vector_store,
        chunker=chunker,
        model=model
    )
    with open(GOLDEN_QA_PATH, "r") as f:
        golden_qa_list = json.load(f)
    total_questions = len(golden_qa_list)

    data_per_config = []
    for config in CONFIGS:
        total_candidates = 0
        rescued_questions = 0
        still_missed_ids = []

        #For each golden question, store (entry["id"], dense_ids, sparse_ids, relevant_ids) n_results=100
        stored_query_data = []
        for entry in golden_qa_list:
            query = entry["question"]
            query_id = entry["id"]
            dense_ids = vector_store.query(query, model, n_results=100)['ids'][0]
            sparse_results = bm25_search(query, retrieval_index.bm25_index, n_results=100)
            sparse_ids = [f"{chunk.source.title}-{chunk.chunk_index}" for _, chunk in sparse_results]
            relevant_ids = find_relevant_chunk_ids(retrieval_index.bm25_index.chunks, entry["answer_key"])
            stored_query_data.append((query_id, dense_ids, sparse_ids, relevant_ids))

            #For each config, for each stored question hit if any relevant id is in candidates
            candidates = build_candidates(
                dense_ids=dense_ids,
                sparse_ids=sparse_ids,
                depth=config[1],
                cut=config[2],
                weights=config[3]
            )
            total_candidates += len(candidates)

            if first_rank(candidates, relevant_ids) is not None:
                rescued_questions += 1
            else:
                still_missed_ids.append(query_id)

        data_per_config.append((config[0], rescued_questions, total_candidates, still_missed_ids))

    #Print one row per config
    print(f"{'Config Name':<20} | Rescued | Average Candidates | Still Missed")
    table = [f"{name:<20} | {rescued_questions}/{total_questions:<4} | {(num_candidates/total_questions):<18} | {', '.join(still_missed_ids)}"
             for (name, rescued_questions, num_candidates, still_missed_ids) in data_per_config]
    print(*table, sep="\n")


if __name__ == "__main__":
    main()