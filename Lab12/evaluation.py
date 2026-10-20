def evaluate_retrieval(
    results,
    expected_chunk_id
):
    retrieved_ids = [
        result["chunk_id"]
        for result in results
    ]

    return (
        expected_chunk_id in retrieved_ids
    )