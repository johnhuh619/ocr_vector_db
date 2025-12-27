import psycopg

from app import build_embeddings, format_vector_literal, PG_CONN


if __name__ == "__main__":
    embeddings = build_embeddings()
    query = "practical llm formatting guidance"
    qvec = embeddings.embed_query(query)

    with psycopg.connect(PG_CONN.replace("postgresql+psycopg", "postgresql")) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT parent_id, view, lang, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM child_chunks
                WHERE view = 'code'
                ORDER BY embedding <=> %s::vector
                LIMIT 10;
                """,
                (format_vector_literal(qvec), format_vector_literal(qvec)),
            )
            results = cur.fetchall()

    print(f"[query] {query}")
    for parent_id, view, lang, content, sim in results:
        print("=" * 80)
        print(f"Parent: {parent_id}")
        print(f"View: {view}, Lang: {lang}")
        print(f"Similarity: {sim:.4f}")
        print("Content:\n" + content)
