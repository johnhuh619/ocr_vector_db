from shared.config import load_config
from embedding import EmbeddingProviderFactory

# Load config and create embedding client
config = load_config()
print(f"Provider: {config.embedding_provider}")
print(f"Model: {config.embedding_model}")
print(f"Dim: {config.embedding_dim}")

embeddings = EmbeddingProviderFactory.create(config)
print(f"Client type: {type(embeddings).__name__}")

# Test embedding
query = "채팅모델"
query_vec = embeddings.embed_query(query)
print(f"\nQuery: '{query}'")
print(f"Embedding dim: {len(query_vec)}")
print(f"First 5 values: {query_vec[:5]}")
