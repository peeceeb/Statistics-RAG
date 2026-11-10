
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "Internal_Docs"
QDRANT_URL = "http://localhost:6333"

client = QdrantClient(
    url=QDRANT_URL,
    timeout=30
)

# 1. Capture existing collection information
before = client.get_collection(COLLECTION_NAME)

original_points = before.points_count

print("Before update:")
print("Points:", original_points)
print("Dense vectors:", before.config.params.vectors)
print("Sparse vectors:", before.config.params.sparse_vectors)

# 2. Add a sparse vector to the existing collection
client.create_vector_name(
    collection_name=COLLECTION_NAME,
    vector_name="sparse",
    vector_name_config=models.SparseVectorNameConfig(
        sparse=models.SparseVectorConfig()
    ),
    wait=True
)

# 3. Retrieve updated collection configuration
after = client.get_collection(COLLECTION_NAME)

print("\nAfter update:")
print("Points:", after.points_count)
print("Dense vectors:", after.config.params.vectors)
print("Sparse vectors:", after.config.params.sparse_vectors)

# 4. Verify collection preservation
assert after.points_count == original_points
assert "" in after.config.params.vectors
assert "sparse" in after.config.params.sparse_vectors

print("\nSUCCESS: Sparse vector schema added.")
print("Existing dense vectors and points preserved.")
