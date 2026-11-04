from qdrant_client import QdrantClient

client = QdrantClient(
    url="http://localhost:6333"
)

collection_name = "Internal_Docs"

info = client.get_collection(collection_name)

print("Collection:", collection_name)
print("Points count:", info.points_count)
print("Vector configuration:")
print(info.config.params.vectors)

print("\nSparse vector configuration:")
print(info.config.params.sparse_vectors)

print("\nQdrant version:")
print(client.info().version)