from qdrant_client import QdrantClient
from pprint import pprint

client = QdrantClient(
    url="http://localhost:6333"
)

collection_name = "Internal_Docs"

points, next_offset = client.scroll(
    collection_name=collection_name,
    limit=2,
    with_payload=True,
    with_vectors=False
)

for point in points:

    print("\nPoint ID:", point.id)

    print("\nPayload:")
    pprint(point.payload)

    print("\nPayload keys:")
    print(list(point.payload.keys()))

client.close()