from qdrant_client import QdrantClient
import hashlib
import json

client = QdrantClient(url="http://localhost:6333")

target_ids = {
    "RAG-001": "bf1bbdd11a6624ae230dca77ee51d3dc1bcefac611e97809d7d41d91f6332c9f",
    "RAG-006": "22fad2671963149b285417ee78e8fd2d61f7eca922fc55f46a3a19d65885e07e",
    "RAG-010": "f3afd871930c712c7def0a9c26e65bef4e1f72dd47eafa318be97332464b0f8b"
}

wanted = set(target_ids.values())
found = {}

offset = None

while True:
    points, offset = client.scroll(
        collection_name="Internal_Docs",
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    for point in points:
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        chunk_id = metadata.get("chunk_id")

        if chunk_id in wanted:
            content = payload.get("page_content", "")

            found.setdefault(chunk_id, []).append({
                "point_id": str(point.id),
                "chunk_id": chunk_id,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "metadata": metadata,
                "content_length": len(content),
                "content_hash": hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()
            })

    if offset is None:
        break

for question_id, chunk_id in target_ids.items():
    print("\n" + "=" * 90)
    print(question_id)
    print("Chunk ID:", chunk_id)

    records = found.get(chunk_id, [])

    print("Physical Qdrant points:", len(records))

    for index, record in enumerate(records, start=1):
        print(f"\nPoint {index}")
        print(json.dumps(record, indent=2, default=str))

    if len(records) > 1:
        hashes = {r["content_hash"] for r in records}
        print("\nIdentical content across points:", len(hashes) == 1)
