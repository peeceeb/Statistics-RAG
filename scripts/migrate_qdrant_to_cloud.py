
import argparse
import getpass
import sys

from qdrant_client import QdrantClient, models


SOURCE_URL = "http://localhost:6334"
SOURCE_COLLECTION = "Internal_Docs_Rebuild"
TARGET_COLLECTION = "Internal_Docs_Rebuild"
ALIAS_NAME = "Internal_Docs"

EXPECTED_POINTS = 3966
BATCH_SIZE = 100


def connect_clients():
    print("Connecting to local Qdrant...")
    source = QdrantClient(url=SOURCE_URL, timeout=60)

    endpoint = input("Enter Qdrant Cloud cluster URL: ").strip()
    if not endpoint.startswith("https://"):
        raise ValueError("Cloud endpoint must start with https://")

    api_key = getpass.getpass("Enter Qdrant Cloud API key: ").strip()
    if not api_key:
        raise ValueError("API key cannot be empty.")

    target = QdrantClient(
        url=endpoint,
        api_key=api_key,
        timeout=120,
    )

    return source, target


def get_alias_mapping(client):
    response = client.get_aliases()
    return {
        alias.alias_name: alias.collection_name
        for alias in response.aliases
    }


def validate_source(source):
    if not source.collection_exists(SOURCE_COLLECTION):
        raise RuntimeError(
            f"Source collection '{SOURCE_COLLECTION}' does not exist."
        )

    info = source.get_collection(SOURCE_COLLECTION)

    if str(info.status).lower() != "green":
        raise RuntimeError(
            f"Source collection status is {info.status}, not GREEN."
        )

    if info.points_count != EXPECTED_POINTS:
        raise RuntimeError(
            f"Expected {EXPECTED_POINTS} source points, "
            f"found {info.points_count}. Stopping."
        )

    vectors = info.config.params.vectors
    sparse_vectors = info.config.params.sparse_vectors

    if not isinstance(vectors, dict) or "" not in vectors:
        raise RuntimeError(
            "Expected the source dense vector to use the default name ''."
        )

    dense_config = vectors[""]

    if dense_config.size != 3072:
        raise RuntimeError(
            f"Expected 3072-dimensional dense vectors; "
            f"found {dense_config.size}."
        )

    if "sparse" not in sparse_vectors:
        raise RuntimeError("Expected sparse vector configuration 'sparse'.")

    print(f"Source collection: {SOURCE_COLLECTION}")
    print(f"Source points: {info.points_count}")
    print(f"Dense dimensions: {dense_config.size}")
    print("Sparse vector: sparse")

    return info


def create_target_collection(target, source_info, resume):
    if target.collection_exists(TARGET_COLLECTION):
        if not resume:
            raise RuntimeError(
                f"Cloud collection '{TARGET_COLLECTION}' already exists. "
                "Use --resume only if this is an interrupted migration."
            )

        info = target.get_collection(TARGET_COLLECTION)

        if info.points_count > EXPECTED_POINTS:
            raise RuntimeError(
                "Cloud collection contains more points than expected. "
                "Stopping to avoid modifying an unexpected dataset."
            )

        vectors = info.config.params.vectors
        sparse_vectors = info.config.params.sparse_vectors

        if (
            not isinstance(vectors, dict)
            or "" not in vectors
            or vectors[""].size != 3072
            or "sparse" not in sparse_vectors
        ):
            raise RuntimeError(
                "Existing cloud collection configuration does not match "
                "the expected dense/sparse configuration."
            )

        print(
            f"Resuming existing cloud collection "
            f"({info.points_count} points already present)."
        )
        return

    aliases = get_alias_mapping(target)

    if ALIAS_NAME in aliases:
        raise RuntimeError(
            f"Cloud alias '{ALIAS_NAME}' already exists and points to "
            f"'{aliases[ALIAS_NAME]}'. Stopping."
        )

    print("Creating cloud collection with source configuration...")

    params = source_info.config.params

    target.create_collection(
        collection_name=TARGET_COLLECTION,
        vectors_config=params.vectors,
        sparse_vectors_config=params.sparse_vectors,
        shard_number=params.shard_number,
        replication_factor=params.replication_factor,
        write_consistency_factor=params.write_consistency_factor,
        on_disk_payload=params.on_disk_payload,
        hnsw_config=models.HnswConfigDiff.model_validate(
        source_info.config.hnsw_config.model_dump(exclude_none=True)
        ),
        optimizers_config=models.OptimizersConfigDiff.model_validate(
        source_info.config.optimizer_config.model_dump(exclude_none=True)
        ),
        wal_config=models.WalConfigDiff.model_validate(
        source_info.config.wal_config.model_dump(exclude_none=True)
        ),
        quantization_config=source_info.config.quantization_config,
        timeout=120,
    )

    print("Cloud collection created.")


def transfer_points(source, target):
    offset = None
    transferred_this_run = 0

    while True:
        points, next_offset = source.scroll(
            collection_name=SOURCE_COLLECTION,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        target.upsert(
            collection_name=TARGET_COLLECTION,
            points=[
                models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload,
                )
                for point in points
            ],
            wait=True,
            timeout=120,
        )

        transferred_this_run += len(points)
        offset = next_offset

        current_count = target.count(
            collection_name=TARGET_COLLECTION,
            exact=True,
        ).count

        print(
            f"Transferred this run: {transferred_this_run} | "
            f"Cloud points: {current_count}/{EXPECTED_POINTS}"
        )

        if next_offset is None:
            break

    return transferred_this_run


def verify_migration(source, target):
    source_count = source.count(
        collection_name=SOURCE_COLLECTION,
        exact=True,
    ).count

    target_count = target.count(
        collection_name=TARGET_COLLECTION,
        exact=True,
    ).count

    print("\nVerifying migration...")
    print(f"Local source points: {source_count}")
    print(f"Cloud target points: {target_count}")

    if source_count != EXPECTED_POINTS or target_count != source_count:
        raise RuntimeError(
            "Point counts do not match. Alias will NOT be created."
        )

    sample, _ = target.scroll(
        collection_name=TARGET_COLLECTION,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )

    if not sample:
        raise RuntimeError("Cloud sample verification returned no points.")

    point = sample[0]
    vectors = point.vector

    if not isinstance(vectors, dict):
        raise RuntimeError("Expected named dense and sparse vectors.")

    if "" not in vectors or "sparse" not in vectors:
        raise RuntimeError(
            "Cloud sample is missing dense or sparse vector data."
        )

    dense_vector = vectors[""]
    sparse_vector = vectors["sparse"]

    if len(dense_vector) != 3072:
        raise RuntimeError("Cloud dense vector dimension mismatch.")

    if not sparse_vector.indices:
        raise RuntimeError("Cloud sparse vector has no indices.")

    if not point.payload or "page_content" not in point.payload:
        raise RuntimeError("Cloud sample payload is incomplete.")

    metadata = point.payload.get("metadata", {})
    required_metadata = {
        "source",
        "document_id",
        "page",
        "chunk_id",
        "chunk_index",
    }

    missing = required_metadata - set(metadata.keys())
    if missing:
        raise RuntimeError(
            f"Cloud sample is missing metadata fields: {sorted(missing)}"
        )

    print("Sample point verification passed.")
    print("Dense vector: 3072 dimensions")
    print(f"Sparse vector indices: {len(sparse_vector.indices)}")
    print("Payload and metadata: present")


def create_alias(target):
    aliases = get_alias_mapping(target)

    if ALIAS_NAME in aliases:
        if aliases[ALIAS_NAME] == TARGET_COLLECTION:
            print(
                f"Alias '{ALIAS_NAME}' already points to "
                f"'{TARGET_COLLECTION}'."
            )
            return

        raise RuntimeError(
            f"Alias '{ALIAS_NAME}' already points to "
            f"'{aliases[ALIAS_NAME]}'."
        )

    target.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=TARGET_COLLECTION,
                    alias_name=ALIAS_NAME,
                )
            )
        ]
    )

    print(
        f"Created alias '{ALIAS_NAME}' -> '{TARGET_COLLECTION}'."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Migrate local Qdrant collection to Qdrant Cloud."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing incomplete cloud collection.",
    )
    args = parser.parse_args()

    source = None
    target = None

    try:
        source, target = connect_clients()

        print("\nChecking cloud connectivity...")
        print("Cloud collections:", target.get_collections().collections)

        source_info = validate_source(source)

        cloud_aliases = get_alias_mapping(target)
        if ALIAS_NAME in cloud_aliases:
            if cloud_aliases[ALIAS_NAME] != TARGET_COLLECTION:
                raise RuntimeError(
                    f"Cloud alias '{ALIAS_NAME}' points to "
                    f"'{cloud_aliases[ALIAS_NAME]}'."
                )

        create_target_collection(
            target,
            source_info,
            resume=args.resume,
        )

        print("\nStarting point transfer...")
        transfer_points(source, target)

        verify_migration(source, target)

        print("\nAll checks passed. Creating application alias...")
        create_alias(target)

        print("\nMigration completed successfully.")
        print(f"Cloud physical collection: {TARGET_COLLECTION}")
        print(f"Application alias: {ALIAS_NAME}")

    except KeyboardInterrupt:
        print(
            "\nMigration interrupted. The local source is unchanged. "
            "If the cloud collection was created, rerun with --resume."
        )
        sys.exit(1)

    except Exception as exc:
        print(f"\nMIGRATION STOPPED: {exc}")
        print(
            "The local source was not modified. "
            "If a partial cloud collection exists, inspect it before "
            "rerunning with --resume."
        )
        sys.exit(1)

    finally:
        if source is not None:
            source.close()
        if target is not None:
            target.close()


if __name__ == "__main__":
    main()
