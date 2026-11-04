from pathlib import Path
from itertools import islice
from typing import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import hashlib
from langchain_qdrant import QdrantVectorStore


class ProcessDocuments:
    def __init__(self,folder_path: str,
                      batch_size: int = 3,
                      chunk_size: int = 1000,
                      chunk_overlap: int = 400,
                      max_workers: int = 3):
        self.folder_path = Path(folder_path)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)

    # Step 1: Discover PDF files
    def get_pdf_files(self) -> Iterator[Path]:
        return self.folder_path.glob("*.pdf")

    # Step 2: Divide PDF files into batches
    def get_batches(self, files) -> Iterator[list[Path]]:
        while True:
            batch = list(islice(files, self.batch_size))
            if not batch:
                break
            yield batch

    # Step 3A: Load an individual PDF
    def load_single_pdf(self, pdf_file: Path) -> list[Document]:
        try:
            reader = PdfReader(str(pdf_file))
            document_id = hashlib.sha256(
                str(pdf_file.resolve()).encode("utf-8")
            ).hexdigest()
            documents = [Document(page_content=page.extract_text() or "",
                                  metadata={
                                      "source": str(pdf_file),
                                      "document_id": document_id,
                                      "page": page_number,
                                  }) for page_number, page in enumerate(reader.pages)]
            return documents
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            return []

    # Step 3B: Load multiple PDFs concurrently
    def load_documents(self,pdf_files: list[Path]) -> list[Document]:
        docs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.load_single_pdf,pdf_files)
            for documents in results:
                docs.extend(documents)
        return docs

    # Step 4: Split documents into chunks
    def split_documents(self,documents: list[Document]) -> list[Document]:
        chunks = self.text_splitter.split_documents(documents)

        chunk_counters={}

        for chunk in chunks:
            document_id=chunk.metadata["document_id"]
            page_number=chunk.metadata["page"]
            page_key = (document_id, page_number)
            chunk_index = chunk_counters.get(page_key, 0)
            chunk_counters[page_key] = chunk_index + 1
            chunk_identifier=(
                f"{document_id}:{page_number}:chunk_{chunk_index}:"
                f"{chunk.page_content}"
            )
            chunk_id = hashlib.sha256(chunk_identifier.encode("utf-8")).hexdigest()
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["chunk_index"] = chunk_index

        return chunks

    # Step 5: Sanitize chunks
    def sanitize_documents(self,documents: Iterable[Document]) -> list[Document]:
        cleaned_docs = []
        for doc in documents:
            safe_page_content = (doc.page_content.encode("utf-8", errors="replace").decode("utf-8"))
            safe_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, str):
                    safe_metadata[key] = (value.encode("utf-8", errors="replace").decode("utf-8"))
                else:
                    safe_metadata[key] = value
            cleaned_docs.append(Document(page_content=safe_page_content,metadata=safe_metadata))
        return cleaned_docs

    # Step 6: Process one batch end-to-end
    def process_batch(self,batch: list[Path]) -> list[Document]:
        start = perf_counter()
        
        # Load PDFs concurrently
        docs = self.load_documents(batch)
        load_time = perf_counter() - start
        
        # Split documents into chunks
        start = perf_counter()
        chunks = self.split_documents(docs)
        split_time = perf_counter() - start
        
        # Sanitize chunks
        start = perf_counter()
        cleaned_chunks = self.sanitize_documents(chunks)
        sanitize_time = perf_counter() - start
        print(f"PDFs processed: {len(batch)}")
        print(f"Pages loaded: {len(docs)}")
        print(f"Chunks generated: {len(cleaned_chunks)}")
        print(f"Loading time: {load_time:.2f}s")
        print(f"Splitting time: {split_time:.2f}s")
        print(f"Sanitization time: {sanitize_time:.2f}s")
        print(f"Total batch time: "f"{load_time + split_time + sanitize_time:.2f}s")

        return cleaned_chunks

    def count_embedding_tokens(self,documents: list, model: str) -> dict:
        try:
            encoding = tiktoken.encoding_for_model(model)
            encode = encoding.encode
        except KeyError:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model)
            encode = tokenizer.encode

        # Initialize the total token counter.
        total_tokens = 0

        # Initialize a list to store individual chunk token counts.
        chunk_token_counts = []

        # Iterate through every document chunk.
        for doc in documents:
            # Extract the text that will be sent to the embedding model.
            text = doc.page_content
            # Convert the text into tokens and count them.
            token_count = len(encode(text))
            # Add the chunk token count to the total.
            total_tokens += token_count
            # Store the individual chunk token count.
            chunk_token_counts.append(token_count)

        # Calculate the number of chunks.
        total_chunks = len(chunk_token_counts)

        # Calculate the average tokens per chunk.
        average_tokens = (
            total_tokens / total_chunks if total_chunks else 0
        )

        # Find the maximum tokens in a single chunk.
        max_tokens = max(chunk_token_counts, default=0)

        # Return the token statistics.
        return {
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "average_tokens_per_chunk": round(average_tokens, 2),
            "max_tokens_per_chunk": max_tokens
        }


