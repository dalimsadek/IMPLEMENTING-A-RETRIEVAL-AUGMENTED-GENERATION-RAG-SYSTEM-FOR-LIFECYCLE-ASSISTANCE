from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import json
from typing import List

class RetrieverBuilder:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 4,
        verbose: bool = True
    ):
        """
        Loads an existing Chroma vectorstore and returns a retriever.

        Args:
            persist_directory: Path where the vector DB is stored.
            model_name: Embedding model used (must match what was used to index).
            top_k: Number of top documents to retrieve.
            verbose: Print status messages.
        """
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.top_k = top_k
        self.verbose = verbose

        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

        if self.verbose:
            print(f"[RetrieverBuilder] Loading Chroma from: {self.persist_directory}")

        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        if self.verbose:
            print(f"[RetrieverBuilder] Chroma loaded. Ready to retrieve top-{self.top_k} documents using cosine similarity.")

    def get_retriever(self):
        """Return retriever with top_k configuration."""
        return self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

    def ingest_json_chunks(self, json_paths: List[str], force: bool = False):
        if not json_paths:
            print("[RetrieverBuilder] ❌ No JSON paths provided.")
            return

        if not force and os.path.exists(os.path.join(self.persist_directory, "chroma-collections.parquet")):
            print(f"[RetrieverBuilder] ✅ Skipping ingestion — Chroma DB already exists at {self.persist_directory}.")
            return

        # Avoid re-ingesting if DB already exists
        if os.path.exists(os.path.join(self.persist_directory, "chroma-collections.parquet")):
            print(f"[RetrieverBuilder] 🛑 Skipping ingestion: Chroma DB already exists at {self.persist_directory}.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        all_docs = []

        for path in json_paths:
            if not os.path.exists(path):
                print(f"[WARN] Skipping missing file: {path}")
                continue

            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            for chunk in chunks:
                if "text" not in chunk:
                    continue

                sub_chunks = text_splitter.split_text(chunk["text"])
                for i, part in enumerate(sub_chunks):
                    all_docs.append(Document(
                        page_content=part,
                        metadata={
                            "source": chunk.get("source", path),
                            "chunk_id": f"{chunk.get('chunk_id', -1)}-{i}"
                        }
                    ))

        if not all_docs:
            print("[RetrieverBuilder] ❌ No valid chunks found to ingest.")
            return

        print(f"[RetrieverBuilder] Ingesting {len(all_docs)} semantic sub-chunks...")

        Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        ).persist()

        print("[RetrieverBuilder] ✅ JSON chunk ingestion complete.")
