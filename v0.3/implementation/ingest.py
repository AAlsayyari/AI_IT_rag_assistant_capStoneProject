# import os
# import glob
# from pathlib import Path
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# import os
# from dotenv import load_dotenv
# load_dotenv(override=True)
# # MUST be placed before importing HuggingFace / LangChain modules
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings



# # BAAI/bge-m3
# # Qwen/Qwen3-Embedding-8B

# import torch

# Embedding_model = "BAAI/bge-m3"  
# DB_NAME = str(Path(__file__).parent.parent / "vector_db_it")
# KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "KB-IT-Corrected")

# # Auto-detect best available device
# if torch.cuda.is_available():
#     DEVICE = "cuda"
# elif torch.backends.mps.is_available():
#     DEVICE = "mps"
# else:
#     DEVICE = "cpu"

# embeddings = HuggingFaceEmbeddings(model_name=Embedding_model,
#                                     model_kwargs={"device": DEVICE,
#                                                    "local_files_only": True},encode_kwargs={"normalize_embeddings": True})


# # embeddings = HuggingFaceEmbeddings(
# #     model_name=Embedding_model,
# #     model_kwargs={
# #         "device": "cuda",
# #         "trust_remote_code": True,  # Mandatory for Qwen3 models
# #         "local_files_only": True,
# #     },
# #     encode_kwargs={"normalize_embeddings": True},
# # )


# # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# def fetch_documents():
#     loader = DirectoryLoader(
#         KNOWLEDGE_BASE,
#         glob="*.md",
#         loader_cls=TextLoader,
#         loader_kwargs={"encoding": "utf-8"}
#     )
#     documents = loader.load()
    
#     for doc in documents:
#         doc.metadata["doc_type"] = "IT"  # Set your desired category or derive it from the path
        
#     return documents


# def create_chunks(documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     chunks = text_splitter.split_documents(documents)
#     return chunks


# def create_embeddings(chunks):
#     if os.path.exists(DB_NAME):
#         Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

#     vectorstore = Chroma.from_documents(
#         documents=chunks, embedding=embeddings, persist_directory=DB_NAME
#     )

#     collection = vectorstore._collection
#     count = collection.count()

#     sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
#     dimensions = len(sample_embedding)
#     print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
#     return vectorstore


# if __name__ == "__main__":
#     documents = fetch_documents()
#     chunks = create_chunks(documents)
#     create_embeddings(chunks)
#     print("Ingestion complete")










# using qdrant

import os
import glob
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import torch

load_dotenv(override=True)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

Embedding_model = "BAAI/bge-m3"  
DB_PATH = str(Path(__file__).parent.parent / "qdrant_db_it")
COLLECTION_NAME = "ksu_it_docs"
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "KB-IT-Corrected")

# Auto-detect best available device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name=Embedding_model,
    model_kwargs={
        "device": DEVICE,
        "local_files_only": True
    },
    encode_kwargs={"normalize_embeddings": True}
)


def fetch_documents():
    """تحميل جميع ملفات .md مباشرة بدون الاعتماد على langchain_community القديم"""
    documents = []
    md_files = glob.glob(os.path.join(KNOWLEDGE_BASE, "*.md"))
    
    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            doc = Document(
                page_content=content,
                metadata={"source": file_path, "doc_type": "IT"}
            )
            documents.append(doc)
            
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    # 1. تنظيف وإعادة إنشاء الـ Collection
    init_client = QdrantClient(path=DB_PATH)
    if init_client.collection_exists(collection_name=COLLECTION_NAME):
        init_client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection: '{COLLECTION_NAME}'")
    
    init_client.close()
    del init_client

    # 2. إنشاء الـ Vector Store وبناء الـ Embeddings
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=DB_PATH,
        collection_name=COLLECTION_NAME
    )

    # 3. استخدام العميل الداخلي لـ vectorstore بدون فتح جلسة ثانية تسبب قفل المجلد
    count = vectorstore.client.get_collection(collection_name=COLLECTION_NAME).points_count
    print(f"There are {count:,} vectors stored in Qdrant under collection '{COLLECTION_NAME}'")
    
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")