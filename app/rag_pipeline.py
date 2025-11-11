"""
Module: rag_pipeline.py
-----------------------
Implements a Retrieval-Augmented Generation (RAG) pipeline using Mistral via Ollama
and a ChromaDB vector store.

Objective:
Enable the LLM to ground its responses in internal product documentation
by retrieving relevant text chunks before generation.
"""

from typing import Any, Dict, List

import logging
import os
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app import constants


logger = logging.getLogger(__name__)


def load_documents() -> List[str]:
    """
    Convert each product entry into a textual document suitable for embedding.

    Each product is represented as a descriptive text block containing
    structured information (category, price, rating, delivery time, etc.).
    These are later embedded for semantic retrieval.

    Returns:
        List[str]: List of formatted product descriptions, where each string
        represents a product document ready for embedding (one per product).
    """
    df = pd.read_csv(constants.PRODUCTS_PATH)
    docs = []

    for _, row in df.iterrows():
        text = (
            f"Product ID: {row['product_id']}\n"
            f"Name: {row['name']}\n"
            f"Category: {row['category']}\n"
            f"Price: {row['price']} euros\n"
            f"Average rating: {row['avg_rating']}\n"
            f"Return rate: {row['return_rate']}\n"
            f"Delivery estimate: {row['delivery_estimate_days']} days\n"
            f"Description: {row['description']}"
        )
        docs.append(text)
    return docs


def build_vectorstore(force_rebuild: bool = False) -> Any:
    """
    Build or load a Chroma vector store containing product embeddings.

    Process:
    1. Each product is transformed into a text document (via `load_documents`).
    2. Documents are split into smaller text units ("chunks") using
       `RecursiveCharacterTextSplitter`. This improves retrieval granularity:
       - avoids exceeding context limits,
       - allows retrieval of only the relevant fragment of a long document.
    3. Each chunk is embedded into a high-dimensional vector using a
       pre-trained sentence-transformer model.
    4. All embeddings are stored in ChromaDB for semantic search.

    Args:
        force_rebuild (bool): If True, rebuilds the index from scratch even if it exists.

    Returns:
        Chroma: Persisted Chroma vector store instance.
    """
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores.chroma import Chroma

    if os.path.exists(constants.CHROMA_DIR) and not force_rebuild:
        logger.info("Using existing Chroma index.")
        return Chroma(
            persist_directory=constants.CHROMA_DIR,
            embedding_function=HuggingFaceEmbeddings(
                model_name=constants.EMBEDDING_MODEL
            ),
        )

    logger.info("Building new Chroma index...")
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDING_MODEL)
    os.makedirs(constants.CHROMA_DIR, exist_ok=True)
    vectorstore = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=constants.CHROMA_DIR
    )
    vectorstore.persist()
    logger.info("Chroma index built and saved.")
    return vectorstore


def get_rag_chain() -> RetrievalQA:
    """
    Create a RetrievalQA chain combining:
      - Chroma retriever for semantic search
      - Mistral model for answer generation

    The retriever fetches the k most relevant chunks, and the model synthesizes
    a grounded response using that context.

    Returns:
        RetrievalQA: Configured LangChain RAG chain.
    """
    from langchain_community.llms import Ollama

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(
        model=constants.LLM_MODEL_RAG, temperature=constants.TEMPERATURE_RAG
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    return qa_chain


def query(question: str) -> Dict[str, Any]:
    """
    Execute a user query through the RAG pipeline.

    Steps:
    1. Retrieve top-k relevant chunks from ChromaDB.
    2. Inject them into Mistral's context window.
    3. Generate a concise, grounded answer.

    Args:
        question (str): User question in natural language.

    Returns:
        dict: Full LangChain response (answer + retrieved source documents).
    """
    chain = get_rag_chain()
    response = chain({"query": question})
    # Convert LangChain Document objects to dicts to ensure JSON-serializable response
    if "source_documents" in response:
        response["source_documents"] = [
            doc.dict() for doc in response["source_documents"]
        ]
    logger.info("\nQuestion:", question)
    logger.info("Answer:", response["result"])
    return response


if __name__ == "__main__":
    query("Which products have a high return rate and low rating?")
