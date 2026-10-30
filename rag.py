"""
Local RAG demo: Ollama (llama3.2 + nomic-embed-text) + Chroma + LangChain.

Setup:
    ollama pull nomic-embed-text
    ollama pull llama3.2
    pip install langchain-text-splitters langchain-chroma \
                langchain-ollama langchain-core pypdf

Usage in the demo loop:
    <question>        -> RAG answer (shows retrieved chunks first)
    !<question>       -> plain LLM answer, no retrieval (to show hallucination)
    exit / quit       -> leave
"""

import os
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DOCUMENT_PATH = "sample.txt"
CHROMA_DIR = "./chroma_db"
TOP_K = 3

PROMPT_TEMPLATE = """Answer the question based ONLY on the following context.
If you cannot find the answer in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""


# 1. Load documents (no langchain-community needed)
def load_documents(file_path):
    print(f"Loading document: {file_path}")
    if file_path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": file_path, "page": i},
            )
            for i, page in enumerate(reader.pages)
        ]
    with open(file_path, encoding="utf-8") as f:
        return [Document(page_content=f.read(), metadata={"source": file_path})]


# 2. Chunk, embed, store
def create_vector_store(docs):
    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Show what an embedding looks like (first chunk, first 5 values only)
    sample_vec = embeddings.embed_query(splits[0].page_content)
    print(f"Vector dimension: {len(sample_vec)}")
    print(f"First 5 values: {[round(v, 4) for v in sample_vec[:5]]}")

    # Start fresh so reruns don't store duplicate chunks
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    print("Building Chroma vector store...")
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )


# 3. RAG pieces
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(llm):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt | llm | StrOutputParser()


def answer_with_rag(query, vectorstore, rag_chain):
    # Retrieve once, so what we display is exactly what the LLM sees
    results = vectorstore.similarity_search_with_score(query, k=TOP_K)

    print("\n--- Retrieved chunks (lower distance = closer match) ---")
    for i, (doc, score) in enumerate(results, 1):
        preview = doc.page_content.replace("\n", " ")[:160]
        print(f"[Chunk {i}] distance={score:.3f}\n  {preview}...")

    context = format_docs([doc for doc, _ in results])
    return rag_chain.invoke({"context": context, "question": query})


# --- Main ---
if __name__ == "__main__":
    if not os.path.exists(DOCUMENT_PATH):
        raise SystemExit(f"{DOCUMENT_PATH} not found. Put your document next to this script.")

    docs = load_documents(DOCUMENT_PATH)
    vectorstore = create_vector_store(docs)

    llm = ChatOllama(model="llama3.2", temperature=0)
    rag_chain = build_rag_chain(llm)

    print("\n--- Local RAG Ready ---")
    print("Tip: prefix a question with '!' to skip retrieval and ask the plain LLM.")

    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        if query.startswith("!"):
            print("\n--- Plain LLM (no retrieval) ---")
            print(llm.invoke(query[1:].strip()).content)
            continue

        response = answer_with_rag(query, vectorstore, rag_chain)
        print(f"\n--- RAG Response ---\n{response}")