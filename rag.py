"""
Local RAG demo: Ollama (llama3.2 + nomic-embed-text) + Chroma + LangChain.

How it works (the whole pipeline in 5 steps):
    1. LOAD      read the document (.txt or .pdf)
    2. CHUNK     split it into small overlapping pieces
    3. EMBED     turn each chunk into a vector (numbers that capture meaning)
    4. RETRIEVE  embed the question and find the closest chunks
    5. GENERATE  give those chunks + the question to the LLM to write the answer

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

# Document: LangChain's container for a piece of text plus metadata (source, page)
from langchain_core.documents import Document
# Splits long text into chunks, preferring paragraph/sentence boundaries
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Chroma: the local vector database that stores chunks and their embeddings
from langchain_chroma import Chroma
# OllamaEmbeddings = "The Reader" (text -> vectors); ChatOllama = "The Writer" (writes answers)
from langchain_ollama import OllamaEmbeddings, ChatOllama
# Prompt template and output parser used to build the generation chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prints the "Ollama active" status and the two model badges (see banner.py)
from banner import show_model_badges

# ---------------------------------------------------------------------------
# Settings you can change
# ---------------------------------------------------------------------------
DOCUMENT_PATH = "sample.txt"    # file to index (.txt or .pdf)
CHROMA_DIR = "./chroma_db"      # where the vector database is saved on disk
TOP_K = 3                       # how many chunks to retrieve per question

# The prompt sent to the LLM. "ONLY" plus the fallback sentence is what makes
# the model say "I don't know" for questions the document doesn't cover.
# {context} = retrieved chunks, {question} = the user's question.
PROMPT_TEMPLATE = """Answer the question based ONLY on the following context.
If you cannot find the answer in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Step 1: Load documents (no langchain-community needed)
# ---------------------------------------------------------------------------
def load_documents(file_path):
    print(f"Loading document: {file_path}")

    # PDFs: read page by page, keeping the page number as metadata
    if file_path.lower().endswith(".pdf"):
        from pypdf import PdfReader  # imported here so .txt users don't need it
        reader = PdfReader(file_path)
        return [
            Document(
                page_content=page.extract_text() or "",  # scanned pages return None
                metadata={"source": file_path, "page": i},
            )
            for i, page in enumerate(reader.pages)
        ]

    # Plain text: the whole file becomes one Document (UTF-8 avoids Windows errors)
    with open(file_path, encoding="utf-8") as f:
        return [Document(page_content=f.read(), metadata={"source": file_path})]


# ---------------------------------------------------------------------------
# Steps 2 + 3: Chunk, embed, and store in Chroma
# ---------------------------------------------------------------------------
def create_vector_store(docs):
    print("Chunking documents...")
    # chunk_size: max characters per chunk
    # chunk_overlap: characters repeated between neighbouring chunks, so a
    # sentence cut at a boundary still appears whole in one of them
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks")

    # The embedding model that turns text into vectors
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Demo only: show what an embedding looks like (first chunk, first 5 values)
    sample_vec = embeddings.embed_query(splits[0].page_content)
    print(f"Vector dimension: {len(sample_vec)}")
    print(f"First 5 values: {[round(v, 4) for v in sample_vec[:5]]}")

    # Delete the old database so reruns don't store duplicate chunks
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    print("Building Chroma vector store...")
    # Embeds every chunk with the model above and saves chunks + vectors to disk
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )


# ---------------------------------------------------------------------------
# Steps 4 + 5: Retrieve and generate
# ---------------------------------------------------------------------------
def format_docs(docs):
    # Join the retrieved chunks into one block of text for the prompt
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(llm):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # Pipeline: fill the prompt -> send to the LLM -> return plain text
    return prompt | llm | StrOutputParser()


def answer_with_rag(query, vectorstore, rag_chain):
    # Embed the question and fetch the TOP_K closest chunks.
    # We retrieve here (not inside the chain) so what we display on screen
    # is exactly what the LLM receives.
    results = vectorstore.similarity_search_with_score(query, k=TOP_K)

    # Chroma returns a DISTANCE: lower = closer match (not a percentage)
    print("\n--- Retrieved chunks (lower distance = closer match) ---")
    for i, (doc, score) in enumerate(results, 1):
        preview = doc.page_content.replace("\n", " ")[:160]
        print(f"[Chunk {i}] distance={score:.3f}\n  {preview}...")

    # Build the context from the chunks and let the LLM write the answer
    context = format_docs([doc for doc, _ in results])
    return rag_chain.invoke({"context": context, "question": query})


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    show_model_badges()  # on-screen status for the video

    # Fail early with a clear message if the document is missing
    if not os.path.exists(DOCUMENT_PATH):
        raise SystemExit(f"{DOCUMENT_PATH} not found. Put your document next to this script.")

    # Build the knowledge base: load -> chunk -> embed -> store
    docs = load_documents(DOCUMENT_PATH)
    vectorstore = create_vector_store(docs)

    # temperature=0 makes answers deterministic, so your demo repeats reliably
    llm = ChatOllama(model="llama3.2", temperature=0)
    rag_chain = build_rag_chain(llm)

    print("\n--- Local RAG Ready ---")
    print("Tip: prefix a question with '!' to skip retrieval and ask the plain LLM.")

    # Interactive question loop
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if not query:
            continue  # ignore empty input
        if query.lower() in ("exit", "quit"):
            break

        # "!" prefix = no retrieval, ask the LLM directly (shows hallucination)
        if query.startswith("!"):
            print("\n--- Plain LLM (no retrieval) ---")
            print(llm.invoke(query[1:].strip()).content)
            continue

        # Normal path: full RAG
        response = answer_with_rag(query, vectorstore, rag_chain)
        print(f"\n--- RAG Response ---\n{response}")