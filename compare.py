"""Print real cosine similarity between a question and a few policy clauses."""

import math
from langchain_ollama import OllamaEmbeddings

QUERY = "Can I work from home?"
CLAUSES = [
    "Employees may work remotely up to 3 days per week.",
    "Flights over 6 hours may be booked in premium economy.",
]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


emb = OllamaEmbeddings(model="nomic-embed-text")
q_vec = emb.embed_query(QUERY)

print(f"Query: {QUERY}\n")
for clause in CLAUSES:
    score = cosine(q_vec, emb.embed_query(clause))
    print(f"{score:.2f}  {clause}")
