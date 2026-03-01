
from langchain_community.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import settings

embeddings = OllamaEmbeddings(
    model=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_BASE_URL
)

def compression_ratio(summary, source):
    return len(summary) / len(source)

def embedding_similarity(summary, source):
    emb1 = embeddings.embed_query(summary)
    emb2 = embeddings.embed_query(source)

    similarity = cosine_similarity(
        np.array(emb1).reshape(1, -1),
        np.array(emb2).reshape(1, -1)
    )

    return similarity[0][0]
