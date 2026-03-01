from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import PGVector

from config import settings

CONNECTION_STRING = (
    f"postgresql://{settings.DB_USER}:"
    f"{settings.DB_PASSWORD}@"
    f"{settings.DB_HOST}:"
    f"{settings.DB_PORT}/"
    f"{settings.DB_NAME}"
)

embeddings = OllamaEmbeddings(model="llama3.2:3b")
llm = Ollama(
    model=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.1
)

vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name=settings.COLLECTION_NAME
)

def _split_sentences(text: str):
    cleaned = " ".join(str(text).split())
    parts = cleaned.replace("?", ".").replace("!", ".").split(".")
    return [p.strip() for p in parts if p.strip()]

def _build_three_lines(article_text: str):
    prompt = (
        "Create exactly 3 concise lines from the article text below.\n"
        "Rules:\n"
        "- Each line should capture a distinct key fact.\n"
        "- Keep each line under 25 words.\n"
        "- Return plain text with exactly 3 lines and no numbering.\n\n"
        f"Article text:\n{article_text[:5000]}"
    )
    response = str(llm.invoke(prompt)).strip()
    lines = [line.strip("- ").strip() for line in response.splitlines() if line.strip()]

    if len(lines) >= 3:
        return lines[:3]

    fallback = _split_sentences(article_text)
    while len(fallback) < 3:
        fallback.append("No additional key detail available in retrieved content.")
    return fallback[:3]

def run_rag_summary():
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K})

    query = "What are the most important and current developments across these news articles?"
    source_documents = retriever.invoke(query)

    if not source_documents:
        return "No relevant articles were retrieved."

    context_blocks = []
    for idx, doc in enumerate(source_documents, start=1):
        source_url = (doc.metadata or {}).get("source", "unknown")
        snippet = " ".join(str(doc.page_content).split())
        context_blocks.append(f"[Doc {idx}] Source: {source_url}\nContent: {snippet}")

    synthesis_prompt = (
        "You are a senior news intelligence analyst.\n"
        "Using ONLY the provided documents, produce a high-quality synthesis.\n"
        "Requirements:\n"
        "- Prioritize facts with concrete entities, numbers, places, and dates if present.\n"
        "- Remove duplication and merge overlapping points.\n"
        "- Distinguish confirmed developments from speculation.\n"
        "- Keep it concise and useful for executives.\n\n"
        "Output format:\n"
        "Key Developments:\n"
        "- 6 to 10 bullets\n"
        "Why It Matters:\n"
        "- 2 to 4 bullets\n"
        "Watchlist:\n"
        "- 2 to 4 bullets\n\n"
        "Documents:\n"
        f"{chr(10).join(context_blocks)}"
    )

    summary = str(llm.invoke(synthesis_prompt))

    links = []
    articles_by_source = {}
    for doc in source_documents:
        source_url = (doc.metadata or {}).get("source", "unknown")
        articles_by_source.setdefault(source_url, [])
        articles_by_source[source_url].append(" ".join(str(doc.page_content).split()))
        if source_url and source_url not in links:
            links.append(source_url)

    if not links:
        return summary

    article_highlights = []
    for link in links:
        article_text = " ".join(articles_by_source.get(link, []))
        highlight_lines = _build_three_lines(article_text) if article_text else [
            "No retrieved text available for this article.",
            "No retrieved text available for this article.",
            "No retrieved text available for this article.",
        ]
        block = (
            f"\nArticle: {link}\n"
            f"- {highlight_lines[0]}\n"
            f"- {highlight_lines[1]}\n"
            f"- {highlight_lines[2]}"
        )
        article_highlights.append(block)

    highlights_section = "\n\nArticle Highlights (3 lines each):" + "\n".join(article_highlights)
    sources_section = "\n\nSources:\n" + "\n".join([f"- {link}" for link in links])
    return summary + highlights_section + sources_section
