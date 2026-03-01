import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from config import settings

CONNECTION_STRING = (
    f"postgresql://{settings.DB_USER}:"
    f"{settings.DB_PASSWORD}@"
    f"{settings.DB_HOST}:"
    f"{settings.DB_PORT}/"
    f"{settings.DB_NAME}"
)

embeddings = OllamaEmbeddings(model="llama3.2:3b")

def fetch_article(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.text for p in paragraphs])
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def ingest_from_excel(file_path: str):
    df = pd.read_excel(file_path)
    urls = df["URL"].dropna().tolist()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    print("vector store creating")
    vectorstore = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name=settings.COLLECTION_NAME
    )

    for url in urls:
        content = fetch_article(url)
        if content:
            chunks = splitter.split_text(content)
            chunk_metadata = [{"source": url} for _ in chunks]
            vectorstore.add_texts(chunks, metadatas=chunk_metadata)

    print("Ingestion completed.")
