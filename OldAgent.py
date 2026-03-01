import os
import re
import asyncio
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import TypedDict, List, Dict
from urllib.parse import urlparse
from dotenv import load_dotenv

# LangChain & LangGraph
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

load_dotenv()

class AgentState(TypedDict):
    links: List[str]
    processed_articles: List[Dict[str, str]]
    summary: str

SUMMARIZER_PROMPT = """
You are a Senior AI Correspondent writing a polished 250-350 word "Daily Intelligence" brief.

Rules:
1. Start with a strong headline in markdown (single line).
2. Write a cohesive 3-4 paragraph narrative (not bullet points).
3. Every key claim must be immediately followed by a markdown source link in this exact format: [Source](URL).
4. Do not add a references section at the end.
5. Use **bold** for important companies, model names, and breakthroughs.
6. If a source is unclear, skip that claim instead of inventing details.

Input stories are delimited and include SOURCE_URL.

{content}
"""

def read_excel_node(state: AgentState):
    df = pd.read_excel("NewsLinks.xlsx")
    return {"links": df['URL'].tolist()}


async def scrape_links_node(state: AgentState):
    articles: List[Dict[str, str]] = []
    config = CrawlerRunConfig(remove_overlay_elements=True, word_count_threshold=40)

    async with AsyncWebCrawler() as crawler:
        for idx, url in enumerate(state["links"], start=1):
            try:
                result = await crawler.arun(url=url, config=config)
                if not result.success:
                    print(f"[{idx}] Skipped (crawl failed): {url}")
                    continue

                text = (getattr(result, "markdown", None) or "").strip()
                if not text:
                    text = (getattr(result, "cleaned_html", None) or "").strip()

                if not text:
                    print(f"[{idx}] Skipped (empty content): {url}")
                    continue

                articles.append(
                    {
                        "url": url,
                        "text": text[:3500],
                    }
                )
                print(f"[{idx}] Scraped: {url}")
            except Exception as exc:
                print(f"[{idx}] Error scraping {url}: {exc}")

    if not articles:
        raise RuntimeError("No articles were successfully scraped.")

    print(f"Scraping complete. {len(articles)} article(s) ready for summarization.")
    return {"processed_articles": articles}


def summarize_node(state: AgentState):
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    llm = ChatOllama(model=model_name, temperature=0.1)

    formatted_chunks = []
    for i, art in enumerate(state["processed_articles"], start=1):
        formatted_chunks.append(
            f"--- STORY {i} ---\n"
            f"SOURCE_URL: {art['url']}\n"
            f"CONTENT:\n{art['text']}\n"
        )

    formatted_data = "\n".join(formatted_chunks)
    prompt_template = ChatPromptTemplate.from_template(SUMMARIZER_PROMPT)
    chain = prompt_template | llm

    response = chain.invoke({"content": formatted_data})
    summary = (response.content or "").strip()

    if not summary:
        raise RuntimeError("Model returned an empty summary.")

    return {"summary": summary}

def _markdown_to_basic_html(markdown_text: str) -> str:
    html = markdown_text
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\[(.+?)\]\((https?://[^\s)]+)\)", r'<a href="\2">\1</a>', html)

    lines = [line.strip() for line in html.splitlines()]
    rendered = []
    for line in lines:
        if not line:
            continue
        if line.startswith("# "):
            rendered.append(f"<h2>{line[2:].strip()}</h2>")
        else:
            rendered.append(f"<p>{line}</p>")
    return "\n".join(rendered)

def send_email_node(state: AgentState):
    sender = os.getenv("EMAIL_ADDRESS")
    password = os.getenv("EMAIL_PASSWORD")
    recipient = os.getenv("RECIPIENT_EMAIL")

    if not sender or not password or not recipient:
        print("Email credentials missing. Printing summary instead:\n")
        print(state["summary"])
        return state

    message = MIMEMultipart("alternative")
    message["Subject"] = "Your AI Intelligence Briefing"
    message["From"] = sender
    message["To"] = recipient

    plain_body = state["summary"]
    html_body = _markdown_to_basic_html(state["summary"])

    message.attach(MIMEText(plain_body, "plain", "utf-8"))
    message.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, [recipient], message.as_string())
        print("Newsletter sent successfully.")
    except Exception as exc:
        print(f"Failed to send email: {exc}")
        print("\nSummary output:\n")
        print(state["summary"])

    return state


builder = StateGraph(AgentState)
builder.add_node("read_excel", read_excel_node)
builder.add_node("scrape", scrape_links_node)
builder.add_node("summarize", summarize_node)
builder.add_node("email", send_email_node)

builder.add_edge(START, "read_excel")
builder.add_edge("read_excel", "scrape")
builder.add_edge("scrape", "summarize")
builder.add_edge("summarize", "email")
builder.add_edge("email", END)

app = builder.compile()


if __name__ == "__main__":
    asyncio.run(app.ainvoke({"links": [], "processed_articles": [], "summary": ""}))
