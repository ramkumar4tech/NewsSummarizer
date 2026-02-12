import os
import asyncio
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from typing import TypedDict, List, Dict
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


# --- THE NEWSLETTER PROMPT ---
SUMMARIZER_PROMPT = """
You are a Senior AI Correspondent. Your task is to write a 250-word "Daily Intelligence" newsletter essay.

[EDITORIAL STRUCTURE]
1. HEADLINE: A punchy, bold title that captures the day's theme.
2. NARRATIVE ESSAY: Write a single, flowing narrative of 3-4 paragraphs. 
3. LINK RULE: Do NOT list links at the end. You MUST embed them naturally as [Source](URL) immediately after the fact they support.
4. HIGHLIGHTS: Use **bold text** for new model names, companies, and key technical breakthroughs.
5. NO FLUFF: Skip the "Based on the text" intros and the concluding questions. Start directly with the news.

[DATA TO PROCESS]
{content}

[NEWSLETTER OUTPUT]
"""


def read_excel_node(state: AgentState):
    df = pd.read_excel("NewsLinks.xlsx")
    return {"links": df['URL'].tolist()}


async def scrape_links_node(state: AgentState):
    articles = []
    # Configure crawler to ignore cookie popups and banners
    config = CrawlerRunConfig(remove_overlay_elements=True, word_count_threshold=30)

    async with AsyncWebCrawler() as crawler:
        for url in state["links"]:
            result = await crawler.arun(url=url, config=config)
            if result.success:
                articles.append({
                    "url": url,
                    "text": result.markdown[:3000]  # Trim for Llama 3.2 context window
                })
    return {"processed_articles": articles}


def summarize_node(state: AgentState):
    # Switching back to Llama 3.2:3b as requested
    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    # We format the content with clear URL markers for the LLM
    formatted_data = ""
    for i, art in enumerate(state["processed_articles"]):
        formatted_data += f"--- STORY {i + 1} ---\nSOURCE_URL: {art['url']}\nCONTENT: {art['text']}\n\n"

    prompt_template = ChatPromptTemplate.from_template(SUMMARIZER_PROMPT)
    chain = prompt_template | llm

    response = chain.invoke({"content": formatted_data})
    return {"summary": response.content}


def send_email_node(state: AgentState):
    # Using plain text for standard newsletter feel, or switch to "html" if needed
    msg = MIMEText(state["summary"], "plain")
    msg['Subject'] = "🤖 Your AI Intelligence Briefing"
    msg['From'] = os.getenv("EMAIL_ADDRESS")
    msg['To'] = os.getenv("RECIPIENT_EMAIL")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(os.getenv("EMAIL_ADDRESS"), os.getenv("EMAIL_PASSWORD"))
            server.sendmail(msg['From'], [msg['To']], msg.as_string())
        print("Newsletter sent successfully!")
    except Exception as e:
        print(f"Failed to send: {e}")
    return state


# --- GRAPH DEFINITION ---
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