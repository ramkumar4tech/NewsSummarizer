from langchain_community.llms import Ollama
from config import settings

llm = Ollama(
    model=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.1
)

def run_multi_agent(summary_text):
    highlights_marker = "\n\nArticle Highlights (3 lines each):"
    sources_marker = "\n\nSources:\n"

    summary_part = summary_text
    preserved_tail = ""

    if highlights_marker in summary_text:
        split_idx = summary_text.index(highlights_marker)
        summary_part = summary_text[:split_idx].strip()
        preserved_tail = summary_text[split_idx:]
    elif sources_marker in summary_text:
        split_idx = summary_text.index(sources_marker)
        summary_part = summary_text[:split_idx].strip()
        preserved_tail = summary_text[split_idx:]

    print("[Agent: News Analyst] Starting analysis...")
    analysis_prompt = (
        "You are a News Analyst.\n"
        "Analyze the input and extract only high-signal insights.\n"
        "Rules:\n"
        "- Keep facts specific and non-redundant.\n"
        "- Preserve concrete entities, numbers, and timelines when present.\n"
        "- Separate facts from assumptions.\n\n"
        "Output:\n"
        "Themes:\n"
        "- 4 to 6 bullets\n"
        "Notable Facts:\n"
        "- 4 to 8 bullets\n"
        "Risks/Uncertainty:\n"
        "- 2 to 4 bullets\n\n"
        f"Input:\n{summary_part}"
    )
    analysis_output = str(llm.invoke(analysis_prompt))
    print("[Agent: News Analyst] Completed.")

    print("[Agent: Chief Editor] Starting executive summary...")
    editor_prompt = (
        "You are a Chief Editor preparing a board-level briefing.\n"
        "Transform the analyst notes into a final summary.\n"
        "Rules:\n"
        "- Output exactly 7 bullet points.\n"
        "- Each bullet should be one sentence.\n"
        "- Start each bullet with a strong keyword label in uppercase followed by ':'.\n"
        "- Prioritize impact, urgency, and decisions.\n"
        "- Do not repeat points.\n\n"
        f"Analyst output:\n{analysis_output}"
    )
    final_output = str(llm.invoke(editor_prompt))
    print("[Agent: Chief Editor] Completed.")

    if preserved_tail:
        return final_output.strip() + preserved_tail
    return final_output
