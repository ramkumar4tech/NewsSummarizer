from langgraph.graph import StateGraph, END
from typing import TypedDict
from rag_pipeline import run_rag_summary
from guardrails import apply_guardrails

class AgentState(TypedDict):
    summary: str
    final_output: str

def summarize_node(state: AgentState):
    summary = run_rag_summary()
    return {"summary": summary}

def guardrail_node(state: AgentState):
    safe_output = apply_guardrails(state["summary"])
    return {"final_output": safe_output}

def build_graph():
    print("inside build graph")
    workflow = StateGraph(AgentState)

    workflow.add_node("summarize", summarize_node)
    workflow.add_node("guardrail", guardrail_node)

    workflow.set_entry_point("summarize")
    workflow.add_edge("summarize", "guardrail")
    workflow.add_edge("guardrail", END)

    return workflow.compile()
