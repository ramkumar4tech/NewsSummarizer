from fastapi import FastAPI
import uvicorn
from ingest import ingest_from_excel
from agent_graph import build_graph
from crew_agents import run_multi_agent
from evaluation import compression_ratio
from email_sender import send_email

# app = FastAPI()
#
# @app.post("/run")
# def run_pipeline():
#     ingest_from_excel("data/NewsLinks.xlsx")
#     print("Pipeline started successfully")
#     graph = build_graph()
#     result = graph.invoke({})
#
#     final_summary = result["final_output"]
#
#     enhanced_summary = run_multi_agent(final_summary)
#
#     send_email(enhanced_summary)
#
#     return {
#         "message": "Pipeline executed successfully",
#         "summary": enhanced_summary
#     }
#
# if __name__ == "__main__":
#     uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=False)


if __name__ == "__main__":
    ingest_from_excel("data/NewsLinks.xlsx")
    print("Pipeline started successfully")
    graph = build_graph()
    result = graph.invoke({})

    final_summary = result["final_output"]

    enhanced_summary = run_multi_agent(final_summary)

    send_email(enhanced_summary)