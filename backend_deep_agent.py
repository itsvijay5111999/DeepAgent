# backend_deep_agent.py

import os
from typing import Optional

from deepagents import create_deep_agent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tavily import TavilyClient

# ------------ Config ------------

# Use a Groq model by default. You can override this via the MODEL env var in Render.
MODEL = os.getenv("MODEL", "groq:llama-3.3-70b-versatile")

# Environment variables you must set in Render B:
# - MODEL (optional, overrides the default above)
# - TAVILY_API_KEY
# - GROQ_API_KEY
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# ------------ Tools ------------

def internet_search(
    query: str,
    max_results: int = 5,
) -> dict:
    """Temporary stub to avoid Tavily 400 errors while testing the deep agent."""
    return {
        "query": query,
        "max_results": max_results,
        "results": [
            {
                "url": "https://example.com/deep-agents",
                "title": "Stub result about deep agents",
                "content": f"This is a stubbed search result for query: {query}.",
                "score": 1.0,
            }
        ],
        "response_time": 0.01,
        "request_id": "local-stub",
    }

# ------------ Deep agent init (singleton) ------------

research_instructions = (
    "You are a deep research agent. "
    "For each user query, break the task into clear steps, "
    "use the internet_search tool as needed, "
    "take notes in your internal files, and then produce a concise, "
    "well-structured answer. Prefer factual, cited responses."
)

# This creates a single deep agent instance reused for all requests.
# Deep Agents will include its own default tools (filesystem, todos, etc.);
# we add our custom internet_search via extra_tools.
agent = create_deep_agent(
    model=MODEL,
    extra_tools=[internet_search],
    system_prompt=research_instructions,
)

# ------------ FastAPI setup ------------

app = FastAPI(title="Deep Agent Service", version="0.1.0")


class DeepTaskRequest(BaseModel):
    query: str
    # Optional: pass along some context from the calling chatbot
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class DeepTaskResponse(BaseModel):
    response: str


@app.get("/health")
async def health():
    return {"status": "ok", "service": "deep-agent"}


@app.post("/deep-task", response_model=DeepTaskResponse)
async def deep_task(payload: DeepTaskRequest):
    """
    Run a deep-agent task.

    The caller (your existing chatbot backend) sends a query.
    We wrap it into the Deep Agents message format, invoke the agent,
    and return only the final answer text.
    """
    try:
        # Deep Agents expects a messages list; we only send a single user turn here.
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": payload.query}
                ]
            }
        )
    except Exception as e:
        # Return a clean 500 error if the agent fails
        raise HTTPException(status_code=500, detail=str(e))

    # Deep Agents returns a dict containing "messages"; last message is final answer.
    try:
        messages = result["messages"]
        final_msg = messages[-1]
        # final_msg.content may be a string or list of content parts depending on model
        content = getattr(final_msg, "content", None) or final_msg.get("content")
        if isinstance(content, list):
            # Simple join for multi-part contents
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Deep agent returned an unexpected format",
        )

    return DeepTaskResponse(response=content)


@app.get("/test-tavily")
async def test_tavily():
    try:
        result = internet_search("LangGraph deep agents", max_results=1)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend_deep_agent:app", host="0.0.0.0", port=port)