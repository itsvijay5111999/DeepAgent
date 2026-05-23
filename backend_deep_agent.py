# backend_deep_agent.py

import os
from typing import Optional, List

from deepagents import create_deep_agent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tavily import TavilyClient

from langchain_core.tools import tool  # explicit tool wrapper

# ------------ Config ------------

MODEL = os.getenv("MODEL", "groq:llama-3.3-70b-versatile")

# Environment variables you must set in Render B:
# - MODEL (optional)
# - TAVILY_API_KEY
# - GROQ_API_KEY
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# ------------ Filesystem tools ------------

def glob(pattern: str) -> List[str]:
    """Return a list of file paths matching the glob pattern under the current directory."""
    import glob as pyglob
    return pyglob.glob(pattern, recursive=True)

def read_file(path: str) -> str:
    """Read text from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path: str, content: str) -> str:
    """Write text to a file, returning the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# ------------ Internet search tool (explicit schema) ------------

class InternetSearchInput(BaseModel):
    type: str = Field(
        default="general",
        description="Search type. Use 'general' for normal web search."
    )
    query: str = Field(
        description="Search query string."
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of results to return (1-10)."
    )

@tool("internet_search", args_schema=InternetSearchInput)
def internet_search_tool(params: InternetSearchInput) -> dict:
    """
    Temporary stub internet search tool.
    Accepts (type, query, max_results) to match how the model calls it.
    """
    max_results_int = params.max_results

    # Stubbed response to avoid Tavily 400s while wiring everything.
    return {
        "type": params.type,
        "query": params.query,
        "max_results": max_results_int,
        "results": [
            {
                "url": "https://example.com/deep-agents",
                "title": "Stub result about deep agents",
                "content": f"This is a stubbed search result for query: {params.query}.",
                "score": 1.0,
            }
        ],
        "response_time": 0.01,
        "request_id": "local-stub",
    }

# If you later want real Tavily, replace the body above with:
#
#     return tavily.search(
#         query=params.query,
#         max_results=params.max_results,
#     )

# ------------ Deep agent init (singleton) ------------

research_instructions = (
    "You are a deep research agent. "
    "For each user query, break the task into clear steps, "
    "use the internet_search tool as needed, "
    "take notes in your internal files, and then produce a concise, "
    "well-structured answer. Prefer factual, cited responses."
)

tools = [glob, read_file, write_file, internet_search_tool]

agent = create_deep_agent(
    model=MODEL,
    tools=tools,
    system_prompt=research_instructions,
)

# ------------ FastAPI setup ------------

app = FastAPI(title="Deep Agent Service", version="0.1.0")


class DeepTaskRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class DeepTaskResponse(BaseModel):
    response: str


@app.get("/health")
async def health():
    return {"status": "ok", "service": "deep-agent"}


@app.post("/deep-task", response_model=DeepTaskResponse)
async def deep_task(payload: DeepTaskRequest):
    try:
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": payload.query}
                ]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        messages = result["messages"]
        final_msg = messages[-1]
        content = getattr(final_msg, "content", None) or final_msg.get("content")
        if isinstance(content, list):
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


@app.get("/test-search")
async def test_search():
    """Test the internet_search tool directly."""
    try:
        result = internet_search_tool.invoke(
            {"type": "general", "query": "LangGraph deep agents", "max_results": 3}
        )
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend_deep_agent:app", host="0.0.0.0", port=port)