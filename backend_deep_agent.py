import asyncio
import logging
import os
from typing import Optional, Union

from deepagents import create_deep_agent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tavily import TavilyClient

# ------------ Logging ------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------ Config ------------

# Use a Groq model by default. You can override this via the MODEL env var in Render.
MODEL = os.getenv("MODEL", "groq:llama-3.3-70b-versatile")

# Required environment variables (validated at startup):
# - TAVILY_API_KEY
# - GROQ_API_KEY
# - MODEL (optional)

# ------------ Startup validation ------------

def _require_env(name: str) -> str:
    """Raise a clear error at startup if a required env var is missing."""
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            f"Please add it in your Render dashboard under Environment."
        )
    return value


# ------------ Tavily client (lazy-initialized) ------------

_tavily_client: Optional[TavilyClient] = None


def get_tavily() -> TavilyClient:
    """Return the shared TavilyClient, initializing it on first call."""
    global _tavily_client
    if _tavily_client is None:
        api_key = _require_env("TAVILY_API_KEY")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


# ------------ Filesystem tools ------------

def glob(pattern: str) -> list[str]:
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


# ------------ Internet search tool (real Tavily) ------------

def internet_search(
    query: str,
    max_results: Union[int, str] = 5,
    type: str = "general",
) -> dict:
    """
    Search the internet using Tavily.

    Args:
        query:       The search query string.
        max_results: Maximum number of results to return (default 5).
        type:        Search type hint — passed through for compatibility.

    Returns:
        Tavily search result dict.
    """
    # Normalize max_results in case the model passes it as a string
    try:
        max_results_int = int(max_results)
    except (ValueError, TypeError):
        max_results_int = 5

    # Clamp to a sensible range accepted by Tavily (1–20)
    max_results_int = max(1, min(max_results_int, 20))

    logger.info("Tavily search | query=%r  max_results=%d  type=%s", query, max_results_int, type)

    try:
        result = get_tavily().search(query=query, max_results=max_results_int)
    except Exception as exc:
        logger.error("Tavily search failed: %s", exc)
        raise RuntimeError(f"Internet search failed: {exc}") from exc

    return result


# ------------ Agent factory ------------

research_instructions = (
    "You are a deep research agent. "
    "For each user query, break the task into clear steps, "
    "use the internet_search tool as needed, "
    "take notes in your internal files, and then produce a concise, "
    "well-structured answer. Prefer factual, cited responses."
)

tools = [glob, read_file, write_file, internet_search]


def _make_agent():
    """
    Create a fresh deep-agent instance.

    Called once at startup (after env-var validation) and again
    per-request if the agent is stateful.
    """
    # Validate GROQ_API_KEY early so failures surface clearly
    _require_env("GROQ_API_KEY")

    return create_deep_agent(
        model=MODEL,
        tools=tools,
        system_prompt=research_instructions,
    )


# ------------ FastAPI setup ------------

app = FastAPI(title="Deep Agent Service", version="0.2.0")


class DeepTaskRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class DeepTaskResponse(BaseModel):
    response: str


# ------------ Lifecycle events ------------

@app.on_event("startup")
async def startup_event():
    """Validate all required env vars at startup so Render shows a clear error."""
    _require_env("TAVILY_API_KEY")
    _require_env("GROQ_API_KEY")
    logger.info("Environment validated. Model: %s", MODEL)


# ------------ Helper: extract text from agent result ------------

def _extract_content(result: dict) -> str:
    """
    Safely pull the final text content out of the agent result dict.

    Handles both object-style messages (LangChain AIMessage) and plain dicts.
    """
    messages = result.get("messages")
    if not messages:
        raise ValueError("Agent result contains no messages.")

    final_msg = messages[-1]

    # Object-style (e.g. LangChain BaseMessage)
    if hasattr(final_msg, "content"):
        content = final_msg.content
    # Dict-style
    elif isinstance(final_msg, dict):
        content = final_msg.get("content", "")
    else:
        raise ValueError(f"Unrecognised message type: {type(final_msg)}")

    # Content can be a list of blocks (tool-use responses)
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()

    if not content:
        raise ValueError("Agent returned an empty response.")

    return content


# ------------ Routes ------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "deep-agent", "model": MODEL}


@app.post("/deep-task", response_model=DeepTaskResponse)
async def deep_task(payload: DeepTaskRequest):
    """
    Run a deep-agent task.

    Each request gets its own agent instance to avoid shared state
    across concurrent requests.
    """
    logger.info(
        "deep_task | user_id=%s  conversation_id=%s  query=%r",
        payload.user_id,
        payload.conversation_id,
        payload.query,
    )

    # Create a fresh agent per request (safe for concurrent use)
    try:
        agent = _make_agent()
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Run the (blocking) agent in a thread with a 120-second timeout
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                agent.invoke,
                {
                    "messages": [
                        {"role": "user", "content": payload.query}
                    ]
                },
            ),
            timeout=120,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Deep agent timed out after 120 seconds.")
    except Exception as exc:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        content = _extract_content(result)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return DeepTaskResponse(response=content)


@app.get("/test-tavily")
async def test_tavily():
    """
    Health-check that exercises the real Tavily API.
    Useful to verify the TAVILY_API_KEY is valid after deployment.
    """
    try:
        result = internet_search("LangGraph deep agents", max_results=1)
        return {"ok": True, "result": result}
    except Exception as exc:
        logger.error("Tavily test failed: %s", exc)
        return {"ok": False, "error": str(exc)}


# ------------ Entrypoint ------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend_deep_agent:app", host="0.0.0.0", port=port, reload=False)