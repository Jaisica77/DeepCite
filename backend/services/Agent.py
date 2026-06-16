import os
import json
from typing import Annotated, TypedDict, List, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    citations: List[dict]
    final_answer: Optional[str]


_retriever_instance = None
_doc_metadata: List[dict] = []


def set_retriever(retriever, doc_metadata: List[dict]):
    global _retriever_instance, _doc_metadata
    _retriever_instance = retriever
    _doc_metadata = doc_metadata


def _get_filename(doc_id: str) -> str:
    for d in _doc_metadata:
        if d.get("doc_id") == doc_id:
            return d.get("filename", doc_id)
    return doc_id


@tool
def search_docs(query: str) -> str:
    """Search the user's uploaded documents for relevant information.
    Always try this first before searching the web."""
    if _retriever_instance is None:
        return json.dumps({"error": "No documents indexed yet"})

    try:
        results = _retriever_instance.hybrid_search(query=query, top_k=4)
        formatted = []
        for r in results:
            doc_id = r["metadata"].get("doc_id", "")
            filename = _get_filename(doc_id)
            formatted.append({
                "source_type": "doc",
                "filename": filename,
                "doc_id": doc_id,
                "score": round(r["score"], 4),
                "text": r["chunk"]
            })
        return json.dumps(formatted)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def search_web(query: str) -> str:
    """Search the web for current or general information not found in documents."""
    if not TAVILY_API_KEY:
        return json.dumps({"error": "TAVILY_API_KEY not configured"})

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=4)
        formatted = []
        for r in response.get("results", []):
            formatted.append({
                "source_type": "web",
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "text": r.get("content", "")
            })
        return json.dumps(formatted)
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOLS = [search_docs, search_web]
TOOL_MAP = {t.name: t for t in TOOLS}


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    ).bind_tools(TOOLS)


SYSTEM_PROMPT = """You are DeepCite, an intelligent research assistant that always cites sources.

You have two tools:
- search_docs: searches the user's uploaded documents (always try this first)
- search_web: searches the internet for current or missing information

Rules:
1. Always call search_docs first for any factual question
2. If documents don't have enough info, also call search_web
3. You may call tools multiple times to gather complete information
4. Write a comprehensive final answer only when you have sufficient info

Citation format — after every sentence that uses retrieved information, add a tag:
- Document source: [DOC: filename.pdf]
- Web source: [WEB: https://example.com]

Example:
"Photosynthesis converts sunlight into chemical energy. [DOC: biology_notes.pdf] Global rates have declined 1% since 2000. [WEB: https://nature.com/article]"

Never state facts without a citation. Never make up information."""


def call_agent(state: AgentState) -> AgentState:
    llm = get_llm()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def call_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    citations = list(state.get("citations", []))
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_fn = TOOL_MAP.get(tool_name)

        if tool_fn:
            result = tool_fn.invoke(tool_args)
        else:
            result = json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                for item in parsed:
                    source = item.get("filename") or item.get("url", "")
                    if source and not any(c["source"] == source for c in citations):
                        citations.append({
                            "type": item.get("source_type", "doc"),
                            "source": source,
                            "title": item.get("title", ""),
                            "snippet": item.get("text", "")[:300]
                        })
        except Exception:
            pass

        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages, "citations": citations}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "end": END
    })
    graph.add_edge("tools", "agent")
    return graph.compile()


AGENT = build_agent()


def run_agent(
    query: str,
    retriever,
    doc_metadata: List[dict],
    chat_history: List[dict] = []
) -> dict:
    set_retriever(retriever, doc_metadata)

    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=query))

    result = AGENT.invoke({
        "messages": messages,
        "citations": [],
        "final_answer": None
    })

    final_message = result["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else ""
    citations = result.get("citations", [])
    tool_calls_made = sum(1 for m in result["messages"] if isinstance(m, ToolMessage))

    return {
        "answer": answer,
        "citations": citations,
        "tool_calls_made": tool_calls_made
    }
