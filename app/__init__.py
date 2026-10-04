"""Apache — a local, voice-first AI assistant.

Layout
------
config    -- paths and tunables
llm       -- ChatOllama factory + server health check
rag       -- document ingestion, vector store, retriever
agent     -- LangChain 1.x ``create_agent`` wiring (tools + memory + streaming)
core      -- single source of truth shared by the text and voice input paths
voice     -- continuous listener, Google STT, edge-tts
tools     -- the tool registry handed to the agent
ui        -- Gradio Blocks front end
"""

__all__ = ["config", "llm", "rag", "agent", "core", "voice", "tools", "ui"]
