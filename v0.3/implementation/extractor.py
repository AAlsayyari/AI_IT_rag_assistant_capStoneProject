"""
Data Extraction Engine (v0.3).

Uses the project's LLM (Ollama / Qwen) to parse a raw chat transcript
into structured JSON containing:
  - ticket_class / sub_class
  - severity
  - sentiment
  - issue_resolved (boolean)
  - session_summary
  - user_id / college / role  (if mentioned in the conversation)
  - external_ticket_ref       (if an existing ticket number was provided)
"""

import json
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

MODEL = "qwen2.5:14b"

_extractor_llm = ChatOllama(
    model=MODEL,
    temperature=0,
    stop=["<|im_end|>", "<|im_start|>"],
)

EXTRACTION_PROMPT = """\
You are a metadata extraction engine for King Saud University IT support conversations.

Analyze the following chat transcript between a user and the AI support assistant, then return a JSON object with EXACTLY the following keys.  Do NOT add extra keys.

{{
  "ticket_class": "<primary category: one of Hardware, Software, Network, LMS, Email, Account, Security, Printing, Other>",
  "sub_class": "<specific sub-category, e.g. Password Reset, Wi-Fi Certificate, VPN Setup, Printer Driver, etc.>",
  "severity": "<one of: low, medium, high, critical>",
  "sentiment": "<one of: positive, neutral, frustrated, angry>",
  "issue_resolved": <true or false>,
  "session_summary": "<concise 2-3 sentence recap of the problem and the AI's troubleshooting steps>",
  "user_id": "<user ID or student/employee number if mentioned, otherwise null>",
  "college": "<college or department if mentioned, otherwise null>",
  "role": "<student, faculty, staff, or null if not mentioned>",
  "external_ticket_ref": "<any existing ticket reference number the user provided, otherwise null>"
}}

Rules:
- Output ONLY the JSON object — no markdown fences, no explanation.
- If a field cannot be determined, set it to null (for strings) or false (for issue_resolved).
- Keep the session_summary in the SAME language the user used (Arabic or English).

--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---
"""


def build_transcript(history: list[dict]) -> str:
    """Convert Gradio message dicts into a plain-text transcript."""
    lines: list[str] = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def extract_metadata(history: list[dict]) -> dict:
    """
    Run the LLM extraction pipeline on the chat history.

    Returns a dict with the keys defined in EXTRACTION_PROMPT.
    On parse failure, returns a dict with all values set to None/False.
    """
    transcript = build_transcript(history)
    prompt = EXTRACTION_PROMPT.format(transcript=transcript)
    response = _extractor_llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Strip markdown code fences if the model wraps the output
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback – return empty metadata rather than crashing
        data = {
            "ticket_class": None,
            "sub_class": None,
            "severity": None,
            "sentiment": None,
            "issue_resolved": False,
            "session_summary": None,
            "user_id": None,
            "college": None,
            "role": None,
            "external_ticket_ref": None,
        }

    return data
