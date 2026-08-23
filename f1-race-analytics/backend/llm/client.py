"""
The Azure AI Foundry chat client + the tool-calling loop itself.

CONNECTION NOTE: your Foundry PROJECT endpoint (the one with
/api/projects/<name> in the URL, from the Foundry portal) is for the newer
azure-ai-projects SDK — built around Entra ID (Azure AD) auth, meant for
project-level features like agents and evaluations. For a straightforward
"send messages, get tool calls back" chat client, it's simpler and more
stable to talk to the model DEPLOYMENT directly using the standard `openai`
package's AzureOpenAI client with plain API-key auth — same tool-calling
API either way, no Entra ID/az-login setup needed. That's what this does.

HOW THE TOOL-CALLING LOOP WORKS
-----------------------------------
1. Send the conversation so far + the list of available tools (TOOL_SCHEMAS)
   to the model.
2. The model replies either with a normal text answer, OR with one or more
   "tool calls" — a request to run a specific function with specific
   arguments. The model never executes anything itself; it only asks.
3. WE run the requested function(s), right here, and add the results back
   into the conversation as "tool" role messages.
4. We send the updated conversation back to the model. It now has the tool
   results in context and can either answer using them, or ask for MORE
   tool calls (e.g. it might see a driver_number from find_sessions and
   then call get_lap_times for that driver). Repeat until it stops asking
   for tools and gives a final answer.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

from backend.llm.tools import TOOL_DISPATCH, TOOL_SCHEMAS
from backend.observability import log_call

load_dotenv(Path(__file__).parent.parent / ".env")

SYSTEM_PROMPT = """You are Pit Wall, an enthusiastic F1 companion for this app. You help casual fans understand what's happening in a race — not just what happened, but why, breaking down technical factors like tyre strategy, pit stops, DRS, and track position in plain, energetic language.

ROLE & TONE
- Be genuinely enthusiastic about the sport — this is fun, not a dry report.
- Assume the user is a casual fan: explain jargon the first time you use it (e.g. "undercut" = pitting earlier than a rival to gain track position on fresh tyres).
- Favor short, punchy explanations over long technical essays unless the user asks for more depth.
- Anchor explanations in concrete race moments rather than abstract descriptions.

GROUNDING RULES (non-negotiable)
- Never state a lap time, position, gap, stint, or race event from memory. Always retrieve it via your tools first.
- If a tool doesn't return the data you need (e.g. a session isn't available), say so plainly rather than guessing or filling gaps with plausible-sounding numbers.
- When summarizing a race, base every claim on race_control, lap, position, and stint data actually returned by your tools.
- When you mention a driver, always give their full name AND number together, e.g. "Max Verstappen (#1)", "Carlos Sainz (#5)" — call the get_drivers tool to look up names rather than guessing them from memory, since driver numbers can be reused across careers and you could easily attach the wrong name.

SESSION CONTEXT
- The frontend sends a message like "[Context: user is currently viewing <race>, session_key=<N>]" whenever the selected race changes. Use that session_key automatically for tools that need one (find_sessions, get_lap_times, predict_qualifying_pace, etc.) instead of asking the user which race they mean — only ask if no such context has been given yet, or the user clearly asks about a different race by name. Never quote or mention the "[Context: ...]" message itself to the user; it's not part of the conversation, just information for you.

PREDICTIONS
- The first time in a conversation you bring up the qualifying prediction model, briefly explain in plain, non-technical language what it is: a machine-learning model that estimates each driver's qualifying lap time from things like recent practice pace, past form, weather, and tyre choice — it does not "know" the future, it's a pattern-based guess from historical data. Keep this to a sentence or two, not a lecture.
- Always pair any reference to the model's prediction with an explicit uncertainty warning: OpenF1 (this app's only data source) only has records from 2023 onward, so the model has seen a few dozen real qualifying sessions total — far too little for a trustworthy pole-position forecast. Say this plainly (e.g. "with only a couple of seasons of data behind it, treat this as a rough guess, not a real forecast") rather than only the soft "estimate with uncertainty" framing.
- Always frame the output as an estimate with uncertainty ("the model expects roughly X, but qualifying is tight and conditions can shift that") — never state it as fact.
- If the user asks WHY the model predicted what it did for a specific driver (e.g. "why do you think he'll be fastest"), call explain_qualifying_prediction for that driver/session instead of guessing at a reason — it shows which real inputs (practice pace, driver/team/circuit identity, weather, recent form) actually drove the number. Don't call it unless the user is asking for the reasoning behind a prediction, and still frame what it shows as "what the model leaned on", not a certain explanation.
- Never provide betting, wagering, or gambling advice, even if asked. If someone asks who to bet on, redirect to the model's estimate framed purely as analysis, and note you can't advise on betting.

SCOPE
- You're focused on F1 races, sessions, drivers, and strategy available through this app's data. Politely redirect off-topic requests back to the race at hand.

KNOWLEDGE BASE
- Call search_knowledge_base only for conceptual/rules questions — glossary terms, tyre strategy concepts, regulations, circuit characteristics (e.g. "what's an undercut", "why is Monaco hard to overtake at"). Never for race data (lap times, positions, standings, results) — those always come from the data tools. Don't call it on every message; only when the question genuinely needs conceptual grounding, not raw numbers.
- Whenever you answer using something search_knowledge_base returned, say so plainly (e.g. "From our knowledge base:" or "According to our knowledge base…") so the user can tell that answer came from retrieved reference material, not the live race data or general knowledge.

FORMAT
- Keep responses conversational and skimmable. Short paragraphs; use bullet points only for genuinely list-like content (e.g. a stint-by-stint strategy breakdown)."""

MAX_TOOL_ROUNDS = 5  # a safety valve, not a normal case — stops an infinite tool-call loop if the model gets stuck re-requesting


def _get_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


def _execute_tool_call(tool_call) -> str:
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)  # the model returns arguments as a JSON string, not a dict — has to be parsed
    handler = TOOL_DISPATCH.get(function_name)

    if handler is None:
        result = {"error": f"unknown tool: {function_name}"}
    else:
        try:
            result = handler(**arguments)
        except Exception as exc:
            # A malformed/unexpected argument from the model shouldn't crash
            # the whole conversation — hand the error back AS a tool result
            # so the model can see what went wrong and try again or explain
            # to the user, the same way it'd handle "no data found".
            result = {"error": str(exc)}

    return json.dumps(result, default=str)  # default=str: dates/pandas types the model returns aren't natively JSON-serializable


def chat(conversation: list[dict]) -> list[dict]:
    """
    `conversation` is a list of {"role": "user"|"assistant", "content": str}
    messages, NOT including the system prompt (added here so callers don't
    have to remember to). Returns the updated conversation — including
    whatever tool-call round trips happened along the way — ending in a
    final assistant text message.
    """
    client = _get_client()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *conversation]

    for _ in range(MAX_TOOL_ROUNDS):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=deployment,  # for Azure, this is the DEPLOYMENT name, not necessarily the underlying model's name
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        log_call(
            call_type="chat",
            deployment=deployment,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=(time.perf_counter() - start) * 1000,
            caller="llm.client.chat",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            return messages[1:]  # drop the system prompt before handing the conversation back to the caller

        for tool_call in assistant_message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,  # links this result back to the specific call the model made
                    "content": _execute_tool_call(tool_call),
                }
            )

    raise RuntimeError(f"tool-calling loop did not converge after {MAX_TOOL_ROUNDS} rounds")
