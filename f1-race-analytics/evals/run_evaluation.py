"""
Runs backend/llm/client.py's actual chat loop against the fixed test cases
in evals/test_cases.jsonl and scores the results with Azure AI Foundry's
evaluation SDK (azure-ai-evaluation) — using the Foundry project already
configured in backend/.env, no new Azure resource needed.

Run with:
    python -m evals.run_evaluation
    python -m evals.run_evaluation --push-to-foundry

WHICH EVALUATOR RUNS ON WHICH TEST CASE
------------------------------------------
- TaskAdherence: every test case — did the assistant actually address the
  request while following SYSTEM_PROMPT's rules (grounding, scope, etc.)?
- ToolCallAccuracy + a direct "was the expected tool actually called" check:
  only test cases with an `expected_tool` (knowledge_base, prediction).
- Groundedness: only the knowledge_base test case — checked against the
  ACTUAL chunks search_knowledge_base returned in that conversation, not a
  hand-written reference answer, so it's measuring the real RAG pipeline.
- The custom GuardrailEvaluator: only test cases with a `guardrail_rule`
  (off_topic, betting) — a domain rule none of the built-in evaluators know.
- A plain substring check against `expected_answer_contains`: only the
  race_data test case, since that one has one verifiably correct answer.

A test case "passes" only if every check that applies to it passes.
"""

import argparse
import json
import os
from pathlib import Path

from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    GroundednessEvaluator,
    TaskAdherenceEvaluator,
    ToolCallAccuracyEvaluator,
    evaluate,
)
from dotenv import load_dotenv

from backend.llm.client import SYSTEM_PROMPT, chat
from backend.llm.tools import TOOL_SCHEMAS
from evals.guardrail_evaluator import GuardrailEvaluator

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

TEST_CASES_PATH = Path(__file__).parent / "test_cases.jsonl"
RESULTS_PATH = Path(__file__).parent / "last_run_results.json"


def _model_config() -> AzureOpenAIModelConfiguration:
    return AzureOpenAIModelConfiguration(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


def _load_test_cases() -> list[dict]:
    with open(TEST_CASES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _context_message(test_case: dict) -> dict | None:
    """Mirrors exactly what the real frontend sends (see ChatPanel.tsx) when a race is selected — so a test case with `session_context` doesn't need to name the session_key in its question text."""
    session_context = test_case.get("session_context")
    if not session_context:
        return None
    return {
        "role": "user",
        "content": f"[Context: user is currently viewing {session_context['label']}, session_key={session_context['session_key']}]",
    }


def _final_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return ""


def _tool_names_called(messages: list[dict]) -> list[str]:
    return [call["function"]["name"] for m in messages for call in (m.get("tool_calls") or [])]


def _to_evaluator_messages(messages: list[dict]) -> list[dict]:
    """
    Converts chat()'s OpenAI chat.completions-format messages (an assistant
    tool call as {id, type, function:{name, arguments}}, a tool result as a
    plain string) into the Azure AI Agents message schema
    azure.ai.evaluation's evaluators actually validate against — `content`
    as a list of typed items ({"type": "tool_call", "name":, "arguments":,
    "tool_call_id":} / {"type": "tool_result", "tool_result":}), NOT a plain
    OpenAI-shaped message. Worked out by reading
    _evaluators/_common/_validators/_conversation_validator.py directly,
    since the evaluators' own auto-detection couldn't parse our messages
    (silently degrades accuracy instead of erroring) and a plain "give every
    message a content key" fix still failed the SDK's own field-shape checks.
    """
    converted = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            content: list[dict] = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                try:
                    arguments = json.loads(tc["function"]["arguments"])
                except (TypeError, ValueError):
                    arguments = {}
                content.append({"type": "tool_call", "tool_call_id": tc["id"], "name": tc["function"]["name"], "arguments": arguments})
            converted.append({"role": "assistant", "content": content})
        elif role == "tool":
            converted.append({"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": [{"type": "tool_result", "tool_result": m.get("content")}]})
        else:
            converted.append({**m, "content": m.get("content") or "(no content)"})
    return converted


def _to_evaluator_tool_definitions(tool_schemas: list[dict]) -> list[dict]:
    """Flattens OpenAI-format tool schemas ({"type": "function", "function": {"name":, "description":, "parameters":}}) — what TOOL_SCHEMAS actually is — into the flat {"name":, "description":, "parameters":} shape ToolCallAccuracyEvaluator's validator requires."""
    return [
        {"name": t["function"]["name"], "description": t["function"].get("description", ""), "parameters": t["function"].get("parameters", {})}
        for t in tool_schemas
        if t.get("type") == "function"
    ]


def _tool_context(messages: list[dict], tool_name: str) -> str:
    """Concatenates every result returned by calls to `tool_name` in this conversation — used as Groundedness's ground-truth context."""
    ids = {c["id"] for m in messages for c in (m.get("tool_calls") or []) if c["function"]["name"] == tool_name}
    return "\n\n".join(m["content"] for m in messages if m.get("role") == "tool" and m.get("tool_call_id") in ids)


def run_test_case(test_case: dict, evaluators: dict) -> dict:
    context_message = _context_message(test_case)
    conversation = ([context_message] if context_message else []) + [{"role": "user", "content": test_case["question"]}]

    result_messages = chat(conversation)  # conversation + whatever the assistant/tools produced, per chat()'s own contract (system prompt already dropped)
    response_messages = result_messages[len(conversation) :]
    final_text = _final_text(response_messages)
    tools_called = _tool_names_called(response_messages)

    row: dict = {
        "id": test_case["id"],
        "category": test_case["category"],
        "question": test_case["question"],
        "final_response": final_text,
        "tools_called": tools_called,
    }

    query_messages = _to_evaluator_messages([{"role": "system", "content": SYSTEM_PROMPT}, *conversation])
    eval_response_messages = _to_evaluator_messages(response_messages)

    ta = evaluators["task_adherence"](query=query_messages, response=eval_response_messages)
    row["task_adherence"] = {"result": ta.get("task_adherence_result"), "score": ta.get("task_adherence_score"), "reason": ta.get("task_adherence_reason")}

    expected_tool = test_case.get("expected_tool")
    if expected_tool:
        row["expected_tool"] = expected_tool
        row["expected_tool_called"] = expected_tool in tools_called
        tca = evaluators["tool_call_accuracy"](query=query_messages, tool_definitions=_to_evaluator_tool_definitions(TOOL_SCHEMAS), response=eval_response_messages)
        row["tool_call_accuracy"] = {
            "result": tca.get("tool_call_accuracy_result"),
            "score": tca.get("tool_call_accuracy_score"),
            "reason": tca.get("tool_call_accuracy_reason"),
        }

    if test_case["category"] == "knowledge_base":
        context = _tool_context(response_messages, "search_knowledge_base") or "<no knowledge base results returned>"
        gr = evaluators["groundedness"](query=test_case["question"], response=final_text, context=context)
        row["groundedness"] = {"result": gr.get("groundedness_result"), "score": gr.get("groundedness_score"), "reason": gr.get("groundedness_reason")}

    expected_answer_contains = test_case.get("expected_answer_contains")
    if expected_answer_contains:
        row["expected_answer_contains"] = expected_answer_contains
        row["expected_answer_match"] = all(term.lower() in final_text.lower() for term in expected_answer_contains)

    guardrail_rule = test_case.get("guardrail_rule")
    if guardrail_rule:
        gd = evaluators["guardrail"](query=test_case["question"], response=final_text, rule=guardrail_rule)
        row["guardrail"] = {"result": gd.get("guardrail_result"), "reason": gd.get("guardrail_reason")}

    return row


def _row_passed(row: dict) -> bool:
    checks = []
    if "task_adherence" in row:
        checks.append(row["task_adherence"]["result"] == "pass")
    if "tool_call_accuracy" in row:
        checks.append(row["tool_call_accuracy"]["result"] == "pass")
    if "expected_tool_called" in row:
        checks.append(row["expected_tool_called"])
    if "groundedness" in row:
        checks.append(row["groundedness"]["result"] == "pass")
    if "expected_answer_match" in row:
        checks.append(row["expected_answer_match"])
    if "guardrail" in row:
        checks.append(row["guardrail"]["result"] == "pass")
    return bool(checks) and all(checks)


def _target(question: str, session_context: dict | None = None, **_ignored) -> dict:
    """Adapter for azure.ai.evaluation's `evaluate()` orchestrator (used only by --push-to-foundry): turns one test_cases.jsonl row into the {query, response, tool_definitions} shape it expects, by running it through the real chat loop exactly like run_test_case does."""
    context_message = None
    if session_context:
        context_message = {"role": "user", "content": f"[Context: user is currently viewing {session_context['label']}, session_key={session_context['session_key']}]"}
    conversation = ([context_message] if context_message else []) + [{"role": "user", "content": question}]
    result_messages = chat(conversation)
    response_messages = result_messages[len(conversation) :]
    return {
        "query": _to_evaluator_messages([{"role": "system", "content": SYSTEM_PROMPT}, *conversation]),
        "response": _to_evaluator_messages(response_messages),
        "tool_definitions": _to_evaluator_tool_definitions(TOOL_SCHEMAS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pit Wall's fixed LLM evaluation suite.")
    parser.add_argument(
        "--push-to-foundry",
        action="store_true",
        help="Also upload results to the Foundry portal's evaluation tab (needs AZURE_AI_PROJECT_ENDPOINT set in backend/.env). Re-runs every test case a second time through azure.ai.evaluation's own orchestrator — separate cost from the primary run above.",
    )
    args = parser.parse_args()

    model_config = _model_config()
    evaluators = {
        "task_adherence": TaskAdherenceEvaluator(model_config=model_config),
        "tool_call_accuracy": ToolCallAccuracyEvaluator(model_config=model_config),
        "groundedness": GroundednessEvaluator(model_config=model_config),
        "guardrail": GuardrailEvaluator(model_config=model_config),
    }

    rows = []
    for test_case in _load_test_cases():
        print(f"Running {test_case['id']} ({test_case['category']}) ...")
        try:
            row = run_test_case(test_case, evaluators)
        except Exception as exc:
            row = {"id": test_case["id"], "category": test_case["category"], "question": test_case["question"], "error": str(exc)}
        rows.append(row)

    RESULTS_PATH.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    print()
    print(f"{'id':16s} {'category':14s} {'pass?':6s} notes")
    print("-" * 100)
    n_passed = 0
    for row in rows:
        if "error" in row:
            print(f"{row['id']:16s} {row['category']:14s} {'ERROR':6s} {row['error'][:70]}")
            continue
        passed = _row_passed(row)
        n_passed += int(passed)
        notes = []
        if "task_adherence" in row:
            notes.append(f"task_adherence={row['task_adherence']['result']}")
        if "tool_call_accuracy" in row:
            notes.append(f"tool_call_accuracy={row['tool_call_accuracy']['result']}")
        if "expected_tool_called" in row:
            notes.append(f"called {row['expected_tool']}={row['expected_tool_called']}")
        if "groundedness" in row:
            notes.append(f"groundedness={row['groundedness']['result']}")
        if "expected_answer_match" in row:
            notes.append(f"answer_match={row['expected_answer_match']}")
        if "guardrail" in row:
            notes.append(f"guardrail={row['guardrail']['result']}")
        print(f"{row['id']:16s} {row['category']:14s} {('PASS' if passed else 'FAIL'):6s} {', '.join(notes)}")
    print("-" * 100)
    print(f"{n_passed}/{len(rows)} test cases passed. Full results (incl. reasoning) written to {RESULTS_PATH}")

    if args.push_to_foundry:
        project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
        if not project_endpoint:
            print("\n--push-to-foundry set, but AZURE_AI_PROJECT_ENDPOINT isn't set in backend/.env — skipping the portal upload.")
        else:
            print("\nRe-running through azure.ai.evaluation's own orchestrator and uploading to the Foundry portal...")
            try:
                result = evaluate(
                    data=str(TEST_CASES_PATH),
                    target=_target,
                    evaluators={"task_adherence": evaluators["task_adherence"], "tool_call_accuracy": evaluators["tool_call_accuracy"]},
                    evaluation_name="pit-wall-eval-suite",
                    azure_ai_project=project_endpoint,
                )
                studio_url = result.get("studio_url")
                print(f"Uploaded. View it in the Foundry portal: {studio_url}" if studio_url else "Uploaded — check the Evaluation tab in your Foundry project.")
            except Exception as exc:
                # The API key in .env authenticates chat/embeddings calls, but
                # uploading TO the Foundry portal is a separate, AAD-based
                # (Entra ID) operation under the hood — almost always the
                # cause the first time this is run on a machine that hasn't
                # done `az login` yet. Give a beginner-friendly nudge instead
                # of a raw stack trace, since that's the most likely fix.
                print(f"\nUpload failed: {exc}")
                print("If this looks like an authentication/credential error: run `az login` in a terminal (Azure CLI must be installed), then try --push-to-foundry again.")


if __name__ == "__main__":
    main()
