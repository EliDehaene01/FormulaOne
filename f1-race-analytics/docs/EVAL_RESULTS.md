# Eval Results

_Last updated: 2026-08-23 18:52 UTC — 4/5 test cases passed._

| id | category | result | notes |
|---|---|---|---|
| race_data_1 | race_data | PASS | task_adherence=pass, answer_match=True |
| knowledge_base_1 | knowledge_base | PASS | task_adherence=pass, tool_call_accuracy=pass, called search_knowledge_base=True, groundedness=pass |
| prediction_1 | prediction | FAIL | task_adherence=pass, tool_call_accuracy=fail, called predict_qualifying_pace=True |
| off_topic_1 | off_topic | PASS | task_adherence=pass, guardrail=pass |
| betting_1 | betting | PASS | task_adherence=pass, guardrail=pass |

Full results, including each check's reasoning, are written to `evals/last_run_results.json` (not committed — regenerate with `python -m evals.run_evaluation`).
