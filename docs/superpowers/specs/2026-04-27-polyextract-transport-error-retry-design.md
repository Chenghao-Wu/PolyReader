# Polyextract: route transport errors through the retry path

**Date:** 2026-04-27
**Author:** Bruce
**Scope:** Bug fix in `polyextract/guided_extraction.py` and a small refactor of `_is_transport_error`.

## Problem

When the per-entity guided extraction loop encounters a transient LLM transport error (connection drop, 502/503/429, timeout), the sample is silently downgraded to `quality: reject` without any retry, and the orchestrator reports the run as `SUCCESS`. Concretely, on the Bicheng Han `ma011288v` paper, AMBA3 was lost this way despite Table 2 containing valid Mn / Mw / PDI for it.

### Trace of the bug

1. `guided_extraction.py:189-200` — the `for iteration in range(max_iterations)` loop wraps `await llm.chat_with_tools(...)` in a broad `except Exception`. On a connection error it logs and `break`s. The exception is **swallowed**.
2. Because the loop exited via `break` (not via `for…else`), the wrap-up at line 246-248 does not run; `final_answer` stays empty.
3. `guided_extraction.py:255-256` returns a synthetic error dict: `{"sample_id": "", "polymer_id": entity_id, "error": "No answer produced"}`. This dict is **truthy**.
4. `extraction_loop.py:131` (`if not raw_data: continue`) does not fire because the dict is truthy.
5. `extraction_loop.py:146` — Pydantic validates the minimal dict because every required field on `PolymerSample` has a default factory.
6. `extraction_loop.py:186` — `classify_quality_tier` finds 0/4 must-haves and returns `reject`.
7. `extraction_loop.py:200` — returns `EntityExtractionOutcome.SUCCESS` with a reject-tier sample. The transport-error retry path at lines 212-221 only catches exceptions that propagate up from `extract_entity`, but the error was caught two stack frames below.

The retry+backoff machinery exists (`extraction_loop.py:53-68` and 212-221) but never fires for guided-extraction transport errors because those errors don't escape `extract_entity`.

The prompts and the sample-id expansion logic are correct. Discovery returned `sample_ids=["AMBA1","AMBA2","AMBA3","PAMBA macro-CTA"]` and the orchestrator created four targeted tasks. Only AMBA3 hit the transport error.

## Goal

A transient transport error inside guided extraction should trigger the existing retry + exponential backoff in `extraction_loop`, identical to a transport error raised directly by `extract_entity`. Non-transport errors keep their current behavior (log + return error dict + reject tier) so we don't mask unrelated bugs.

## Non-goals

- No new retry mechanism. We're routing through the one that already exists.
- No change to prompts, entity discovery, sample expansion, or reconciliation.
- No partial-state preservation across retries. The next attempt restarts the guided loop from scratch and will re-issue the same tool calls. Cost is some duplicated tool calls; benefit is correctness.

## Design

### Component layout

```
polyextract/
├── _transport.py           [NEW] tiny shared module
│   └── is_transport_error(exc) -> bool
├── guided_extraction.py    [MODIFY] re-raise transport errors at 2 callsites
├── tools.py                [MODIFY] re-raise transport errors in wrap_up_conversation
└── extraction_loop.py      [MODIFY] import is_transport_error from _transport
```

### `polyextract/_transport.py` (new, ~15 lines)

Move `_is_transport_error` out of `extraction_loop.py:53-68` and into a small shared module so `guided_extraction` can import it without creating a circular dependency (extraction_loop already imports from guided_extraction). Public name: `is_transport_error` (drop the leading underscore since it now crosses module boundaries).

Body identical to the current implementation: lowercase the stringified exception and check for any of the existing indicators (`rate limit`, `timeout`, `connection`, `disconnected`, `network`, `service unavailable`, `503`, `502`, `429`, `overloaded`).

### Three callsites swallow LLM exceptions

All three live on the path between `extract_entity` and the LLM client. Each must be patched to re-raise transport errors while preserving current behavior for non-transport errors.

1. **`guided_extraction.py:198-200`** — main tool-calling loop (`llm.chat_with_tools`):
   ```python
   except Exception as e:
       logger.error(f"Guided extraction LLM call failed: {e}")
       break
   ```
   Becomes:
   ```python
   except Exception as e:
       if is_transport_error(e):
           logger.warning(f"Guided extraction transport error at iter {iteration + 1}: {e}")
           raise
       logger.error(f"Guided extraction LLM call failed: {e}")
       break
   ```

2. **`guided_extraction.py:107-110`** — `_extract_with_structured_output` (`llm.chat_structured`). This is the final structured-output call that runs *after* the guided loop completes. It also swallows transport errors and returns `{"sample_id": "", "polymer_id": entity_id, "error": str(e)}`, producing the same reject-tier outcome:
   ```python
   except Exception as e:
       logger.error(f"Structured extraction failed for '{entity_id}': {e}")
       return {"sample_id": "", "polymer_id": entity_id, "error": str(e)}
   ```
   Becomes: re-raise if `is_transport_error(e)`; otherwise keep current fallback. Note: Instructor's `max_retries=3` (line 104) only retries validation errors; transport errors still escape, so this guard is necessary.

3. **`tools.py:383-392`** — `wrap_up_conversation` (`llm.chat_with_tools` with `tool_choice="none"`). Used by the `for…else` branch when the guided loop hits `max_iterations`. Apply the same re-raise treatment.

### `extraction_loop.py` change

Replace `_is_transport_error` (line 53-68) with `from ._transport import is_transport_error`. Update the call site at line 213 to use the new name. No behavioral change to the retry loop itself — the existing exponential backoff at lines 215-221 already does the right thing.

### Behavior after the fix

- Transport error at iter K of guided extraction → re-raise → caught at `extraction_loop.py:212-213` → `is_transport_error` returns True → `await asyncio.sleep((2 ** attempt) + random)` → `continue` → fresh attempt.
- After 3 attempts fail with transport errors: the final iteration falls out of the retry loop and returns `EntityExtractionOutcome.VALIDATION_FAILED` at line 232. The orchestrator already handles this outcome (the sample is omitted from emitted output, not silently emitted as reject).
- Non-transport errors (e.g., a programming error in a tool) → unchanged: log + break + synthetic error dict + reject tier. We surface real bugs as reject samples rather than swallowing them as crashes.

## Testing

One new unit test in `tests/polyextract/test_extraction_loop_retry.py` (or wherever existing extraction_loop tests live):

- **Test:** `test_transport_error_in_guided_extraction_triggers_retry`
  - Mock `LLMClient.chat_with_tools` to raise `ConnectionError("Connection error")` on the first call, then return a valid structured response on the second.
  - Call `extract_single_entity` with `max_retries=3`.
  - Assert: outcome is `SUCCESS`, sample is the valid response, and `chat_with_tools` was awaited at least twice.

- **Test (regression):** `test_non_transport_error_in_guided_extraction_does_not_retry`
  - Mock to raise `ValueError("bad json")`.
  - Assert: only one attempt of guided extraction, sample comes back with `quality_tier == "reject"`, outcome is `SUCCESS` (current behavior preserved).

No integration test against a real model — the unit-level mock covers the routing logic which is the only thing changing.

## Risks

- **Re-running tool calls on retry costs tokens.** Acceptable: transport errors are rare, and the alternative (silent data loss) is worse. Existing extraction_loop already accepts this cost for transport errors raised directly by `extract_entity`.
- **`is_transport_error` is a string-match heuristic.** Same risk as before — we're not changing the heuristic, just sharing it. If it ever needs upgrading (e.g., to inspect HTTP status codes via exception type), both callsites will benefit at once.
- **Circular-import surface.** Mitigated by putting the helper in its own tiny module rather than threading it through an existing one.

## Out of scope (consider for future work)

- Surface a distinct `EntityExtractionOutcome.TRANSPORT_ERROR` after exhausting retries instead of `VALIDATION_FAILED`. The enum value already exists at `extraction_loop.py:35` but is unused. Useful for the orchestrator's summary stats but not required to fix this bug.
- Inspect the partial conversation when guided extraction fails to see whether a structured-output retry could salvage the run without restarting tool calls. Bigger change; revisit only if transport errors become common.
