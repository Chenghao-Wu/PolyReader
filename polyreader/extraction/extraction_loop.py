"""Per-entity extraction loop with validation and retry.

Wraps GuidedExtractionAgent with domain validation, transport error backoff,
and entity triage handling. Pydantic validation is handled internally by
Instructor in the extraction agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from enum import Enum
from typing import Any

from pydantic import ValidationError

from polyreader.llm.client import LLMClient

from .guided_extraction import extract_entity
from .config import ExtractionConfig
from .document_set import DocumentSet
from polyreader.schema.polymer import PolymerSample, TYPE_BASE, TYPE_DERIVED_BLEND, TYPE_COMPOSITE
from polyreader.schema.validation import (
    MW_LESS_THAN_MN,
    PDI_INCONSISTENT,
    canonicalize_smiles,
    classify_quality_tier,
    rescue_mw_from_properties,
    validate_sample_issues,
)
from ._transport import is_transport_error

logger = logging.getLogger(__name__)


class EntityExtractionOutcome(Enum):
    """Outcome of a single entity extraction attempt."""

    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    TRANSPORT_ERROR = "transport_error"
    PARSE_ERROR = "parse_error"
    SKIPPED = "skipped"  # Entity triage: not synthesized in paper
    FATAL = "fatal"


# Validation issue codes that warrant retry (critical issues).
CRITICAL_FLAGS = {
    MW_LESS_THAN_MN,
    PDI_INCONSISTENT,
}


def _is_critical_flag(flag: str) -> bool:
    """Check if a validation issue code is critical enough to warrant retry."""
    return flag in CRITICAL_FLAGS


def postprocess_raw_sample(
    raw_data: dict[str, Any],
) -> tuple[PolymerSample, list[str], list]:
    """Validate one raw extraction result like the extraction loop does.

    Returns ``(sample, flags, critical_issues)``; raises pydantic's
    ValidationError when the data does not fit the schema. The caller decides
    what to do with critical issues and then calls :func:`_finalize_sample`.
    """
    sample = PolymerSample.model_validate(raw_data)
    rescue_mw_from_properties(sample)
    issues = validate_sample_issues(sample)
    flags = [issue.message for issue in issues]
    critical = [issue for issue in issues if _is_critical_flag(issue.code)]
    return sample, flags, critical


def _finalize_sample(
    sample: PolymerSample, flags: list[str], entity_info: dict[str, Any],
) -> PolymerSample:
    """Canonicalize SMILES, set quality tier, flags and sample type."""
    canonicalize_smiles(sample)
    sample.quality_tier, sample.quality_tier_reason = classify_quality_tier(sample)
    sample.validation_flags = flags
    if entity_info.get("extraction_class") == "derived":
        sample.sample_type = TYPE_DERIVED_BLEND if "blend" in str(entity_info.get("type", "")).lower() else TYPE_COMPOSITE
    else:
        sample.sample_type = TYPE_BASE
    return sample


async def extract_single_entity(
    entity_id: str,
    entity_info: dict[str, Any],
    entity_map: dict[str, Any],
    doc_set: DocumentSet,
    llm: LLMClient,
    config: ExtractionConfig,
    base_samples_context: dict[str, Any] | None = None,
    max_retries: int | None = None,
    target_sample_id: str | None = None,
    is_multi_sample: bool = False,
) -> tuple[EntityExtractionOutcome, PolymerSample | None, list[str]]:
    """Extract and validate data for one entity with retry loop.

    Args:
        entity_id: The abbreviation/name of the entity to extract.
        entity_info: Entity metadata from entity_map.
        entity_map: Full entity map for context.
        doc_set: DocumentSet with one or more indexed documents.
        llm: LLM client for API calls.
        config: Extraction configuration.
        base_samples_context: For derived entities, extracted data from base polymers.
        max_retries: Maximum retry attempts (defaults to config.max_entity_retries).
        target_sample_id: When set, extract data for this specific sample only.
        is_multi_sample: True when this entity was expanded into multiple tasks.

    Returns:
        Tuple of (outcome, sample_or_none, validation_flags).
        - outcome: SUCCESS, VALIDATION_FAILED, SKIPPED, etc.
        - sample_or_none: Validated PolymerSample if successful, None otherwise.
        - validation_flags: List of non-blocking validation warnings.
    """
    if max_retries is None:
        max_retries = config.max_entity_retries

    task_label = target_sample_id or entity_id
    logger.info(
        f"  [extraction_loop] Starting '{task_label}' "
        f"(docs={doc_set.labels()}, retries={max_retries})"
    )

    error_history: list[dict[str, Any]] = []
    # Last sample that only failed on critical flags: returned (flagged) if
    # later attempts produce nothing better, so retries never lose data.
    fallback: tuple[PolymerSample, list[str]] | None = None
    last_failure_transport = False

    for attempt in range(max_retries):
        logger.info(f"  [extraction_loop] '{task_label}' attempt {attempt + 1}/{max_retries}")
        last_failure_transport = False
        try:
            # Run guided extraction (uses Instructor internally for Pydantic validation)
            raw_data = await extract_entity(
                doc_set=doc_set,
                llm=llm,
                entity_id=entity_id,
                entity_info=entity_info,
                entity_map=entity_map,
                config=config,
                base_samples_context=base_samples_context,
                error_history=error_history if error_history else None,
                target_sample_id=target_sample_id,
                is_multi_sample=is_multi_sample,
            )

            if not raw_data or not isinstance(raw_data, dict):
                error_history.append({
                    "attempt": attempt + 1,
                    "errors": ["No data returned from extraction agent"],
                })
                continue

            # Check if entity was triaged out
            if raw_data.get("skipped"):
                logger.info(f"  Entity '{entity_id}' was triaged out: {raw_data.get('reason', 'no reason given')}")
                return EntityExtractionOutcome.SKIPPED, None, []

            # Validate with Pydantic (data comes from Instructor, should be valid)
            # This is a safety net in case structured output was disabled or failed
            try:
                sample = PolymerSample.model_validate(raw_data)
            except ValidationError as e:
                error_msgs = [f"{list(err['loc'])}: {err['msg']}" for err in e.errors()]
                error_history.append({
                    "attempt": attempt + 1,
                    "errors": error_msgs,
                })
                logger.warning(
                    f"  Entity '{entity_id}' Pydantic validation failed "
                    f"(attempt {attempt + 1}): {error_msgs}"
                )
                if attempt < max_retries - 1:
                    continue
                if fallback is not None:
                    break
                return EntityExtractionOutcome.VALIDATION_FAILED, None, error_msgs

            # Rescue MW data that the LLM may have put in properties
            rescue_mw_from_properties(sample)

            # Domain validation (non-blocking flags; codes drive the retry)
            issues = validate_sample_issues(sample)
            flags = [issue.message for issue in issues]

            # Check for critical flags that warrant retry
            critical = [issue for issue in issues if _is_critical_flag(issue.code)]

            if critical and attempt < max_retries - 1:
                error_history.append({
                    "attempt": attempt + 1,
                    "errors": [f"Critical validation issue: {issue.message}" for issue in critical],
                })
                logger.warning(
                    f"  Entity '{entity_id}' has critical flags "
                    f"(attempt {attempt + 1}): {[issue.code for issue in critical]}"
                )
                fallback = (sample, flags)
                continue

            _finalize_sample(sample, flags, entity_info)

            logger.info(
                f"  Entity '{entity_id}' extracted successfully "
                f"(quality: {sample.quality_tier}, flags: {len(flags)})"
            )

            return EntityExtractionOutcome.SUCCESS, sample, flags

        except json.JSONDecodeError as e:
            error_history.append({
                "attempt": attempt + 1,
                "errors": [f"JSON parse error: {str(e)}"],
            })
            logger.warning(
                f"  Entity '{entity_id}' JSON parse failed "
                f"(attempt {attempt + 1}): {e}"
            )

        except Exception as e:
            if is_transport_error(e):
                last_failure_transport = True
                if attempt == max_retries - 1:
                    logger.warning(
                        f"  Entity '{entity_id}' transport error on final attempt: {e}"
                    )
                    break
                # Exponential backoff for transport errors
                backoff = (2 ** attempt) + random.random()
                logger.warning(
                    f"  Entity '{entity_id}' transport error "
                    f"(attempt {attempt + 1}), retrying in {backoff:.1f}s: {e}"
                )
                await asyncio.sleep(backoff)
                continue
            else:
                logger.error(f"  Entity '{entity_id}' fatal error: {type(e).__name__}: {e}")
                if fallback is not None:
                    break
                return EntityExtractionOutcome.FATAL, None, [str(e)]

    if fallback is not None:
        # Retries did not improve on a sample that only had critical flags.
        sample, flags = fallback
        _finalize_sample(sample, flags, entity_info)
        logger.warning(
            f"  Entity '{entity_id}' keeps its flagged result after {max_retries} attempts "
            f"(quality: {sample.quality_tier})"
        )
        return EntityExtractionOutcome.SUCCESS, sample, flags

    # All retries exhausted
    error_summary = [f"Failed after {max_retries} attempts"]
    for entry in error_history:
        error_summary.extend(entry.get("errors", []))

    logger.warning(f"  Entity '{entity_id}' extraction failed after {max_retries} attempts")
    if last_failure_transport:
        error_summary.append("Last attempt failed with a transport error")
        return EntityExtractionOutcome.TRANSPORT_ERROR, None, error_summary
    return EntityExtractionOutcome.VALIDATION_FAILED, None, error_summary
