"""
app/services/pipeline_service.py
─────────────────────────────────
Orchestrates triage → quality → SLA recommendation.

Runs triage and quality concurrently via threading to reduce wall-clock latency.
"""

from __future__ import annotations

import concurrent.futures

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.domain import (
    PipelineRequest,
    PipelineResult,
    Priority,
    QualityRequest,
    TriageRequest,
)
from app.services.quality_service import QualityService
from app.services.triage_service import TriageService

logger = get_logger(__name__)


class PipelineService:
    def __init__(
        self,
        triage_service: TriageService,
        quality_service: QualityService,
        settings: Settings,
    ) -> None:
        self._triage = triage_service
        self._quality = quality_service
        self._settings = settings

    def run(self, request: PipelineRequest) -> PipelineResult:
        """
        Execute the full triage + quality pipeline.

        Triage and quality are run concurrently since they are independent.
        Both call the LLM, so running in parallel cuts end-to-end latency ~50%.
        """
        logger.info("Starting pipeline run")

        triage_req = TriageRequest(ticket_text=request.ticket_text)
        quality_req = QualityRequest(
            ticket_text=request.ticket_text,
            agent_response=request.agent_response,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            triage_future = executor.submit(self._triage.triage, triage_req)
            quality_future = executor.submit(self._quality.evaluate, quality_req)

            triage_result = triage_future.result()
            quality_result = quality_future.result()

        sla_minutes = self._settings.sla_map[triage_result.priority.value]

        # Pipeline passes only if quality passes AND ticket is not critical-and-unresolved
        workflow_passed = quality_result.passed and not (
            triage_result.priority == Priority.CRITICAL
            and not quality_result.checks.resolved_or_escalated
        )

        result = PipelineResult(
            triage=triage_result,
            quality=quality_result,
            recommended_sla_minutes=sla_minutes,
            workflow_passed=workflow_passed,
        )

        logger.info(
            "Pipeline complete",
            priority=triage_result.priority,
            quality_score=quality_result.score,
            sla_minutes=sla_minutes,
            workflow_passed=workflow_passed,
        )
        return result
