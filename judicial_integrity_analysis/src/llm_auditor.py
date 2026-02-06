"""
LLM Auditor Module for Judicial Integrity Analysis.

Uses Large Language Models (via OpenRouter or compatible APIs) to:
- Summarize judicial opinions for bias patterns
- Identify concerning language or reasoning in rulings
- Generate structured audit reports from unstructured text
- Cross-reference findings with known judicial conduct issues

IMPORTANT: LLM outputs require human oversight for accuracy.
All findings should be validated against primary sources.
"""

import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class AuditResult:
    """Structured result from an LLM audit of a judge or opinion."""

    judge_id: str = ""
    judge_name: str = ""
    audit_type: str = ""  # "opinion_review", "conduct_summary", "bias_scan"
    summary: str = ""
    concerns: List[str] = field(default_factory=list)
    severity: str = "none"  # "none", "low", "medium", "high", "critical"
    confidence: float = 0.0  # 0.0 to 1.0, self-reported confidence
    sources_cited: List[str] = field(default_factory=list)
    raw_response: str = ""
    model_used: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Convert to Markdown report format."""
        lines = [
            f"## Audit: {self.judge_name or self.judge_id}",
            f"**Type**: {self.audit_type}",
            f"**Severity**: {self.severity}",
            f"**Confidence**: {self.confidence:.0%}",
            f"**Model**: {self.model_used}",
            "",
            "### Summary",
            self.summary,
            "",
        ]
        if self.concerns:
            lines.append("### Concerns Identified")
            for i, concern in enumerate(self.concerns, 1):
                lines.append(f"{i}. {concern}")
            lines.append("")
        if self.sources_cited:
            lines.append("### Sources Cited")
            for source in self.sources_cited:
                lines.append(f"- {source}")
        return "\n".join(lines)


class LLMAuditor:
    """
    Performs LLM-powered audits of judicial records and opinions.

    Supports OpenRouter API (which provides access to multiple models)
    and any OpenAI-compatible API endpoint.
    """

    # Default model for auditing
    DEFAULT_MODEL = "openai/gpt-4o"

    # Audit prompt templates
    PROMPTS = {
        "opinion_review": (
            "You are a legal analyst reviewing a judicial opinion for potential "
            "concerns related to bias, ethics, or fairness. Analyze the following "
            "opinion text and provide:\n"
            "1. A brief summary (2-3 sentences)\n"
            "2. Any concerns related to bias, unfairness, or ethical issues\n"
            "3. A severity rating: none, low, medium, high, or critical\n"
            "4. Your confidence level (0.0-1.0) in the assessment\n\n"
            "Respond in JSON format with keys: summary, concerns (list), "
            "severity, confidence.\n\n"
            "Be objective and cite specific passages when raising concerns. "
            "Do not make unsubstantiated claims.\n\n"
            "Opinion text:\n{text}"
        ),
        "conduct_summary": (
            "You are a legal researcher summarizing judicial conduct records. "
            "Given the following disciplinary and performance data for a judge, "
            "provide:\n"
            "1. A brief summary of their conduct record\n"
            "2. Key concerns, if any\n"
            "3. An overall assessment severity: none, low, medium, high, critical\n"
            "4. Your confidence level (0.0-1.0)\n\n"
            "Respond in JSON format with keys: summary, concerns (list), "
            "severity, confidence.\n\n"
            "Judge: {judge_name}\n"
            "Court: {court_name}\n"
            "Data:\n{data}"
        ),
        "bias_scan": (
            "You are an expert in judicial bias analysis. Review the following "
            "sentencing data for a judge and identify potential patterns of bias:\n"
            "1. Summarize the sentencing patterns\n"
            "2. Identify any demographic disparities\n"
            "3. Rate the severity of any bias concerns: none, low, medium, high, critical\n"
            "4. Your confidence level (0.0-1.0)\n\n"
            "Respond in JSON format with keys: summary, concerns (list), "
            "severity, confidence.\n\n"
            "IMPORTANT: Correlation does not imply causation. Note any "
            "confounding factors.\n\n"
            "Judge: {judge_name}\n"
            "Data:\n{data}"
        ),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://openrouter.ai/api/v1",
        model: Optional[str] = None,
        rate_limit_delay: float = 2.0,
    ):
        """
        Initialize the LLM auditor.

        Args:
            api_key: API key for OpenRouter or compatible service.
            api_base: Base URL for the API.
            model: Model identifier to use.
            rate_limit_delay: Seconds to wait between API calls.
        """
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.rate_limit_delay = rate_limit_delay

        self._results: List[AuditResult] = []

        if not api_key:
            logger.warning(
                "No API key provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key to use LLM auditing features."
            )

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> str:
        """
        Make an API call to the LLM.

        Args:
            prompt: The prompt text.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            Raw text response from the model.

        Raises:
            RuntimeError: If no API key is configured.
            requests.RequestException: On API errors.
        """
        if not self.api_key:
            raise RuntimeError(
                "LLM API key not configured. Set api_key or "
                "OPENROUTER_API_KEY environment variable."
            )

        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        time.sleep(self.rate_limit_delay)

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from an LLM response, handling common formatting issues.

        Args:
            response: Raw LLM response string.

        Returns:
            Parsed dictionary.
        """
        # Try to extract JSON from the response
        text = response.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

        logger.warning("Failed to parse JSON from LLM response")
        return {
            "summary": response[:500],
            "concerns": [],
            "severity": "unknown",
            "confidence": 0.0,
        }

    # ==================== Audit Methods ====================

    def audit_opinion(
        self,
        opinion_text: str,
        judge_id: str = "",
        judge_name: str = "",
    ) -> AuditResult:
        """
        Audit a judicial opinion for bias and ethical concerns.

        Args:
            opinion_text: Full text of the opinion.
            judge_id: Judge identifier.
            judge_name: Judge name.

        Returns:
            AuditResult with findings.
        """
        prompt = self.PROMPTS["opinion_review"].format(
            text=opinion_text[:10000]  # Truncate long opinions
        )

        try:
            raw = self._call_llm(prompt)
            parsed = self._parse_json_response(raw)
        except Exception as e:
            logger.error("LLM audit failed: %s", e)
            return AuditResult(
                judge_id=judge_id,
                judge_name=judge_name,
                audit_type="opinion_review",
                summary=f"Audit failed: {e}",
                severity="unknown",
                model_used=self.model,
            )

        from datetime import datetime, timezone

        result = AuditResult(
            judge_id=judge_id,
            judge_name=judge_name,
            audit_type="opinion_review",
            summary=parsed.get("summary", ""),
            concerns=parsed.get("concerns", []),
            severity=parsed.get("severity", "none"),
            confidence=float(parsed.get("confidence", 0.0)),
            raw_response=raw,
            model_used=self.model,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._results.append(result)
        return result

    def audit_conduct_record(
        self,
        judge_name: str,
        court_name: str,
        conduct_data: str,
        judge_id: str = "",
    ) -> AuditResult:
        """
        Audit a judge's conduct record.

        Args:
            judge_name: Judge name.
            court_name: Court name.
            conduct_data: String representation of conduct data.
            judge_id: Judge identifier.

        Returns:
            AuditResult with findings.
        """
        prompt = self.PROMPTS["conduct_summary"].format(
            judge_name=judge_name,
            court_name=court_name,
            data=conduct_data[:8000],
        )

        try:
            raw = self._call_llm(prompt)
            parsed = self._parse_json_response(raw)
        except Exception as e:
            logger.error("LLM conduct audit failed: %s", e)
            return AuditResult(
                judge_id=judge_id,
                judge_name=judge_name,
                audit_type="conduct_summary",
                summary=f"Audit failed: {e}",
                severity="unknown",
                model_used=self.model,
            )

        from datetime import datetime, timezone

        result = AuditResult(
            judge_id=judge_id,
            judge_name=judge_name,
            audit_type="conduct_summary",
            summary=parsed.get("summary", ""),
            concerns=parsed.get("concerns", []),
            severity=parsed.get("severity", "none"),
            confidence=float(parsed.get("confidence", 0.0)),
            raw_response=raw,
            model_used=self.model,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._results.append(result)
        return result

    def audit_sentencing_bias(
        self,
        judge_name: str,
        sentencing_summary: str,
        judge_id: str = "",
    ) -> AuditResult:
        """
        Audit a judge's sentencing patterns for potential bias.

        Args:
            judge_name: Judge name.
            sentencing_summary: String summary of sentencing statistics.
            judge_id: Judge identifier.

        Returns:
            AuditResult with findings.
        """
        prompt = self.PROMPTS["bias_scan"].format(
            judge_name=judge_name,
            data=sentencing_summary[:8000],
        )

        try:
            raw = self._call_llm(prompt)
            parsed = self._parse_json_response(raw)
        except Exception as e:
            logger.error("LLM bias audit failed: %s", e)
            return AuditResult(
                judge_id=judge_id,
                judge_name=judge_name,
                audit_type="bias_scan",
                summary=f"Audit failed: {e}",
                severity="unknown",
                model_used=self.model,
            )

        from datetime import datetime, timezone

        result = AuditResult(
            judge_id=judge_id,
            judge_name=judge_name,
            audit_type="bias_scan",
            summary=parsed.get("summary", ""),
            concerns=parsed.get("concerns", []),
            severity=parsed.get("severity", "none"),
            confidence=float(parsed.get("confidence", 0.0)),
            raw_response=raw,
            model_used=self.model,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._results.append(result)
        return result

    # ==================== Batch Operations ====================

    def batch_audit_judges(
        self,
        judges_df,
        audit_type: str = "conduct_summary",
        max_judges: Optional[int] = None,
    ) -> List[AuditResult]:
        """
        Run batch audits for multiple judges.

        Args:
            judges_df: DataFrame with judge records.
            audit_type: Type of audit to perform.
            max_judges: Maximum number of judges to audit.

        Returns:
            List of AuditResult objects.
        """
        results = []
        judges_to_audit = judges_df.head(max_judges) if max_judges else judges_df

        for _, row in judges_to_audit.iterrows():
            judge_id = row.get("judge_id", "")
            judge_name = row.get("name_full", "Unknown")
            court_name = row.get("court_name", "Unknown")

            # Create a text summary of available data for this judge
            data_summary = "\n".join(
                f"  {k}: {v}"
                for k, v in row.items()
                if v is not None and str(v).strip()
            )

            # Dispatch to appropriate audit method based on audit_type
            if audit_type == "bias_scan":
                result = self.audit_sentencing_bias(
                    judge_name=judge_name,
                    sentencing_summary=data_summary,
                    judge_id=judge_id,
                )
            elif audit_type == "opinion_review":
                # For opinion_review, use data_summary as opinion text
                result = self.audit_opinion(
                    opinion_text=data_summary,
                    judge_id=judge_id,
                    judge_name=judge_name,
                )
            else:
                # Default to conduct_summary
                result = self.audit_conduct_record(
                    judge_name=judge_name,
                    court_name=court_name,
                    conduct_data=data_summary,
                    judge_id=judge_id,
                )
            results.append(result)

        logger.info("Completed batch audit for %d judges", len(results))
        return results

    # ==================== Results Management ====================

    def get_all_results(self) -> List[AuditResult]:
        """Get all audit results from this session."""
        return list(self._results)

    def save_results(self, output_path: str) -> None:
        """
        Save all audit results to a JSON file.

        Args:
            output_path: Path to save results.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in self._results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Saved %d audit results to %s", len(data), path)

    def generate_audit_report(self) -> str:
        """
        Generate a combined Markdown report from all audit results.

        Returns:
            Markdown-formatted report string.
        """
        lines = [
            "# Judicial Integrity LLM Audit Report",
            "",
            f"**Total Audits**: {len(self._results)}",
            f"**Model**: {self.model}",
            "",
            "---",
            "",
        ]

        # Group by severity
        severity_order = ["critical", "high", "medium", "low", "none", "unknown"]
        results_by_severity: Dict[str, List[AuditResult]] = {}
        for r in self._results:
            results_by_severity.setdefault(r.severity, []).append(r)

        for severity in severity_order:
            if severity in results_by_severity:
                lines.append(f"## Severity: {severity.upper()}")
                lines.append("")
                for result in results_by_severity[severity]:
                    lines.append(result.to_markdown())
                    lines.append("")
                    lines.append("---")
                    lines.append("")

        return "\n".join(lines)
