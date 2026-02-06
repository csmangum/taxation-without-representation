"""
Report Generator Module for Judicial Integrity Analysis.

Generates structured Markdown and HTML reports for individual judges,
states, and cross-state comparisons. Designed for publication on
GitHub Pages or as standalone documents.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
import re

import pandas as pd

logger = logging.getLogger(__name__)


def _sanitize_for_filename(text: str) -> str:
    """
    Sanitize text for use in filenames to prevent path traversal.
    
    Args:
        text: Input text to sanitize.
        
    Returns:
        Sanitized text safe for use in filenames.
    """
    if not text:
        return "unknown"
    # Remove path separators and parent directory references
    text = text.replace("/", "_").replace("\\", "_").replace("..", "_")
    # Replace any character that is not alphanumeric, underscore, or hyphen
    text = re.sub(r"[^a-zA-Z0-9_-]", "_", text)
    # Remove leading dots and spaces
    text = text.lstrip(". ")
    # Collapse multiple underscores
    text = re.sub(r"_+", "_", text)
    # Ensure non-empty
    return text[:255] if text else "unknown"


class ReportGenerator:
    """
    Generates publication-ready reports from judicial integrity analysis.

    Report types:
    - Individual judge profile reports
    - State summary reports
    - Sentencing disparity reports
    - Methodology and data source documentation
    """

    def __init__(
        self,
        state: str = "az",
        output_dir: str = "output/reports",
    ):
        """
        Initialize the report generator.

        Args:
            state: Two-letter state code.
            output_dir: Directory to write report files.
        """
        # Sanitize state to prevent path traversal
        self.state = _sanitize_for_filename(state).upper()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_state_summary(
        self,
        judges_df: pd.DataFrame,
        integrity_summary: Optional[pd.DataFrame] = None,
        bias_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a state-level summary report.

        Args:
            judges_df: Preprocessed judge profiles.
            integrity_summary: Optional integrity summary DataFrame.
            bias_results: Optional sentencing bias analysis results.

        Returns:
            Markdown-formatted report string.
        """
        lines = [
            f"# Judicial Integrity Analysis: {self.state}",
            "",
            f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**State**: {self.state}",
            "",
            "---",
            "",
            "## Overview",
            "",
        ]

        if not judges_df.empty:
            lines.extend([
                f"- **Total Judges Analyzed**: {len(judges_df)}",
                f"- **Federal Judges**: {len(judges_df[judges_df['level'] == 'federal'])}"
                if "level" in judges_df.columns else "",
                f"- **State Judges**: {len(judges_df[judges_df['level'] == 'state'])}"
                if "level" in judges_df.columns else "",
                f"- **Active Judges**: {judges_df['is_active'].sum()}"
                if "is_active" in judges_df.columns else "",
                "",
            ])

        # Risk tier section
        if integrity_summary is not None and not integrity_summary.empty:
            lines.extend([
                "## Risk Tier Distribution",
                "",
            ])
            if "risk_tier" in integrity_summary.columns:
                tier_counts = integrity_summary["risk_tier"].value_counts()
                lines.append("| Risk Tier | Count | Percentage |")
                lines.append("|-----------|-------|------------|")
                for tier, count in tier_counts.items():
                    pct = count / len(integrity_summary) * 100
                    lines.append(f"| {tier} | {count} | {pct:.1f}% |")
                lines.append("")

            # Average scores
            score_cols = [
                ("corruption_risk_score", "Corruption Risk"),
                ("ethics_score", "Ethics"),
                ("partisanship_score", "Partisanship Indicators"),
                ("integrity_composite", "Integrity Composite"),
            ]
            lines.append("## Average Scores")
            lines.append("")
            lines.append("| Metric | Mean | Median | Std Dev |")
            lines.append("|--------|------|--------|---------|")
            for col, label in score_cols:
                if col in integrity_summary.columns:
                    s = integrity_summary[col]
                    lines.append(
                        f"| {label} | {s.mean():.1f} | {s.median():.1f} | {s.std():.1f} |"
                    )
            lines.append("")

        # Bias analysis section
        if bias_results and "groups" in bias_results:
            lines.extend([
                "## Sentencing Disparity Analysis",
                "",
                f"**Demographic Factor**: {bias_results.get('demographic_column', 'N/A')}",
                f"**Total Records**: {bias_results.get('n_records', 0)}",
                "",
            ])

            lines.append("| Group | N | Mean Sentence (mo) | Median Sentence (mo) |")
            lines.append("|-------|---|-------------------|---------------------|")
            for group_name, group_stats in bias_results["groups"].items():
                lines.append(
                    f"| {group_name} | {group_stats.get('n', 0)} | "
                    f"{group_stats.get('mean_sentence_months', 0):.1f} | "
                    f"{group_stats.get('median_sentence_months', 0):.1f} |"
                )
            lines.append("")

            if "kruskal_wallis" in bias_results:
                kw = bias_results["kruskal_wallis"]
                lines.extend([
                    "### Statistical Test (Kruskal-Wallis)",
                    f"- **H-statistic**: {kw['statistic']:.4f}",
                    f"- **p-value**: {kw['p_value']:.6f}",
                    f"- **Significant at 0.05**: {'Yes' if kw['significant_at_05'] else 'No'}",
                    "",
                ])

        # Methodology disclaimer
        lines.extend([
            "---",
            "",
            "## Methodology",
            "",
            "This analysis is based on publicly available data from:",
            "- CourtListener (free.law) for federal judge profiles and opinions",
            "- Arizona Commission on Judicial Conduct for disciplinary records",
            "- Arizona Commission on Judicial Performance Review for performance data",
            "- U.S. Sentencing Commission for sentencing statistics",
            "",
            ("**Disclaimer**: This report is for educational and transparency purposes "
             "only. Findings are based on statistical patterns in publicly available data "
             "and do not constitute legal conclusions or accusations. All analysis should "
             "be interpreted in context and verified against primary sources."),
            "",
        ])

        report = "\n".join(lines)

        # Save report
        filepath = self.output_dir / f"{self.state.lower()}_summary.md"
        with open(filepath, "w") as f:
            f.write(report)
        logger.info("Saved state summary report to %s", filepath)

        return report

    def generate_judge_report(
        self,
        judge_row: pd.Series,
        audit_result: Optional[Any] = None,
    ) -> str:
        """
        Generate an individual judge profile report.

        Args:
            judge_row: Series with judge data (from integrity summary).
            audit_result: Optional LLM AuditResult for this judge.

        Returns:
            Markdown-formatted report string.
        """
        name = judge_row.get("name_full", "Unknown Judge")
        judge_id = judge_row.get("judge_id", "unknown")

        lines = [
            f"# Judge Profile: {name}",
            "",
            f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
            "## Basic Information",
            "",
            f"- **Name**: {name}",
            f"- **Court**: {judge_row.get('court_name', 'Unknown')}",
            f"- **Level**: {judge_row.get('level', 'Unknown')}",
            f"- **Active**: {judge_row.get('is_active', 'Unknown')}",
            f"- **Party**: {judge_row.get('party', 'Unknown')}",
            f"- **Appointing Authority**: {judge_row.get('appointing_president', 'Unknown')}",
            "",
        ]

        # Integrity scores
        lines.extend([
            "## Integrity Assessment",
            "",
        ])

        score_fields = [
            ("integrity_composite", "Composite Score"),
            ("risk_tier", "Risk Tier"),
            ("corruption_risk_score", "Corruption Risk"),
            ("ethics_score", "Ethics Score"),
            ("partisanship_score", "Partisanship Indicators"),
        ]

        for field_name, label in score_fields:
            val = judge_row.get(field_name)
            if val is not None and str(val) != "nan":
                if isinstance(val, float):
                    lines.append(f"- **{label}**: {val:.1f}")
                else:
                    lines.append(f"- **{label}**: {val}")

        lines.append("")

        # Disciplinary record
        n_disc = judge_row.get("n_disciplinary", 0)
        if isinstance(n_disc, (int, float)) and n_disc > 0:
            lines.extend([
                "## Disciplinary Record",
                "",
                f"- **Actions on Record**: {int(n_disc)}",
                f"- **Severe Actions**: {int(judge_row.get('n_severe', 0))}",
                "",
            ])

        # LLM audit results
        if audit_result:
            lines.extend([
                "## LLM Audit",
                "",
                f"**Audit Type**: {getattr(audit_result, 'audit_type', 'N/A')}",
                f"**Severity**: {getattr(audit_result, 'severity', 'N/A')}",
                f"**Confidence**: {getattr(audit_result, 'confidence', 0):.0%}",
                "",
                "### Summary",
                getattr(audit_result, "summary", "No summary available."),
                "",
            ])
            concerns = getattr(audit_result, "concerns", [])
            if concerns:
                lines.append("### Concerns")
                for concern in concerns:
                    lines.append(f"- {concern}")
                lines.append("")

        # Disclaimer
        lines.extend([
            "---",
            "",
            ("*This profile is generated from publicly available data for "
             "educational and transparency purposes. It does not constitute "
             "a legal evaluation or accusation.*"),
        ])

        report = "\n".join(lines)

        # Save report with sanitized judge_id
        safe_judge_id = _sanitize_for_filename(judge_id)
        filepath = self.output_dir / f"judge_{safe_judge_id}.md"
        with open(filepath, "w") as f:
            f.write(report)
        logger.info("Saved judge report to %s", filepath)

        return report

    def generate_batch_reports(
        self,
        integrity_summary: pd.DataFrame,
        audit_results: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate reports for all judges in the summary.

        Args:
            integrity_summary: DataFrame from compute_integrity_summary().
            audit_results: Optional dict mapping judge_id to AuditResult.

        Returns:
            List of generated report file paths.
        """
        audit_results = audit_results or {}
        filepaths = []

        for _, row in integrity_summary.iterrows():
            judge_id = row.get("judge_id", "")
            audit = audit_results.get(judge_id)
            self.generate_judge_report(row, audit_result=audit)
            safe_judge_id = _sanitize_for_filename(judge_id)
            filepath = self.output_dir / f"judge_{safe_judge_id}.md"
            filepaths.append(str(filepath))

        logger.info("Generated %d judge reports", len(filepaths))
        return filepaths

    def generate_methodology_doc(self) -> str:
        """
        Generate methodology documentation.

        Returns:
            Markdown methodology document.
        """
        doc = """# Methodology: Judicial Integrity Analysis

## Purpose

This document describes the methodology used for analyzing the integrity
of state and federal judges. The analysis covers four dimensions:
corruption, ethics, bias, and partisanship.

## Data Sources

### Federal Court Data
- **CourtListener API** (free.law): Judge profiles, judicial positions,
  financial disclosures, and court opinions.
- **PACER**: Federal court dockets and case filings.

### State Court Data
- **Arizona Commission on Judicial Conduct**: Disciplinary actions and
  public complaints.
- **Arizona Commission on Judicial Performance Review**: Judicial
  performance evaluations including integrity, temperament, and legal
  ability scores.

### Sentencing Data
- **U.S. Sentencing Commission**: Federal sentencing statistics by
  district, including demographic breakdowns.
- **Arizona Judicial Branch Reports**: State-level sentencing data.

## Analysis Methods

### Corruption Risk Assessment
- Count and severity classification of disciplinary actions
- Financial disclosure availability and anomaly detection
- Composite risk score (0-100 scale)

### Ethics Evaluation
- Weighted composite of performance review scores
- Integrity, temperament, and legal ability sub-scores
- Compliance with judicial codes of conduct

### Bias Analysis
- Sentencing disparity indices across demographic groups
- Kruskal-Wallis tests for statistically significant differences
- Per-judge z-scores relative to peer sentencing patterns
- Mann-Whitney U pairwise comparisons with effect sizes

### Partisanship Indicators
- Appointing authority's political party
- Judge's known political affiliations (if public)
- Data availability score (not a bias verdict)

### LLM-Assisted Auditing
- Structured summarization of judicial opinions
- Pattern identification in conduct records
- All LLM outputs require human verification

## Scoring Methodology

### Integrity Composite Score (0-100)
- 30% weight: Inverted corruption risk score
- 40% weight: Ethics score
- 30% weight: Inverted partisanship indicator score

### Risk Tiers
- **Low Risk**: Composite >= 70
- **Moderate**: 50 <= Composite < 70
- **Elevated**: 30 <= Composite < 50
- **High Risk**: Composite < 30

## Limitations

1. Data availability varies significantly across jurisdictions
2. Absence of disciplinary records does not guarantee integrity
3. Statistical patterns may reflect systemic factors beyond individual judges
4. LLM analysis may contain errors and requires human validation
5. Sentencing disparities may be influenced by factors not captured in the data

## Ethical Considerations

- All data used is publicly available
- No unsubstantiated accusations are made
- Findings are presented as statistical patterns, not verdicts
- Reports include appropriate disclaimers
- Anonymization is applied where legally required

## References

- CourtListener API Documentation: https://www.courtlistener.com/api/rest-info/
- U.S. Sentencing Commission: https://www.ussc.gov/
- Arizona Courts: https://www.azcourts.gov/
"""

        filepath = self.output_dir / "methodology.md"
        with open(filepath, "w") as f:
            f.write(doc)
        logger.info("Saved methodology documentation to %s", filepath)

        return doc
