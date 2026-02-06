#!/usr/bin/env python3
"""
Judicial Integrity Analysis - Main Analysis Script

This script performs comprehensive judicial integrity analysis focusing on
corruption, ethics, bias, and partisanship for judges in a given state.

Usage:
    python main.py --state az                    # Analyze Arizona judges
    python main.py --state az --collect          # Collect data only
    python main.py --state az --analyze          # Analyze only (requires data)
    python main.py --state az --audit            # Run LLM audits
    python main.py --state az --report           # Generate reports only
    python main.py --state az --full             # Full pipeline
    python main.py --state az --max-judges 10    # Limit to 10 judges
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from judicial_integrity_analysis.src.data_acquisition import JudicialDataAggregator
from judicial_integrity_analysis.src.preprocessing import JudicialDataPreprocessor
from judicial_integrity_analysis.src.analysis import JudicialIntegrityAnalyzer
from judicial_integrity_analysis.src.visualization import JudicialVisualizer
from judicial_integrity_analysis.src.llm_auditor import LLMAuditor
from judicial_integrity_analysis.src.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_directories(state: str) -> dict:
    """Create necessary directories for a state analysis."""
    # Validate state to prevent path traversal
    if not state or not state.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid state code: {state}")
    if ".." in state or "/" in state or "\\" in state:
        raise ValueError(f"State code contains invalid path characters: {state}")
    
    dirs = {
        "data_raw": f"data/{state}/raw",
        "data_processed": f"data/{state}/processed",
        "data_reports": f"data/{state}/reports",
        "output_figures": f"output/figures/{state}",
        "output_reports": f"output/reports/{state}",
        "output_data": f"output/data/{state}",
    }
    for name, path in dirs.items():
        Path(path).mkdir(parents=True, exist_ok=True)
    return dirs


def collect_data(
    state: str,
    max_judges: Optional[int] = None,
    courtlistener_token: Optional[str] = None,
) -> dict:
    """
    Collect judicial data for a state.

    Args:
        state: Two-letter state code.
        max_judges: Maximum number of judges to collect.
        courtlistener_token: CourtListener API token.

    Returns:
        Dictionary of collected data.
    """
    logger.info("Collecting data for %s...", state.upper())

    aggregator = JudicialDataAggregator(
        state=state,
        data_dir="data",
        courtlistener_token=courtlistener_token,
    )

    data = aggregator.collect_all(max_judges=max_judges)
    logger.info(
        "Collected: %d federal judges, %d state judges, %d disciplinary records",
        len(data.get("federal_judges", [])),
        len(data.get("state_judges", [])),
        len(data.get("disciplinary_records", [])),
    )

    return data


def preprocess_data(raw_data: dict, state: str) -> JudicialDataPreprocessor:
    """
    Preprocess collected data.

    Args:
        raw_data: Dictionary of collected data.
        state: Two-letter state code.

    Returns:
        Preprocessor instance with processed data.
    """
    logger.info("Preprocessing data for %s...", state.upper())

    preprocessor = JudicialDataPreprocessor(raw_data)
    preprocessor.preprocess_all()
    preprocessor.save_processed_data(f"data/{state}/processed")

    if preprocessor.judges is not None:
        logger.info("Preprocessed %d judges", len(preprocessor.judges))

    return preprocessor


def run_analysis(preprocessor: JudicialDataPreprocessor) -> JudicialIntegrityAnalyzer:
    """
    Run integrity analysis.

    Args:
        preprocessor: Preprocessor with processed data.

    Returns:
        Analyzer instance with results.
    """
    logger.info("Running integrity analysis...")

    analyzer = JudicialIntegrityAnalyzer(
        judges_df=preprocessor.judges,
        disciplinary_df=preprocessor.disciplinary_records,
        performance_df=preprocessor.performance_reviews,
        sentencing_df=preprocessor.sentencing_data,
    )

    # Run all analyses
    summary = analyzer.compute_integrity_summary()

    # Print summary
    if not summary.empty:
        print(f"\n{'='*60}")
        print("JUDICIAL INTEGRITY ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Judges Analyzed: {len(summary)}")
        if "risk_tier" in summary.columns:
            tier_counts = summary["risk_tier"].value_counts()
            for tier, count in tier_counts.items():
                print(f"  {tier}: {count}")
        if "integrity_composite" in summary.columns:
            print(f"Average Integrity Score: {summary['integrity_composite'].mean():.1f}")
        print(f"{'='*60}\n")

    # Sentencing bias analysis
    if preprocessor.sentencing_data is not None and len(preprocessor.sentencing_data) > 0:
        bias = analyzer.analyze_sentencing_bias()
        if "kruskal_wallis" in bias:
            kw = bias["kruskal_wallis"]
            print(f"Sentencing Disparity Test (Kruskal-Wallis):")
            print(f"  H-statistic: {kw['statistic']:.4f}")
            print(f"  p-value: {kw['p_value']:.6f}")
            print(f"  Significant: {kw['significant_at_05']}")
            print()

    return analyzer


def run_llm_audits(
    analyzer: JudicialIntegrityAnalyzer,
    state: str,
    api_key: Optional[str] = None,
    max_judges: Optional[int] = None,
) -> LLMAuditor:
    """
    Run LLM-powered audits.

    Args:
        analyzer: Analyzer instance.
        state: State code.
        api_key: OpenRouter API key.
        max_judges: Maximum judges to audit.

    Returns:
        LLMAuditor instance with results.
    """
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        logger.warning("No LLM API key available. Skipping audits.")
        return LLMAuditor()

    logger.info("Running LLM audits...")
    auditor = LLMAuditor(api_key=api_key)

    if analyzer.judges is not None and not analyzer.judges.empty:
        auditor.batch_audit_judges(analyzer.judges, max_judges=max_judges)
        auditor.save_results(f"output/data/{state}/audit_results.json")

    return auditor


def generate_reports(
    analyzer: JudicialIntegrityAnalyzer,
    state: str,
    auditor: Optional[LLMAuditor] = None,
) -> None:
    """
    Generate analysis reports.

    Args:
        analyzer: Analyzer instance.
        state: State code.
        auditor: Optional LLMAuditor with results.
    """
    logger.info("Generating reports for %s...", state.upper())

    report_gen = ReportGenerator(
        state=state,
        output_dir=f"output/reports/{state}",
    )

    # State summary
    summary = analyzer.compute_integrity_summary()
    bias = (
        analyzer.analyze_sentencing_bias()
        if not analyzer.sentencing.empty
        else None
    )

    report_gen.generate_state_summary(
        judges_df=analyzer.judges if analyzer.judges is not None else summary,
        integrity_summary=summary,
        bias_results=bias,
    )

    # Individual judge reports
    if not summary.empty:
        audit_results = {}
        if auditor:
            for result in auditor.get_all_results():
                audit_results[result.judge_id] = result

        report_gen.generate_batch_reports(
            integrity_summary=summary,
            audit_results=audit_results,
        )

    # Methodology doc
    report_gen.generate_methodology_doc()

    logger.info("Reports saved to output/reports/%s/", state)


def create_visualizations(
    analyzer: JudicialIntegrityAnalyzer,
    state: str,
) -> None:
    """
    Create analysis visualizations.

    Args:
        analyzer: Analyzer instance.
        state: State code.
    """
    logger.info("Creating visualizations for %s...", state.upper())

    viz = JudicialVisualizer()
    output_dir = f"output/figures/{state}"

    saved = viz.create_analysis_dashboard(
        analyzer=analyzer,
        output_dir=output_dir,
        prefix=f"{state}_",
    )

    logger.info("Created %d visualizations in %s", len(saved), output_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Judicial Integrity Analysis - State-by-State Research"
    )
    parser.add_argument(
        "--state",
        type=str,
        default="az",
        help="Two-letter state code (default: az)",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect data only",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis only (requires prior data collection)",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run LLM audits (requires OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate reports only",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations only",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (collect, analyze, report, visualize)",
    )
    parser.add_argument(
        "--max-judges",
        type=int,
        default=None,
        help="Maximum number of judges to process",
    )
    parser.add_argument(
        "--courtlistener-token",
        type=str,
        default=None,
        help="CourtListener API token (or set COURTLISTENER_API_TOKEN env var)",
    )
    parser.add_argument(
        "--openrouter-key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )

    args = parser.parse_args()
    state = args.state.lower()

    # Resolve API keys from args or environment
    cl_token = args.courtlistener_token or os.environ.get("COURTLISTENER_API_TOKEN")
    or_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")

    # Default: if no specific action, run full pipeline
    run_all = args.full or not any(
        [args.collect, args.analyze, args.audit, args.report, args.visualize]
    )

    # Setup
    setup_directories(state)
    logger.info("Starting Judicial Integrity Analysis for %s", state.upper())

    # Collect
    if args.collect or run_all:
        raw_data = collect_data(
            state=state,
            max_judges=args.max_judges,
            courtlistener_token=cl_token,
        )
    else:
        raw_data = {}

    # Preprocess
    processed_dir = f"data/{state}/processed"
    if raw_data:
        preprocessor = preprocess_data(raw_data, state)
    elif Path(processed_dir).exists() and any(Path(processed_dir).iterdir()):
        logger.info("Loading previously processed data...")
        preprocessor = JudicialDataPreprocessor.load_processed_data(processed_dir)
    else:
        preprocessor = JudicialDataPreprocessor()

    # Analyze
    analyzer = None
    if args.analyze or run_all:
        analyzer = run_analysis(preprocessor)

    # Visualize
    if (args.visualize or run_all) and analyzer:
        create_visualizations(analyzer, state)

    # LLM Audit
    auditor = None
    if (args.audit or run_all) and analyzer:
        auditor = run_llm_audits(
            analyzer, state, api_key=or_key, max_judges=args.max_judges
        )

    # Reports
    if (args.report or run_all) and analyzer:
        generate_reports(analyzer, state, auditor)

    logger.info("Analysis complete for %s!", state.upper())


if __name__ == "__main__":
    main()
