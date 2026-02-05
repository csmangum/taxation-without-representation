"""
Judicial Integrity Analysis Package.

This package provides tools for analyzing the integrity of state and federal
judges across the United States, focusing on corruption, ethical concerns,
bias, and partisanship in their rulings.
"""

from .data_acquisition import CourtListenerClient, JudicialDataAggregator
from .preprocessing import JudicialDataPreprocessor
from .analysis import JudicialIntegrityAnalyzer, SentencingDisparityAnalyzer
from .visualization import JudicialVisualizer
from .llm_auditor import LLMAuditor
from .report_generator import ReportGenerator

__version__ = "0.1.0"
__all__ = [
    "CourtListenerClient",
    "JudicialDataAggregator",
    "JudicialDataPreprocessor",
    "JudicialIntegrityAnalyzer",
    "SentencingDisparityAnalyzer",
    "JudicialVisualizer",
    "LLMAuditor",
    "ReportGenerator",
]
