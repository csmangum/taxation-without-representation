"""
Congressional Voting Networks Analysis Package.

This package provides tools for network analysis of US Congress voting records
from 1789 to present, using data from Voteview.com.
"""

from .data_acquisition import VoteviewDataLoader
from .preprocessing import VoteDataPreprocessor
from .network_builder import CongressionalNetworkBuilder
from .analysis import NetworkAnalyzer
from .visualization import NetworkVisualizer

__version__ = "1.0.0"
__all__ = [
    "VoteviewDataLoader",
    "VoteDataPreprocessor",
    "CongressionalNetworkBuilder",
    "NetworkAnalyzer",
    "NetworkVisualizer",
]
