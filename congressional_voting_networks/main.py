#!/usr/bin/env python3
"""
Congressional Voting Networks - Main Analysis Script

This script performs comprehensive network analysis on US Congress voting records.
It downloads data from Voteview.com, builds co-voting networks, and computes
various network metrics including centrality, community detection, and polarization.

Usage:
    python main.py                          # Analyze most recent Congress
    python main.py --congress 117           # Analyze specific Congress
    python main.py --range 110 119          # Analyze range of Congresses
    python main.py --chamber House          # Analyze only House
    python main.py --full                   # Full historical analysis
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.data_acquisition import VoteviewDataLoader
from src.preprocessing import VoteDataPreprocessor
from src.network_builder import CongressionalNetworkBuilder, build_temporal_networks
from src.analysis import NetworkAnalyzer, analyze_temporal_networks
from src.visualization import NetworkVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories."""
    dirs = [
        'data/raw',
        'data/processed',
        'output/figures',
        'output/reports',
        'output/networks'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def download_and_prepare_data(
    loader: VoteviewDataLoader,
    congress_range: Optional[Tuple[int, int]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download data if needed and load into DataFrames.
    
    Args:
        loader: VoteviewDataLoader instance.
        congress_range: Optional tuple of (start, end) Congress numbers.
        
    Returns:
        Tuple of (members, rollcalls, votes) DataFrames.
    """
    logger.info("Loading data from Voteview.com...")
    
    # Download if files don't exist
    loader.download_all_data()
    
    # Load with optional filtering
    members = loader.load_members(congress_range=congress_range)
    rollcalls = loader.load_rollcalls(congress_range=congress_range)
    votes = loader.load_votes(congress_range=congress_range)
    
    logger.info(f"Loaded {len(members)} member records, {len(rollcalls)} rollcalls, {len(votes)} votes")
    
    return members, rollcalls, votes


def analyze_single_congress(
    preprocessor: VoteDataPreprocessor,
    congress: int,
    chamber: Optional[str] = None,
    output_dir: str = 'output',
    similarity_threshold: float = 0.5
):
    """
    Analyze a single Congress.
    
    Args:
        preprocessor: VoteDataPreprocessor instance.
        congress: Congress number to analyze.
        chamber: Optional chamber filter.
        output_dir: Output directory.
        similarity_threshold: Edge threshold for network construction.
    """
    logger.info(f"Analyzing Congress {congress}...")
    
    # Create vote matrix
    matrix, leg_ids, _ = preprocessor.create_vote_matrix(
        congress=congress,
        chamber=chamber
    )
    
    if matrix is None or matrix.shape[0] < 10:
        logger.warning(f"Insufficient data for Congress {congress}")
        return
    
    leg_info = preprocessor.get_legislator_info(leg_ids)
    
    # Build network
    builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
    G = builder.build_similarity_network(
        similarity_threshold=similarity_threshold,
        method="cosine"
    )
    
    # Save network
    network_path = f"{output_dir}/networks/congress_{congress}.graphml"
    builder.save_network(network_path)
    
    # Analyze network
    analyzer = NetworkAnalyzer(G)
    
    # Generate report
    report = analyzer.generate_report()
    report_path = f"{output_dir}/reports/congress_{congress}_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Analysis Report for Congress {congress}\n")
        f.write(f"Chamber: {chamber or 'Both'}\n\n")
        f.write(report)
    logger.info(f"Saved report to {report_path}")
    
    # Get top legislators
    top_legislators = analyzer.get_top_legislators(centrality_type='eigenvector', top_n=10)
    top_path = f"{output_dir}/reports/congress_{congress}_top_legislators.csv"
    top_legislators.to_csv(top_path)
    logger.info(f"Saved top legislators to {top_path}")
    
    # Create visualizations
    viz = NetworkVisualizer(G)
    prefix = f"congress_{congress}_"
    saved_files = viz.create_analysis_dashboard(
        analyzer,
        output_dir=f"{output_dir}/figures",
        prefix=prefix
    )
    
    logger.info(f"Created {len(saved_files)} visualizations")
    
    # Print summary
    stats = analyzer.compute_network_statistics()
    print(f"\n{'='*60}")
    print(f"CONGRESS {congress} ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Legislators: {stats['n_nodes']}")
    print(f"Connections: {stats['n_edges']}")
    print(f"Density: {stats['density']:.4f}")
    print(f"Polarization Score: {analyzer.compute_polarization_score():.4f}")
    print(f"Party Assortativity: {analyzer.compute_party_assortativity():.4f}")
    print(f"{'='*60}\n")


def analyze_temporal_range(
    preprocessor: VoteDataPreprocessor,
    congress_range: Tuple[int, int],
    chamber: Optional[str] = None,
    output_dir: str = 'output',
    similarity_threshold: float = 0.5
):
    """
    Analyze a range of Congresses over time.
    
    Args:
        preprocessor: VoteDataPreprocessor instance.
        congress_range: Tuple of (start, end) Congress numbers.
        chamber: Optional chamber filter.
        output_dir: Output directory.
        similarity_threshold: Edge threshold for network construction.
    """
    logger.info(f"Analyzing Congress range {congress_range[0]}-{congress_range[1]}...")
    
    # Build temporal networks
    networks = build_temporal_networks(
        preprocessor,
        congress_range,
        similarity_threshold=similarity_threshold,
        chamber=chamber
    )
    
    if len(networks) == 0:
        logger.error("No networks could be built")
        return
    
    # Analyze temporal networks
    temporal_df = analyze_temporal_networks(networks)
    
    # Save temporal data
    temporal_path = f"{output_dir}/reports/temporal_analysis_{congress_range[0]}_{congress_range[1]}.csv"
    temporal_df.to_csv(temporal_path, index=False)
    logger.info(f"Saved temporal analysis to {temporal_path}")
    
    # Create temporal visualizations
    viz = NetworkVisualizer()
    
    # Polarization trend
    fig = viz.plot_polarization_over_time(
        temporal_df,
        metrics=['polarization', 'assortativity', 'modularity'],
        title=f"Congressional Polarization ({congress_range[0]}-{congress_range[1]})"
    )
    fig.savefig(f"{output_dir}/figures/temporal_polarization.png", dpi=150, bbox_inches='tight')
    
    # Cohesion trend
    fig = viz.plot_party_cohesion_over_time(
        temporal_df,
        title=f"Party Cohesion ({congress_range[0]}-{congress_range[1]})"
    )
    fig.savefig(f"{output_dir}/figures/temporal_cohesion.png", dpi=150, bbox_inches='tight')
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEMPORAL ANALYSIS SUMMARY ({congress_range[0]}-{congress_range[1]})")
    print(f"{'='*60}")
    print(f"Congresses Analyzed: {len(networks)}")
    print(f"\nPolarization Trend:")
    print(f"  First Congress ({congress_range[0]}): {temporal_df['polarization'].iloc[0]:.4f}")
    print(f"  Last Congress ({congress_range[1]}): {temporal_df['polarization'].iloc[-1]:.4f}")
    print(f"  Change: {temporal_df['polarization'].iloc[-1] - temporal_df['polarization'].iloc[0]:+.4f}")
    print(f"\nParty Cohesion Trend:")
    if 'dem_cohesion' in temporal_df.columns:
        print(f"  Democrat: {temporal_df['dem_cohesion'].iloc[0]:.4f} -> {temporal_df['dem_cohesion'].iloc[-1]:.4f}")
    if 'rep_cohesion' in temporal_df.columns:
        print(f"  Republican: {temporal_df['rep_cohesion'].iloc[0]:.4f} -> {temporal_df['rep_cohesion'].iloc[-1]:.4f}")
    print(f"{'='*60}\n")
    
    # Also analyze most recent Congress in detail
    most_recent = max(networks.keys())
    G = networks[most_recent]
    analyzer = NetworkAnalyzer(G)
    viz = NetworkVisualizer(G)
    
    prefix = f"congress_{most_recent}_"
    viz.create_analysis_dashboard(
        analyzer,
        temporal_data=temporal_df,
        output_dir=f"{output_dir}/figures",
        prefix=prefix
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze US Congressional Voting Networks"
    )
    parser.add_argument(
        '--congress', type=int, default=None,
        help='Specific Congress number to analyze'
    )
    parser.add_argument(
        '--range', type=int, nargs=2, default=None,
        metavar=('START', 'END'),
        help='Range of Congresses to analyze (e.g., --range 110 119)'
    )
    parser.add_argument(
        '--chamber', type=str, choices=['House', 'Senate'], default=None,
        help='Filter by chamber'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Similarity threshold for network edges (default: 0.5)'
    )
    parser.add_argument(
        '--output', type=str, default='output',
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Run full historical analysis (1st to current Congress)'
    )
    parser.add_argument(
        '--download-only', action='store_true',
        help='Only download data, do not analyze'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    loader = VoteviewDataLoader('data/raw')
    
    # Download only mode
    if args.download_only:
        logger.info("Downloading data only...")
        loader.download_all_data(force=True)
        logger.info("Data download complete")
        return
    
    # Determine analysis scope
    if args.full:
        congress_range = (1, 119)  # Full history
    elif args.range:
        congress_range = tuple(args.range)
    elif args.congress:
        congress_range = (args.congress, args.congress)
    else:
        # Default: analyze recent Congress
        congress_range = (118, 118)
    
    # Load and prepare data
    members, rollcalls, votes = download_and_prepare_data(loader, congress_range)
    
    # Create preprocessor
    preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
    preprocessor.preprocess_all()
    
    # Save processed data
    preprocessor.save_processed_data('data/processed')
    
    # Run analysis
    if congress_range[0] == congress_range[1]:
        # Single Congress analysis
        analyze_single_congress(
            preprocessor,
            congress=congress_range[0],
            chamber=args.chamber,
            output_dir=args.output,
            similarity_threshold=args.threshold
        )
    else:
        # Temporal analysis
        analyze_temporal_range(
            preprocessor,
            congress_range=congress_range,
            chamber=args.chamber,
            output_dir=args.output,
            similarity_threshold=args.threshold
        )
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
