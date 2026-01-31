# Congressional Voting Networks Analysis

A comprehensive Python package for network analysis of US Congress voting records from 1789 to present. This tool models relationships between legislators using graph theory, enabling insights into bipartisanship, party cohesion, ideological shifts, and the temporal evolution of congressional dynamics.

## Features

- **Data Acquisition**: Automated download of voting data from [Voteview.com](https://voteview.com) (UCLA Political Science)
- **Network Construction**: Build co-voting similarity networks, bipartite graphs, and party-aggregated networks
- **Centrality Analysis**: Identify influential legislators using degree, betweenness, eigenvector, closeness, and PageRank centrality
- **Community Detection**: Discover voting coalitions using Louvain modularity optimization
- **Polarization Metrics**: Quantify partisan division with assortativity, cross-party edge ratios, and party cohesion scores
- **Temporal Analysis**: Track changes in network metrics across multiple Congresses
- **Visualization**: Static (matplotlib) and interactive (plotly) network visualizations

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd congressional_voting_networks
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

```bash
# Analyze the most recent Congress (default: 118th)
python main.py

# Analyze a specific Congress
python main.py --congress 117

# Analyze a range of Congresses (temporal analysis)
python main.py --range 110 118

# Analyze only the House or Senate
python main.py --congress 118 --chamber House

# Download data only (no analysis)
python main.py --download-only

# Full historical analysis (all Congresses)
python main.py --full
```

### Python API Usage

```python
from src.data_acquisition import VoteviewDataLoader
from src.preprocessing import VoteDataPreprocessor
from src.network_builder import CongressionalNetworkBuilder
from src.analysis import NetworkAnalyzer
from src.visualization import NetworkVisualizer

# Load data
loader = VoteviewDataLoader('data/raw')
loader.download_all_data()

members = loader.load_members(congress_range=(117, 118))
rollcalls = loader.load_rollcalls(congress_range=(117, 118))
votes = loader.load_votes(congress_range=(117, 118))

# Preprocess
preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
preprocessor.preprocess_all()

# Create vote matrix
matrix, leg_ids, _ = preprocessor.create_vote_matrix(congress=118)
leg_info = preprocessor.get_legislator_info(leg_ids)

# Build network
builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
G = builder.build_similarity_network(similarity_threshold=0.6)

# Analyze
analyzer = NetworkAnalyzer(G)
print(analyzer.generate_report())

# Visualize
viz = NetworkVisualizer(G)
viz.plot_network(color_by='party', title='118th Congress Voting Network')
```

### Jupyter Notebook

See `notebooks/analysis_example.ipynb` for an interactive walkthrough of the analysis pipeline.

## Project Structure

```
congressional_voting_networks/
├── data/
│   ├── raw/              # Downloaded CSV files from Voteview
│   └── processed/        # Preprocessed parquet files
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py   # Data download utilities
│   ├── preprocessing.py      # Data cleaning and transformation
│   ├── network_builder.py    # Network construction
│   ├── analysis.py           # Network analysis methods
│   └── visualization.py      # Plotting utilities
├── tests/
│   ├── test_preprocessing.py
│   ├── test_network_builder.py
│   └── test_analysis.py
├── notebooks/
│   └── analysis_example.ipynb
├── output/
│   ├── figures/          # Generated visualizations
│   ├── reports/          # Analysis reports
│   └── networks/         # Saved network files
├── main.py               # Main analysis script
├── requirements.txt
└── README.md
```

## Data Sources

### Primary Dataset: Voteview.com

- **Coverage**: 1st Congress (1789) through the most recently available Congress (currently the 119th)
- **Files**:
  - `HSall_members.csv`: Legislator metadata (ICPSR ID, party, state, DW-NOMINATE scores)
  - `HSall_rollcalls.csv`: Vote metadata (date, bill, outcome)
  - `HSall_votes.csv`: Individual vote records (~10M+ records)

### DW-NOMINATE Scores

The data includes DW-NOMINATE ideology scores (dimensions 1 and 2), which provide established measures of legislator ideology based on voting patterns.

## Analysis Methods

### Network Construction

The primary network type is a **co-voting similarity network** where:
- **Nodes** represent legislators (with attributes: party, state, chamber, DW-NOMINATE scores)
- **Edges** represent voting agreement (weighted by cosine similarity of vote vectors)
- **Threshold** parameter controls edge density (default: 0.5)

### Centrality Measures

| Measure | Interpretation |
|---------|---------------|
| Degree | Number of similar-voting connections |
| Betweenness | Bridging role between different groups |
| Eigenvector | Influence within highly-connected clusters |
| Closeness | Average distance to all other legislators |
| PageRank | Importance based on connection importance |

### Polarization Metrics

| Metric | Range | Interpretation |
|--------|-------|---------------|
| Party Assortativity | -1 to 1 | Tendency to connect within party |
| Cross-Party Edge Ratio | 0 to 1 | Proportion of bipartisan connections |
| Party Cohesion | 0 to 1 | Average similarity within party |
| Polarization Score | 0 to 1 | Combined polarization measure |

### Community Detection

Uses the Louvain algorithm to detect communities based on network modularity. Results are compared to party affiliations using Normalized Mutual Information (NMI).

## Output Files

### Reports

- `congress_X_report.txt`: Full analysis report for Congress X
- `congress_X_top_legislators.csv`: Top legislators by centrality
- `temporal_analysis_X_Y.csv`: Metrics across Congress range X to Y

### Figures

- `network.png`: Spring layout network visualization
- `network_party_split.png`: Party-separated layout
- `communities.png`: Community-colored network
- `nominate_positions.png`: DW-NOMINATE scatter plot
- `centrality_*.png`: Centrality distributions
- `temporal_polarization.png`: Polarization over time
- `temporal_cohesion.png`: Party cohesion over time

### Networks

- `congress_X.graphml`: NetworkX-compatible graph file

## Running Tests

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies including test requirements
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Performance Notes

- Full historical analysis covers ~120,000 roll-call votes and ~12,000 legislators
- The votes dataset is ~10M+ rows; use `congress_range` parameter for memory efficiency
- Sparse matrix representations are used for efficient similarity computation
- Expected runtime: <1 hour for full processing on a standard machine (16GB RAM)

## References

- **Voteview**: Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and Luke Sonnet. 2024. Voteview: Congressional Roll-Call Votes Database. https://voteview.com/
- **DW-NOMINATE**: Poole, Keith T., and Howard Rosenthal. 1997. Congress: A Political-Economic History of Roll Call Voting. Oxford University Press.
- **Louvain Algorithm**: Blondel, Vincent D., et al. "Fast unfolding of communities in large networks." Journal of Statistical Mechanics (2008).

## License

This project is provided for educational and research purposes. Data from Voteview.com is used under their terms of service.

## Contributing

Contributions are welcome! Please submit pull requests for bug fixes, new features, or documentation improvements.
