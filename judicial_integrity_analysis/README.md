# Judicial Integrity Analysis

A comprehensive Python package for analyzing the integrity of state and federal judges across the United States. This project focuses on identifying patterns of potential corruption, ethical concerns, bias, and partisanship using publicly available data, automated analysis tools, and LLM-assisted auditing.

This project complements the [Congressional Voting Networks](/congressional_voting_networks/) research, extending transparency analysis from the legislative branch to the judiciary.

## Research Scope

The analysis proceeds state by state, beginning with **Arizona (AZ)** as Phase 1. For each state, the project examines:

- **Corruption**: Disciplinary actions, financial conflicts of interest, conduct commission records
- **Ethics**: Performance review scores, code of conduct compliance, temperament assessments
- **Bias**: Sentencing disparity analysis across demographic groups, statistical significance testing
- **Partisanship**: Appointment backgrounds, political affiliations, ruling pattern correlations

## Features

- **Multi-Source Data Acquisition**: Automated collection from CourtListener API, state judicial conduct commissions, and sentencing databases
- **Structured Preprocessing**: Normalizes judge profiles, disciplinary records, performance reviews, and sentencing data from heterogeneous sources
- **Four-Dimensional Analysis**: Corruption risk scoring, ethics evaluation, sentencing bias detection, and partisanship indicator assessment
- **LLM-Assisted Auditing**: Uses OpenRouter (or compatible) LLMs to summarize opinions, identify patterns, and generate structured audit reports with human oversight
- **Visualization Dashboard**: Risk tier distributions, sentencing disparity plots, judge comparisons, performance heatmaps
- **Report Generation**: Publication-ready Markdown reports for individual judges and state summaries
- **State-by-State Expansion**: Config-driven architecture for easy addition of new states

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd judicial_integrity_analysis
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

4. (Optional) Set up API keys:
```bash
export COURTLISTENER_API_TOKEN="your_token_here"
export OPENROUTER_API_KEY="your_key_here"
```

## Quick Start

### Command Line Usage

```bash
# Full analysis pipeline for Arizona (default state)
python main.py --state az

# Collect data only
python main.py --state az --collect

# Run analysis on previously collected data
python main.py --state az --analyze

# Run LLM audits (requires OPENROUTER_API_KEY)
python main.py --state az --audit

# Generate reports only
python main.py --state az --report

# Create visualizations
python main.py --state az --visualize

# Limit to first 10 judges (for testing)
python main.py --state az --full --max-judges 10
```

### Python API Usage

```python
from src.data_acquisition import JudicialDataAggregator
from src.preprocessing import JudicialDataPreprocessor
from src.analysis import JudicialIntegrityAnalyzer
from src.visualization import JudicialVisualizer
from src.report_generator import ReportGenerator

# Collect data for Arizona
aggregator = JudicialDataAggregator(state="az")
raw_data = aggregator.collect_all(max_judges=50)

# Preprocess
preprocessor = JudicialDataPreprocessor(raw_data)
preprocessor.preprocess_all()

# Analyze
analyzer = JudicialIntegrityAnalyzer(
    judges_df=preprocessor.judges,
    disciplinary_df=preprocessor.disciplinary_records,
    performance_df=preprocessor.performance_reviews,
    sentencing_df=preprocessor.sentencing_data,
)
summary = analyzer.compute_integrity_summary()
print(analyzer.generate_text_report())

# Visualize
viz = JudicialVisualizer()
viz.create_analysis_dashboard(analyzer, output_dir="output/figures/az")

# Generate reports
reporter = ReportGenerator(state="az")
reporter.generate_state_summary(preprocessor.judges, summary)
```

## Project Structure

```
judicial_integrity_analysis/
├── configs/
│   ├── default_config.yaml      # Default analysis configuration
│   └── az_config.yaml           # Arizona-specific configuration
├── data/
│   ├── az/
│   │   ├── raw/                 # Downloaded data from APIs/scrapers
│   │   ├── processed/           # Preprocessed parquet files
│   │   └── reports/             # Generated reports for AZ
│   └── templates/               # Report templates
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py      # API clients and data scrapers
│   ├── preprocessing.py         # Data cleaning and normalization
│   ├── analysis.py              # Integrity analysis methods
│   ├── visualization.py         # Plotting utilities
│   ├── llm_auditor.py           # LLM-powered auditing
│   └── report_generator.py      # Report generation
├── tests/
│   ├── test_preprocessing.py
│   ├── test_analysis.py
│   └── test_data_acquisition.py
├── notebooks/
│   └── analysis_example.ipynb   # Interactive analysis walkthrough
├── output/
│   ├── figures/                 # Generated visualizations
│   ├── reports/                 # Published reports
│   └── data/                    # Exported analysis data
├── docs/
│   └── PHASE1_AZ_PLAN.md       # Arizona phase plan
├── main.py                      # Main analysis script
├── requirements.txt
├── setup.py
└── README.md
```

## Data Sources

### CourtListener (free.law)
- Federal judge profiles, positions, and financial disclosures
- Court opinions and dockets
- API: https://www.courtlistener.com/api/rest-info/

### Arizona State Sources
- Arizona Commission on Judicial Conduct: https://www.azcourts.gov/cjc/
- Arizona Commission on Judicial Performance Review: https://www.azcourts.gov/jpr/
- Arizona Judicial Branch: https://www.azcourts.gov/

### U.S. Sentencing Commission
- Federal sentencing statistics by district
- Demographic breakdowns of sentencing outcomes
- https://www.ussc.gov/

## Analysis Methods

### Integrity Composite Score (0-100)

| Component | Weight | Description |
|-----------|--------|-------------|
| Corruption Risk (inverted) | 30% | Disciplinary actions, financial conflicts |
| Ethics Score | 40% | Performance reviews, integrity ratings |
| Partisanship (inverted) | 30% | Appointment indicators, affiliation data |

### Risk Tiers

| Tier | Score Range | Description |
|------|-------------|-------------|
| Low Risk | >= 70 | No significant concerns identified |
| Moderate | 50 - 69 | Minor concerns, limited data |
| Elevated | 30 - 49 | Notable indicators present |
| High Risk | < 30 | Multiple serious indicators |

### Sentencing Bias Detection

- Kruskal-Wallis tests for group-level differences
- Mann-Whitney U pairwise comparisons with effect sizes
- Per-judge z-scores relative to peer sentencing patterns
- Disparity indices with configurable reference groups

## Running Tests

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Phase Plan

### Phase 1: Arizona (Current)
- Compile complete list of AZ judges (federal and state, criminal focus)
- Build data collection pipeline for AZ sources
- Run initial integrity analysis and audits
- Publish AZ findings

### Phase 2: California / Nevada
- Expand data acquisition to CA and NV judicial systems
- Adapt scraping for state-specific conduct commissions
- Cross-state comparison methodology

### Phase 3+: Nationwide Expansion
- Priority based on data availability and user interest
- Standardized pipeline for rapid state onboarding

## Ethical Considerations

- All data is sourced from publicly available records
- Analysis focuses on factual patterns from verifiable sources
- No unsubstantiated accusations are made
- Findings are presented as statistical patterns, not legal conclusions
- Reports include appropriate disclaimers and methodology documentation
- LLM outputs are flagged as requiring human verification
- Anonymization is applied where legally required

## Dependencies

- **API Keys**: CourtListener (optional, for higher rate limits), OpenRouter (for LLM auditing)
- **Core Libraries**: pandas, numpy, scipy, requests, matplotlib, seaborn
- **Scraping**: beautifulsoup4, pdfplumber
- **LLM**: openai SDK (for OpenRouter-compatible API calls)

## License

This project is provided for educational and research purposes. Data from public sources is used under their respective terms of service.

## Contributing

Contributions are welcome! Areas of particular interest:
- State-specific data acquisition adapters
- Additional statistical analysis methods
- Legal/ethical review of methodology
- Report template improvements
