# Next Steps for Congressional Voting Networks Analysis

## Issue: Roadmap for v1.1 and Beyond

### Overview

The initial implementation of the Congressional Voting Networks Analysis package is complete with core functionality for data acquisition, preprocessing, network construction, analysis, and visualization. This issue outlines the next steps for enhancing the project.

---

## Phase 1: Validation and Testing (Priority: High)

### 1.1 Integration Testing with Real Data
- [ ] Run full analysis on 118th Congress data to validate pipeline
- [ ] Compare polarization metrics against published academic research (e.g., Voteview's own NOMINATE analysis)
- [ ] Validate that detected communities align with known party coalitions
- [ ] Benchmark against Fowler (2006) "Connecting the Congress" results for historical validation

### 1.2 Edge Case Handling
- [ ] Test with early Congresses (1st-10th) where data is sparse
- [ ] Handle legislators who switch parties mid-Congress
- [ ] Address special elections and mid-term replacements
- [ ] Test with Congresses that have significant third-party presence

### 1.3 Data Quality Checks
- [ ] Add validation for downloaded CSV integrity (checksums)
- [ ] Implement data freshness checks (compare against Voteview update timestamps)
- [ ] Add warnings for missing or incomplete vote records
- [ ] Create data quality report as part of preprocessing

---

## Phase 2: Performance Optimization (Priority: High)

### 2.1 Scalability for Full Historical Analysis
- [ ] Profile memory usage for full dataset (~10M+ votes)
- [ ] Implement chunked processing for similarity matrix computation
- [ ] Add option for approximate similarity using random projections (LSH)
- [ ] Consider using Dask for out-of-core computation on large datasets

### 2.2 Caching and Incremental Updates
- [ ] Cache computed similarity matrices to disk (HDF5 or pickle)
- [ ] Implement incremental network updates when new votes are added
- [ ] Add progress bars for long-running operations
- [ ] Optimize vote matrix creation with vectorized operations

### 2.3 Database Backend (Optional)
- [ ] Add SQLite backend for querying large datasets
- [ ] Consider MongoDB integration (Voteview provides MongoDB dumps)
- [ ] Implement lazy loading for legislator metadata

---

## Phase 3: Extended Data Sources (Priority: Medium)

### 3.1 Co-Sponsorship Networks
- [ ] Integrate bill co-sponsorship data from Congress.gov or GovTrack
- [ ] Build co-sponsorship networks as alternative/complement to voting networks
- [ ] Compare centrality rankings between voting and co-sponsorship networks

### 3.2 Additional Metadata
- [ ] Add committee membership data
- [ ] Integrate campaign finance data (OpenSecrets API)
- [ ] Add geographic/demographic state data for regional analysis
- [ ] Include bill text/topics for issue-based network analysis

### 3.3 ProPublica API Integration
- [ ] Add ProPublica Congress API client for recent data (1989-present)
- [ ] Fetch bill details and vote explanations
- [ ] Enable near-real-time analysis of current Congress

---

## Phase 4: Advanced Analysis Methods (Priority: Medium)

### 4.1 Machine Learning Integration
- [ ] Implement node embeddings using Node2Vec or GraphSAGE
- [ ] Add vote prediction based on network position
- [ ] Cluster legislators using embedding similarity
- [ ] Explore GNN-based approaches for temporal prediction

### 4.2 Temporal Network Analysis
- [ ] Implement sliding window networks for smooth temporal evolution
- [ ] Add change point detection for polarization shifts
- [ ] Create animated visualizations of network evolution
- [ ] Analyze legislator trajectory through ideological space over career

### 4.3 Bipartite Projections
- [ ] Implement weighted bipartite projections (legislators ↔ bills)
- [ ] Analyze bill similarity networks
- [ ] Identify "bridge" bills that attract bipartisan support

### 4.4 Statistical Testing
- [ ] Add bootstrap confidence intervals for network metrics
- [ ] Implement permutation tests for significance of polarization changes
- [ ] Compare observed networks against null models (random graphs)

---

## Phase 5: Interactive Dashboard (Priority: Medium)

### 5.1 Streamlit Application
- [ ] Create interactive web dashboard for exploring networks
- [ ] Add Congress/chamber/date range selectors
- [ ] Enable dynamic filtering by party, state, or committee
- [ ] Implement legislator search and profile views

### 5.2 Interactive Visualizations
- [ ] Add Plotly-based interactive network exploration
- [ ] Implement zoom/pan/hover for large networks
- [ ] Create interactive time slider for temporal analysis
- [ ] Add exportable reports (PDF/HTML)

### 5.3 Comparison Tools
- [ ] Side-by-side Congress comparison view
- [ ] Chamber comparison (House vs. Senate) dashboard
- [ ] Historical vs. current polarization comparisons

---

## Phase 6: Documentation and DevOps (Priority: Low-Medium)

### 6.1 Documentation
- [ ] Add API documentation with Sphinx
- [ ] Create tutorial notebooks for common use cases
- [ ] Document data dictionary for all fields
- [ ] Add academic citation guide

### 6.2 CI/CD Pipeline
- [ ] Set up GitHub Actions for automated testing
- [ ] Add code coverage reporting
- [ ] Implement linting with ruff or flake8
- [ ] Add type checking with mypy

### 6.3 Package Distribution
- [ ] Publish to PyPI for easy installation
- [ ] Create Docker container for reproducible analysis
- [ ] Add conda-forge recipe

---

## Phase 7: Research Extensions (Priority: Low)

### 7.1 Academic Applications
- [ ] Replicate key findings from published network analysis papers
- [ ] Create methodology for comparing across political systems (other countries)
- [ ] Add export formats for Gephi and other network tools

### 7.2 Predictive Models
- [ ] Predict vote outcomes based on network structure
- [ ] Identify legislators likely to defect from party line
- [ ] Model coalition formation dynamics

### 7.3 Natural Language Integration
- [ ] Analyze correlation between speech patterns and voting networks
- [ ] Use Congressional Record for topic modeling
- [ ] Link network position to rhetoric/messaging

---

## Immediate Next Steps (Sprint 1)

1. **Run validation analysis** on 117th and 118th Congress
2. **Document findings** comparing to known polarization trends
3. **Optimize memory** for handling votes.csv (~10M rows)
4. **Create Streamlit prototype** for interactive exploration
5. **Set up GitHub Actions** for CI/CD

---

## Success Metrics

- [ ] Full historical analysis completes in <1 hour on standard hardware
- [ ] Polarization trend matches Voteview/academic consensus
- [ ] 90%+ test coverage maintained
- [ ] Interactive dashboard deployed and accessible

---

## References

- Lewis et al. (2024). Voteview: Congressional Roll-Call Votes Database
- Fowler, J. H. (2006). Connecting the Congress: A Study of Cosponsorship Networks
- Poole & Rosenthal (1997). Congress: A Political-Economic History of Roll Call Voting
- Moody & Mucha (2013). Portrait of Political Party Polarization

---

## Labels

`enhancement` `documentation` `performance` `help wanted`

## Assignees

_Unassigned - contributions welcome!_
