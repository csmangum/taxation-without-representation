# Phase 1: Arizona - Detailed Plan

## Overview

Arizona is the initial target state for judicial integrity analysis. This document outlines the specific data sources, judges covered, and analysis plan for the AZ phase.

## Judges Covered

### Federal Courts
- **District of Arizona (azd)**: All current district judges, senior judges, and magistrate judges handling criminal cases
- Source: CourtListener API

### State Courts
- **Superior Court**: All current judges in Arizona's 15 counties
- **Court of Appeals**: Division 1 (Phoenix) and Division 2 (Tucson)
- Source: Arizona Judicial Branch website, Commission on Judicial Performance Review

## Data Sources

### 1. CourtListener (Federal)
- Judge profiles and positions
- Financial disclosures
- Federal opinions
- Docket information

### 2. Arizona Commission on Judicial Conduct
- URL: https://www.azcourts.gov/cjc/
- Disciplinary actions (public records)
- Formal complaints
- Advisory opinions

### 3. Arizona Commission on Judicial Performance Review (JPR)
- URL: https://www.azcourts.gov/jpr/
- Performance evaluation reports (PDF)
- Scores: Legal ability, integrity, communication, temperament, administrative performance
- Retention election recommendations

### 4. U.S. Sentencing Commission
- District of Arizona sentencing data
- Demographic breakdowns
- Departure rates from guidelines

### 5. PACER (Federal)
- Case dockets for District of Arizona
- Note: PACER charges per-page fees; use CourtListener's free RECAP archive where possible

### 6. Media and Academic Sources
- Arizona Republic judicial coverage
- Academic studies on AZ judicial behavior
- Arizona State Law Journal

## Tasks

- [ ] Compile complete list of current AZ federal judges from CourtListener
- [ ] Compile list of AZ Superior Court judges from state website
- [ ] Implement CourtListener data collection pipeline
- [ ] Implement AZ JPR PDF scraper (pdfplumber)
- [ ] Implement AZ Judicial Conduct commission scraper
- [ ] Collect USSC sentencing data for AZ district
- [ ] Run preprocessing pipeline
- [ ] Run integrity analysis
- [ ] Run LLM audits for top-concern judges
- [ ] Generate judge profiles and state summary
- [ ] Review and validate findings
- [ ] Publish AZ report

## Timeline

- **Week 1-2**: Data collection infrastructure and initial AZ data pull
- **Week 3**: Preprocessing, analysis, and initial audits
- **Week 4**: Report generation, review, and publication

## Legal Considerations

- All data used is from public records
- PACER data usage subject to PACER terms
- CourtListener data subject to free.law terms
- No sealed records or confidential information
- Reports avoid unsubstantiated claims
- Statistical findings presented with appropriate caveats
