"""Tests for the analysis module."""

import pytest
import numpy as np
import pandas as pd

from judicial_integrity_analysis.src.analysis import (
    JudicialIntegrityAnalyzer,
    SentencingDisparityAnalyzer,
)


@pytest.fixture
def sample_judges():
    """Create sample judge profiles."""
    return pd.DataFrame({
        "judge_id": ["smith_john_azd", "doe_jane_azd", "johnson_robert_azd",
                      "williams_mary_azd", "brown_david_azd"],
        "name_full": ["John Smith", "Jane Doe", "Robert Johnson",
                       "Mary Williams", "David Brown"],
        "court_name": ["District of Arizona"] * 3 + ["Superior Court"] * 2,
        "level": ["federal", "federal", "federal", "state", "state"],
        "is_active": [True, True, False, True, True],
        "party": ["Democrat", "Republican", "Republican", "Nonpartisan", "Unknown"],
        "appointing_president": ["Obama", "Trump", "Bush", "", ""],
        "has_financial_disclosures": [True, False, True, False, False],
        "n_financial_disclosures": [3, 0, 2, 0, 0],
    })


@pytest.fixture
def sample_disciplinary():
    """Create sample disciplinary records."""
    return pd.DataFrame({
        "judge_id": ["smith_john_azd", "smith_john_azd", "doe_jane_azd"],
        "date": ["2015-01-01", "2018-06-15", "2020-03-10"],
        "action_type": ["reprimand", "censure", "warning"],
        "description": ["Conflict of interest", "Inappropriate conduct", "Late filings"],
        "source": ["AZ CJC"] * 3,
        "severity": ["moderate", "severe", "minor"],
    })


@pytest.fixture
def sample_performance():
    """Create sample performance reviews."""
    return pd.DataFrame({
        "judge_id": ["smith_john_azd", "doe_jane_azd", "johnson_robert_azd",
                      "williams_mary_azd"],
        "review_year": [2022, 2022, 2019, 2022],
        "overall_score": [72.0, 85.0, 65.0, 90.0],
        "integrity": [68.0, 88.0, 55.0, 92.0],
        "temperament": [75.0, 82.0, 60.0, 88.0],
        "communication": [78.0, 86.0, 70.0, 91.0],
        "administrative": [70.0, 84.0, 62.0, 87.0],
        "meets_standard": [True, True, False, True],
    })


@pytest.fixture
def sample_sentencing():
    """Create sample sentencing data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "judge_id": np.random.choice(
            ["smith_john_azd", "doe_jane_azd", "johnson_robert_azd"], n
        ),
        "case_id": [f"case_{i}" for i in range(n)],
        "offense_type": np.random.choice(["drug", "fraud", "violent"], n),
        "offense_level": np.random.randint(10, 30, n),
        "criminal_history": np.random.randint(1, 6, n),
        "guideline_min": np.random.randint(12, 60, n),
        "guideline_max": np.random.randint(60, 120, n),
        "sentence_months": np.random.randint(6, 100, n),
        "departure_pct": np.random.uniform(-30, 30, n),
        "defendant_race": np.random.choice(
            ["White", "Black", "Hispanic", "Asian"], n, p=[0.4, 0.3, 0.2, 0.1]
        ),
        "defendant_gender": np.random.choice(["Male", "Female"], n, p=[0.7, 0.3]),
        "defendant_age": np.random.randint(18, 65, n),
        "district": ["AZ"] * n,
        "year": np.random.choice([2021, 2022, 2023], n),
    })


@pytest.fixture
def analyzer(sample_judges, sample_disciplinary, sample_performance, sample_sentencing):
    """Create an analyzer with sample data."""
    return JudicialIntegrityAnalyzer(
        judges_df=sample_judges,
        disciplinary_df=sample_disciplinary,
        performance_df=sample_performance,
        sentencing_df=sample_sentencing,
    )


class TestJudicialIntegrityAnalyzer:
    """Test cases for JudicialIntegrityAnalyzer."""

    def test_init(self, analyzer):
        """Test analyzer initialization."""
        assert not analyzer.judges.empty
        assert not analyzer.disciplinary.empty
        assert not analyzer.performance.empty
        assert not analyzer.sentencing.empty

    def test_init_empty(self):
        """Test analyzer with no data."""
        analyzer = JudicialIntegrityAnalyzer()
        assert analyzer.judges.empty
        assert analyzer.disciplinary.empty

    # ==================== Corruption Analysis ====================

    def test_corruption_indicators(self, analyzer):
        """Test corruption indicator analysis."""
        result = analyzer.analyze_corruption_indicators()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "corruption_risk_score" in result.columns
        assert "n_disciplinary" in result.columns

        # All scores should be 0-100
        assert result["corruption_risk_score"].min() >= 0
        assert result["corruption_risk_score"].max() <= 100

    def test_corruption_scoring(self, analyzer):
        """Test that judges with more disciplinary actions get higher risk."""
        result = analyzer.analyze_corruption_indicators()

        smith = result[result["judge_id"] == "smith_john_azd"]
        brown = result[result["judge_id"] == "brown_david_azd"]

        # Smith has 2 disciplinary actions, Brown has 0
        assert smith["corruption_risk_score"].iloc[0] > brown["corruption_risk_score"].iloc[0]

    def test_corruption_empty_data(self):
        """Test corruption analysis with no data."""
        analyzer = JudicialIntegrityAnalyzer()
        result = analyzer.analyze_corruption_indicators()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    # ==================== Ethics Analysis ====================

    def test_ethics_indicators(self, analyzer):
        """Test ethics indicator analysis."""
        result = analyzer.analyze_ethics_indicators()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "ethics_score" in result.columns

        # Scores should be 0-100
        assert result["ethics_score"].min() >= 0
        assert result["ethics_score"].max() <= 100

    def test_ethics_scores_ordering(self, analyzer):
        """Test that ethics scores reflect performance reviews."""
        result = analyzer.analyze_ethics_indicators()

        doe = result[result["judge_id"] == "doe_jane_azd"]["ethics_score"].iloc[0]
        johnson = result[result["judge_id"] == "johnson_robert_azd"]["ethics_score"].iloc[0]

        # Doe has better reviews than Johnson
        assert doe > johnson

    # ==================== Bias Analysis ====================

    def test_sentencing_bias_analysis(self, analyzer):
        """Test sentencing bias analysis."""
        result = analyzer.analyze_sentencing_bias(demographic_column="defendant_race")

        assert isinstance(result, dict)
        assert "groups" in result
        assert "n_records" in result
        assert result["n_records"] == 100

    def test_sentencing_bias_by_gender(self, analyzer):
        """Test sentencing bias by gender."""
        result = analyzer.analyze_sentencing_bias(demographic_column="defendant_gender")

        assert "groups" in result
        assert "Male" in result["groups"]
        assert "Female" in result["groups"]

    def test_sentencing_bias_missing_column(self, analyzer):
        """Test sentencing bias with nonexistent column."""
        result = analyzer.analyze_sentencing_bias(
            demographic_column="nonexistent_column"
        )
        assert "error" in result

    def test_sentencing_bias_no_data(self):
        """Test sentencing bias with empty data."""
        analyzer = JudicialIntegrityAnalyzer()
        result = analyzer.analyze_sentencing_bias()
        assert "error" in result

    def test_judge_sentencing_patterns(self, analyzer):
        """Test individual judge sentencing pattern analysis."""
        result = analyzer.analyze_judge_sentencing_patterns()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 3 judges in sentencing data
        assert "sentence_zscore" in result.columns
        assert "is_outlier" in result.columns
        assert "direction" in result.columns

    # ==================== Partisanship Analysis ====================

    def test_partisanship_indicators(self, analyzer):
        """Test partisanship indicator analysis."""
        result = analyzer.analyze_partisanship_indicators()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "partisanship_score" in result.columns
        assert "appointer_party" in result.columns

    def test_appointer_party_classification(self):
        """Test appointing president party classification."""
        classify = JudicialIntegrityAnalyzer._classify_appointer_party

        assert classify("Obama") == "Democrat"
        assert classify("Trump") == "Republican"
        assert classify("Biden") == "Democrat"
        assert classify("Bush") == "Republican"
        assert classify("Clinton") == "Democrat"
        assert classify("Reagan") == "Republican"
        assert classify("") == "Unknown"
        assert classify(None) == "Unknown"

    def test_partisanship_scoring(self, analyzer):
        """Test that partisanship scores reflect available indicators."""
        result = analyzer.analyze_partisanship_indicators()

        # Federal judges with known party + president should score higher
        smith = result[result["judge_id"] == "smith_john_azd"]["partisanship_score"].iloc[0]
        brown = result[result["judge_id"] == "brown_david_azd"]["partisanship_score"].iloc[0]

        # Smith has known party and appointer; Brown has neither
        assert smith > brown

    # ==================== Composite Analysis ====================

    def test_integrity_summary(self, analyzer):
        """Test composite integrity summary."""
        result = analyzer.compute_integrity_summary()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "integrity_composite" in result.columns
        assert "risk_tier" in result.columns

        # All composites should be 0-100
        assert result["integrity_composite"].min() >= 0
        assert result["integrity_composite"].max() <= 100

    def test_risk_tier_assignment(self, analyzer):
        """Test that risk tiers are properly assigned."""
        result = analyzer.compute_integrity_summary()

        valid_tiers = {"High Risk", "Elevated", "Moderate", "Low Risk"}
        for tier in result["risk_tier"].dropna():
            assert tier in valid_tiers

    # ==================== Report Generation ====================

    def test_generate_text_report(self, analyzer):
        """Test text report generation."""
        report = analyzer.generate_text_report()

        assert isinstance(report, str)
        assert "JUDICIAL INTEGRITY ANALYSIS REPORT" in report
        assert "Risk Tier Distribution" in report

    def test_generate_judge_report(self, analyzer):
        """Test individual judge report generation."""
        report = analyzer.generate_text_report(judge_id="smith_john_azd")

        assert isinstance(report, str)
        assert "John Smith" in report or "smith_john_azd" in report

    def test_generate_report_unknown_judge(self, analyzer):
        """Test report for nonexistent judge."""
        report = analyzer.generate_text_report(judge_id="nonexistent")
        assert "No data found" in report

    def test_generate_report_no_data(self):
        """Test report with no data."""
        analyzer = JudicialIntegrityAnalyzer()
        report = analyzer.generate_text_report()
        assert "No data available" in report


class TestSentencingDisparityAnalyzer:
    """Test cases for SentencingDisparityAnalyzer."""

    @pytest.fixture
    def disparity_analyzer(self, sample_sentencing):
        """Create a disparity analyzer with sample data."""
        return SentencingDisparityAnalyzer(sample_sentencing)

    def test_init(self, disparity_analyzer):
        """Test analyzer initialization."""
        assert not disparity_analyzer.data.empty
        assert len(disparity_analyzer.data) == 100

    def test_disparity_index(self, disparity_analyzer):
        """Test disparity index computation."""
        result = disparity_analyzer.compute_disparity_index(
            group_col="defendant_race"
        )

        assert isinstance(result, pd.DataFrame)
        assert "disparity_index" in result.columns
        assert "reference_group" in result.columns

        # Reference group should have index = 1.0
        ref_group = result["reference_group"].iloc[0]
        ref_row = result[result["defendant_race"] == ref_group]
        assert abs(ref_row["disparity_index"].iloc[0] - 1.0) < 0.001

    def test_pairwise_comparisons(self, disparity_analyzer):
        """Test pairwise statistical comparisons."""
        results = disparity_analyzer.pairwise_comparisons(
            group_col="defendant_race"
        )

        assert isinstance(results, list)
        assert len(results) > 0

        for comp in results:
            assert "group_1" in comp
            assert "group_2" in comp
            assert "p_value" in comp
            assert "effect_size" in comp
            assert 0 <= comp["p_value"] <= 1

    def test_judge_demographic_interaction(self, disparity_analyzer):
        """Test judge-demographic interaction analysis."""
        result = disparity_analyzer.judge_demographic_interaction(
            min_cases=5
        )

        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "disparity_range" in result.columns

    def test_empty_data(self):
        """Test with empty sentencing data."""
        analyzer = SentencingDisparityAnalyzer(pd.DataFrame())

        assert analyzer.compute_disparity_index().empty
        assert analyzer.pairwise_comparisons() == []
        assert analyzer.judge_demographic_interaction().empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
