"""Tests for the preprocessing module."""

import pytest
import pandas as pd

from judicial_integrity_analysis.src.preprocessing import JudicialDataPreprocessor


@pytest.fixture
def sample_raw_data():
    """Create sample raw data mimicking JudicialDataAggregator output."""
    federal_judges = [
        {
            "id": 101,
            "name_first": "John",
            "name_middle": "A",
            "name_last": "Smith",
            "name_suffix": "",
            "court": "azd",
            "court_name": "District of Arizona",
            "date_nominated": "2010-03-15",
            "date_dob": "1960-05-20",
            "date_terminated": None,
            "appointing_president": "Obama",
            "political_affiliation": "Democrat",
            "appointment_method": "presidential",
            "gender": "Male",
            "race": "White",
            "financial_disclosures": [{"year": 2022}],
        },
        {
            "id": 102,
            "name_first": "Jane",
            "name_middle": "",
            "name_last": "Doe",
            "name_suffix": "Jr.",
            "court": "azd",
            "court_name": "District of Arizona",
            "date_nominated": "2018-06-01",
            "date_dob": "1965-11-10",
            "date_terminated": None,
            "appointing_president": "Trump",
            "political_affiliation": "Republican",
            "appointment_method": "presidential",
            "gender": "Female",
            "race": "White",
            "financial_disclosures": [],
        },
        {
            "id": 103,
            "name_first": "Robert",
            "name_middle": "B",
            "name_last": "Johnson",
            "name_suffix": "",
            "court": "azd",
            "court_name": "District of Arizona",
            "date_nominated": "2005-01-10",
            "date_dob": "1955-08-22",
            "date_terminated": "2020-12-31",
            "appointing_president": "Bush",
            "political_affiliation": "rep",
            "appointment_method": "presidential",
            "gender": "Male",
            "race": "Black",
            "financial_disclosures": [{"year": 2019}, {"year": 2020}],
        },
    ]

    disciplinary_records = [
        {
            "judge_id": "smith_john_azd",
            "date": "2015-06-15",
            "action_type": "reprimand",
            "description": "Failure to recuse in a conflict of interest case.",
            "source": "AZ Commission on Judicial Conduct",
        },
        {
            "judge_id": "doe_jane_azd",
            "date": "2020-01-10",
            "action_type": "warning",
            "description": "Inappropriate comments during proceedings.",
            "source": "AZ Commission on Judicial Conduct",
        },
    ]

    performance_reviews = [
        {
            "judge_id": "smith_john_azd",
            "review_year": 2022,
            "overall_score": 78.5,
            "legal_ability": 82.0,
            "integrity": 75.0,
            "temperament": 70.0,
            "communication": 80.0,
            "administrative": 76.0,
            "meets_standard": True,
            "source": "AZ JPR",
        },
        {
            "judge_id": "doe_jane_azd",
            "review_year": 2022,
            "overall_score": 85.0,
            "legal_ability": 88.0,
            "integrity": 90.0,
            "temperament": 82.0,
            "communication": 86.0,
            "administrative": 84.0,
            "meets_standard": True,
            "source": "AZ JPR",
        },
    ]

    return {
        "federal_judges": federal_judges,
        "state_judges": [],
        "disciplinary_records": disciplinary_records,
        "performance_reviews": performance_reviews,
        "collection_timestamp": "2025-01-01T00:00:00",
        "state": "az",
    }


@pytest.fixture
def preprocessor(sample_raw_data):
    """Create a preprocessor with sample data."""
    return JudicialDataPreprocessor(sample_raw_data)


class TestJudicialDataPreprocessor:
    """Test cases for JudicialDataPreprocessor."""

    def test_init(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.raw_data is not None
        assert preprocessor.judges is None  # Not yet processed

    def test_preprocess_judges(self, preprocessor):
        """Test judge preprocessing."""
        result = preprocessor.preprocess_judges()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "judge_id" in result.columns
        assert "name_full" in result.columns
        assert "party" in result.columns
        assert "court_level" in result.columns
        assert "years_serving" in result.columns

    def test_judge_name_normalization(self, preprocessor):
        """Test that judge names are normalized."""
        result = preprocessor.preprocess_judges()

        names = result["name_full"].tolist()
        # Names should be title case and cleaned
        for name in names:
            assert name == name.strip()
            assert "  " not in name  # No double spaces

    def test_party_normalization(self, preprocessor):
        """Test party affiliation normalization."""
        result = preprocessor.preprocess_judges()

        valid_parties = {"Democrat", "Republican", "Independent", "Libertarian",
                         "Green", "Nonpartisan", "Unknown"}
        for party in result["party"]:
            assert party in valid_parties

    def test_party_normalization_mapping(self, preprocessor):
        """Test specific party normalization mappings."""
        assert preprocessor._normalize_party("Democrat") == "Democrat"
        assert preprocessor._normalize_party("republican") == "Republican"
        assert preprocessor._normalize_party("rep") == "Republican"
        assert preprocessor._normalize_party("dem") == "Democrat"
        assert preprocessor._normalize_party("d") == "Democrat"
        assert preprocessor._normalize_party("r") == "Republican"
        assert preprocessor._normalize_party("independent") == "Independent"
        assert preprocessor._normalize_party("") == "Unknown"
        assert preprocessor._normalize_party(None) == "Unknown"

    def test_court_level_classification(self, preprocessor):
        """Test court level classification from court name."""
        assert preprocessor._classify_court_level("District of Arizona") == 2
        assert preprocessor._classify_court_level("Arizona Supreme Court") == 4
        assert preprocessor._classify_court_level("Court of Appeals") == 3
        assert preprocessor._classify_court_level("Superior Court") == 2
        assert preprocessor._classify_court_level("Magistrate Court") == 1
        assert preprocessor._classify_court_level("") == -1

    def test_judge_active_status(self, preprocessor):
        """Test that active status is correctly determined."""
        result = preprocessor.preprocess_judges()

        # Smith and Doe should be active (no termination date)
        active_judges = result[result["is_active"] == True]
        inactive_judges = result[result["is_active"] == False]

        assert len(active_judges) == 2
        assert len(inactive_judges) == 1

    def test_preprocess_disciplinary_records(self, preprocessor):
        """Test disciplinary record preprocessing."""
        result = preprocessor.preprocess_disciplinary_records()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "severity" in result.columns

    def test_severity_classification(self, preprocessor):
        """Test disciplinary severity classification."""
        assert preprocessor._classify_severity("reprimand") == "moderate"
        assert preprocessor._classify_severity("warning") == "minor"
        assert preprocessor._classify_severity("suspension") == "severe"
        assert preprocessor._classify_severity("removal") == "critical"
        assert preprocessor._classify_severity("censure") == "severe"
        assert preprocessor._classify_severity("unknown action") == "unknown"
        assert preprocessor._classify_severity(None) == "unknown"

    def test_preprocess_performance_reviews(self, preprocessor):
        """Test performance review preprocessing."""
        result = preprocessor.preprocess_performance_reviews()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "overall_score" in result.columns
        assert "integrity" in result.columns

    def test_preprocess_empty_performance_reviews(self):
        """Test preprocessing with no performance review data."""
        preprocessor = JudicialDataPreprocessor({})
        result = preprocessor.preprocess_performance_reviews()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "overall_score" in result.columns

    def test_preprocess_sentencing_data(self, preprocessor):
        """Test sentencing data preprocessing."""
        sent_df = pd.DataFrame({
            "judge_id": ["j1", "j1", "j2"],
            "case_id": ["c1", "c2", "c3"],
            "offense_type": ["drug", "fraud", "drug"],
            "offense_level": [20, 15, 22],
            "criminal_history": [3, 1, 4],
            "guideline_min": [30, 12, 36],
            "guideline_max": [60, 24, 72],
            "sentence_months": [36, 18, 48],
            "departure_type": ["none", "below", "above"],
            "departure_reason": ["", "cooperation", "criminal history"],
            "defendant_race": ["White", "Black", "Hispanic"],
            "defendant_gender": ["Male", "Female", "Male"],
            "defendant_age": [35, 42, 28],
            "district": ["AZ", "AZ", "AZ"],
            "year": [2022, 2022, 2023],
        })
        result = preprocessor.preprocess_sentencing_data(sent_df)

        assert len(result) == 3
        assert "departure_months" in result.columns
        assert "departure_pct" in result.columns

    def test_preprocess_all(self, preprocessor):
        """Test full preprocessing pipeline."""
        result = preprocessor.preprocess_all()

        assert "judges" in result
        assert "disciplinary_records" in result
        assert "performance_reviews" in result
        assert "sentencing_data" in result

    def test_link_records(self, preprocessor):
        """Test cross-source record linkage."""
        preprocessor.preprocess_all()
        linked = preprocessor.link_records()

        assert isinstance(linked, pd.DataFrame)
        assert len(linked) == 3  # 3 judges
        assert "n_disciplinary_actions" in linked.columns or "n_reviews" in linked.columns

    def test_normalize_name_static(self):
        """Test static name normalization."""
        assert JudicialDataPreprocessor._normalize_name("john  a  smith") == "John A Smith"
        assert JudicialDataPreprocessor._normalize_name("") == ""
        assert JudicialDataPreprocessor._normalize_name(None) == ""

    def test_create_judge_id(self):
        """Test unique judge ID creation."""
        row = pd.Series({
            "name_last": "Smith",
            "name_first": "John",
            "court_id": "azd",
        })
        judge_id = JudicialDataPreprocessor._create_judge_id(row)
        assert judge_id == "smith_john_azd"

    def test_save_and_load_processed_data(self, preprocessor, tmp_path):
        """Test saving and loading processed data."""
        preprocessor.preprocess_all()

        # Save
        output_dir = str(tmp_path / "processed")
        preprocessor.save_processed_data(output_dir)

        # Load
        loaded = JudicialDataPreprocessor.load_processed_data(output_dir)

        assert loaded.judges is not None
        assert len(loaded.judges) == len(preprocessor.judges)


class TestEmptyDataHandling:
    """Test handling of empty/missing data."""

    def test_empty_raw_data(self):
        """Test with completely empty raw data."""
        preprocessor = JudicialDataPreprocessor({})
        result = preprocessor.preprocess_all()

        assert all(isinstance(v, pd.DataFrame) for v in result.values())

    def test_empty_judges(self):
        """Test with no judge data."""
        preprocessor = JudicialDataPreprocessor({"federal_judges": [], "state_judges": []})
        result = preprocessor.preprocess_judges()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_link_without_preprocessing(self):
        """Test link_records without prior preprocessing."""
        preprocessor = JudicialDataPreprocessor({})
        result = preprocessor.link_records()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
