"""
Data Preprocessing Module for Judicial Integrity Analysis.

Cleans, normalizes, and transforms raw judicial data from multiple sources
into structured formats suitable for integrity analysis.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class JudicialDataPreprocessor:
    """
    Preprocesses raw judicial data into analysis-ready DataFrames.

    Handles:
    - Judge profile normalization (name, court, appointment details)
    - Disciplinary record structuring
    - Performance review score extraction
    - Sentencing data cleaning
    - Cross-source record linkage
    """

    # Appointment method categories
    APPOINTMENT_METHODS = {
        "presidential": "Presidential Appointment",
        "gubernatorial": "Gubernatorial Appointment",
        "merit_selection": "Merit Selection",
        "partisan_election": "Partisan Election",
        "nonpartisan_election": "Nonpartisan Election",
        "legislative": "Legislative Appointment",
    }

    # Court level hierarchy
    COURT_LEVELS = {
        "supreme": 4,
        "appellate": 3,
        "appeals": 3,
        "district": 2,
        "superior": 2,
        "magistrate": 1,
        "municipal": 0,
        "justice": 0,
    }

    # Political party normalization
    PARTY_NORMALIZATION = {
        "republican": "Republican",
        "rep": "Republican",
        "r": "Republican",
        "gop": "Republican",
        "democrat": "Democrat",
        "dem": "Democrat",
        "d": "Democrat",
        "independent": "Independent",
        "ind": "Independent",
        "i": "Independent",
        "libertarian": "Libertarian",
        "green": "Green",
        "nonpartisan": "Nonpartisan",
        "unknown": "Unknown",
        "none": "Unknown",
    }

    def __init__(self, raw_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor.

        Args:
            raw_data: Dictionary of raw data from JudicialDataAggregator.collect_all().
        """
        self.raw_data = raw_data or {}

        # Processed DataFrames
        self.judges: Optional[pd.DataFrame] = None
        self.disciplinary_records: Optional[pd.DataFrame] = None
        self.performance_reviews: Optional[pd.DataFrame] = None
        self.opinions: Optional[pd.DataFrame] = None
        self.sentencing_data: Optional[pd.DataFrame] = None

    # ==================== Judge Profile Processing ====================

    def preprocess_judges(
        self,
        federal_judges: Optional[List[Dict[str, Any]]] = None,
        state_judges: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Clean and normalize judge profile data.

        Args:
            federal_judges: List of federal judge records from CourtListener.
            state_judges: List of state judge records.

        Returns:
            Normalized DataFrame of judge profiles.
        """
        federal = federal_judges or self.raw_data.get("federal_judges", [])
        state = state_judges or self.raw_data.get("state_judges", [])

        records = []
        for judge in federal:
            records.append(self._normalize_judge_record(judge, level="federal"))
        for judge in state:
            records.append(self._normalize_judge_record(judge, level="state"))

        if not records:
            self.judges = pd.DataFrame(columns=self._judge_columns())
            return self.judges

        df = pd.DataFrame(records)

        # Standardize names
        df["name_full"] = df["name_full"].apply(self._normalize_name)

        # Normalize party affiliation
        df["party"] = df["party"].apply(self._normalize_party)

        # Parse dates
        for date_col in ["date_appointed", "date_born", "date_terminated"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Compute derived fields
        if "date_appointed" in df.columns:
            df["years_serving"] = (
                pd.Timestamp.now() - df["date_appointed"]
            ).dt.days / 365.25
            df.loc[df["date_terminated"].notna(), "years_serving"] = (
                (df["date_terminated"] - df["date_appointed"]).dt.days / 365.25
            )

        # Assign court level
        df["court_level"] = df["court_name"].apply(self._classify_court_level)

        # Create unique judge ID
        df["judge_id"] = df.apply(
            lambda row: self._create_judge_id(row), axis=1
        )

        df = df.sort_values(["level", "court_name", "name_full"]).reset_index(
            drop=True
        )

        self.judges = df
        logger.info("Preprocessed %d judge records", len(df))
        return df

    def _normalize_judge_record(
        self, record: Dict[str, Any], level: str
    ) -> Dict[str, Any]:
        """Convert a raw judge record into a standardized dictionary."""
        return {
            "source_id": record.get("id", ""),
            "name_full": (
                f"{record.get('name_first', '')} "
                f"{record.get('name_middle', '')} "
                f"{record.get('name_last', '')}".strip()
            ),
            "name_first": record.get("name_first", ""),
            "name_last": record.get("name_last", ""),
            "name_suffix": record.get("name_suffix", ""),
            "level": level,
            "court_id": record.get("court", ""),
            "court_name": record.get("court_name", ""),
            "date_appointed": record.get("date_nominated")
            or record.get("date_appointed"),
            "date_born": record.get("date_dob"),
            "date_terminated": record.get("date_terminated"),
            "appointing_president": record.get("appointing_president", ""),
            "party": record.get("political_affiliation", "Unknown"),
            "appointment_method": record.get("appointment_method", ""),
            "is_active": record.get("date_terminated") is None,
            "gender": record.get("gender", ""),
            "race": record.get("race", ""),
            "has_financial_disclosures": bool(
                record.get("financial_disclosures")
            ),
            "n_financial_disclosures": len(
                record.get("financial_disclosures", [])
            ),
        }

    def _judge_columns(self) -> List[str]:
        """Return the standard column set for judge DataFrames."""
        return [
            "judge_id",
            "source_id",
            "name_full",
            "name_first",
            "name_last",
            "name_suffix",
            "level",
            "court_id",
            "court_name",
            "date_appointed",
            "date_born",
            "date_terminated",
            "appointing_president",
            "party",
            "appointment_method",
            "is_active",
            "gender",
            "race",
            "has_financial_disclosures",
            "n_financial_disclosures",
            "years_serving",
            "court_level",
        ]

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a judge's name for consistency."""
        if not isinstance(name, str):
            return ""
        # Collapse multiple spaces
        name = re.sub(r"\s+", " ", name).strip()
        # Title case
        name = name.title()
        # Fix Roman numeral suffixes that need uppercasing (longest first to avoid partial matches)
        for suffix in ["Iii", "Ii", "Iv"]:
            name = name.replace(f" {suffix}", f" {suffix.upper()}")
        return name

    def _normalize_party(self, party: str) -> str:
        """Normalize a political party affiliation string."""
        if not isinstance(party, str):
            return "Unknown"
        return self.PARTY_NORMALIZATION.get(party.lower().strip(), "Unknown")

    def _classify_court_level(self, court_name: str) -> int:
        """Classify court level from court name (higher = more senior)."""
        if not isinstance(court_name, str):
            return -1
        name_lower = court_name.lower()
        for key, level in self.COURT_LEVELS.items():
            if key in name_lower:
                return level
        return -1

    @staticmethod
    def _create_judge_id(row: pd.Series) -> str:
        """Create a stable unique judge identifier."""
        parts = [
            str(row.get("name_last", "")).lower().replace(" ", "_"),
            str(row.get("name_first", "")).lower().replace(" ", "_"),
            str(row.get("court_id", "")),
        ]
        return "_".join(p for p in parts if p)

    # ==================== Disciplinary Record Processing ====================

    def preprocess_disciplinary_records(
        self, records: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Clean and structure disciplinary / conduct records.

        Args:
            records: List of raw disciplinary records.

        Returns:
            Structured DataFrame of disciplinary records.
        """
        raw = records or self.raw_data.get("disciplinary_records", [])

        if not raw:
            self.disciplinary_records = pd.DataFrame(
                columns=[
                    "judge_id",
                    "date",
                    "action_type",
                    "description",
                    "source",
                    "severity",
                ]
            )
            return self.disciplinary_records

        df = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
        df["severity"] = df.get("action_type", pd.Series(dtype=str)).apply(
            self._classify_severity
        )

        self.disciplinary_records = df
        logger.info("Preprocessed %d disciplinary records", len(df))
        return df

    @staticmethod
    def _classify_severity(action_type: str) -> str:
        """Classify disciplinary action severity."""
        if not isinstance(action_type, str):
            return "unknown"
        action_lower = action_type.lower()
        if any(
            w in action_lower
            for w in ["removal", "disbarment", "termination"]
        ):
            return "critical"
        if any(w in action_lower for w in ["suspension", "censure"]):
            return "severe"
        if any(w in action_lower for w in ["reprimand", "admonition"]):
            return "moderate"
        if any(w in action_lower for w in ["warning", "caution", "advisory"]):
            return "minor"
        return "unknown"

    # ==================== Performance Review Processing ====================

    def preprocess_performance_reviews(
        self, reviews: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Clean and structure judicial performance review data.

        Args:
            reviews: List of raw performance review records.

        Returns:
            Structured DataFrame of performance reviews.
        """
        raw = reviews or self.raw_data.get("performance_reviews", [])

        if not raw:
            self.performance_reviews = pd.DataFrame(
                columns=[
                    "judge_id",
                    "review_year",
                    "overall_score",
                    "legal_ability",
                    "integrity",
                    "temperament",
                    "communication",
                    "administrative",
                    "meets_standard",
                    "source",
                ]
            )
            return self.performance_reviews

        df = pd.DataFrame(raw)

        # Normalize scores to 0-100 scale
        score_columns = [
            "overall_score",
            "legal_ability",
            "integrity",
            "temperament",
            "communication",
            "administrative",
        ]
        for col in score_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self.performance_reviews = df
        logger.info("Preprocessed %d performance review records", len(df))
        return df

    # ==================== Sentencing Data Processing ====================

    def preprocess_sentencing_data(
        self, sentencing_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Clean and normalize sentencing data for disparity analysis.

        Args:
            sentencing_df: Raw sentencing DataFrame.

        Returns:
            Cleaned sentencing DataFrame with standardized fields.
        """
        if sentencing_df is None:
            self.sentencing_data = pd.DataFrame(
                columns=[
                    "judge_id",
                    "case_id",
                    "offense_type",
                    "offense_level",
                    "criminal_history",
                    "guideline_min",
                    "guideline_max",
                    "sentence_months",
                    "departure_type",
                    "departure_reason",
                    "defendant_race",
                    "defendant_gender",
                    "defendant_age",
                    "district",
                    "year",
                ]
            )
            return self.sentencing_data

        df = sentencing_df.copy()

        # Standardize column names
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

        # Compute departure from guidelines
        if "guideline_min" in df.columns and "sentence_months" in df.columns:
            df["departure_months"] = df["sentence_months"] - df["guideline_min"]
            df["departure_pct"] = np.where(
                df["guideline_min"] > 0,
                (df["sentence_months"] - df["guideline_min"])
                / df["guideline_min"]
                * 100,
                0,
            )

        self.sentencing_data = df
        logger.info("Preprocessed %d sentencing records", len(df))
        return df

    # ==================== Cross-Source Linkage ====================

    def link_records(self) -> pd.DataFrame:
        """
        Link records across data sources using judge IDs.

        Returns:
            Merged DataFrame with judge profiles enriched with
            disciplinary counts, performance scores, and sentencing stats.
        """
        if self.judges is None:
            logger.warning("No judge data to link. Run preprocess_judges first.")
            return pd.DataFrame()

        df = self.judges.copy()

        # Merge disciplinary summary
        if (
            self.disciplinary_records is not None
            and len(self.disciplinary_records) > 0
        ):
            disc_summary = (
                self.disciplinary_records.groupby("judge_id")
                .agg(
                    n_disciplinary_actions=("action_type", "count"),
                    max_severity=("severity", lambda s: s.mode().iloc[0] if len(s) > 0 else "none"),
                )
                .reset_index()
            )
            df = df.merge(disc_summary, on="judge_id", how="left")
        else:
            df["n_disciplinary_actions"] = 0
            df["max_severity"] = "none"

        # Merge performance review summary
        if (
            self.performance_reviews is not None
            and len(self.performance_reviews) > 0
        ):
            perf_summary = (
                self.performance_reviews.groupby("judge_id")
                .agg(
                    avg_overall_score=("overall_score", "mean"),
                    avg_integrity_score=("integrity", "mean"),
                    n_reviews=("review_year", "count"),
                    meets_standard_pct=(
                        "meets_standard",
                        lambda s: s.mean() * 100 if s.dtype == bool else np.nan,
                    ),
                )
                .reset_index()
            )
            df = df.merge(perf_summary, on="judge_id", how="left")
        else:
            df["avg_overall_score"] = np.nan
            df["avg_integrity_score"] = np.nan
            df["n_reviews"] = 0
            df["meets_standard_pct"] = np.nan

        # Merge sentencing summary
        if self.sentencing_data is not None and len(self.sentencing_data) > 0:
            sent_summary = (
                self.sentencing_data.groupby("judge_id")
                .agg(
                    n_sentences=("case_id", "count"),
                    avg_sentence_months=("sentence_months", "mean"),
                    avg_departure_pct=("departure_pct", "mean"),
                    median_sentence_months=("sentence_months", "median"),
                )
                .reset_index()
            )
            df = df.merge(sent_summary, on="judge_id", how="left")
        else:
            df["n_sentences"] = 0
            df["avg_sentence_months"] = np.nan
            df["avg_departure_pct"] = np.nan
            df["median_sentence_months"] = np.nan

        logger.info("Linked records for %d judges", len(df))
        return df

    # ==================== Full Pipeline ====================

    def preprocess_all(self) -> Dict[str, pd.DataFrame]:
        """
        Run the full preprocessing pipeline.

        Returns:
            Dictionary of processed DataFrames keyed by data type.
        """
        result = {
            "judges": self.preprocess_judges(),
            "disciplinary_records": self.preprocess_disciplinary_records(),
            "performance_reviews": self.preprocess_performance_reviews(),
            "sentencing_data": self.preprocess_sentencing_data(),
        }
        return result

    # ==================== Persistence ====================

    def save_processed_data(self, output_dir: str = "data/processed") -> None:
        """
        Save processed DataFrames to parquet files.

        Args:
            output_dir: Directory to write output files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, df in [
            ("judges", self.judges),
            ("disciplinary_records", self.disciplinary_records),
            ("performance_reviews", self.performance_reviews),
            ("sentencing_data", self.sentencing_data),
        ]:
            if df is not None and len(df) > 0:
                filepath = output_path / f"{name}.parquet"
                df.to_parquet(filepath, index=False)
                logger.info("Saved %s to %s", name, filepath)

    @classmethod
    def load_processed_data(
        cls, processed_dir: str = "data/processed"
    ) -> "JudicialDataPreprocessor":
        """
        Load previously processed data from parquet files.

        Args:
            processed_dir: Directory containing processed parquet files.

        Returns:
            JudicialDataPreprocessor with loaded data.
        """
        path = Path(processed_dir)
        preprocessor = cls()

        for name in ["judges", "disciplinary_records", "performance_reviews", "sentencing_data"]:
            filepath = path / f"{name}.parquet"
            if filepath.exists():
                setattr(preprocessor, name, pd.read_parquet(filepath))
                logger.info("Loaded %s from %s", name, filepath)

        return preprocessor
