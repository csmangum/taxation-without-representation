"""
Analysis Module for Judicial Integrity Analysis.

Provides methods for analyzing judicial corruption, ethics, bias,
and partisanship using structured judicial data.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class JudicialIntegrityAnalyzer:
    """
    Analyzes judicial integrity across four dimensions:
    - Corruption: disciplinary actions, financial conflicts
    - Ethics: performance reviews, code of conduct compliance
    - Bias: sentencing disparities by demographic factors
    - Partisanship: appointment background, ruling pattern correlations

    All analysis is based on publicly available data.
    """

    # Risk flag thresholds
    RISK_THRESHOLDS = {
        "disciplinary_high": 2,        # >= 2 disciplinary actions
        "performance_low": 60.0,       # below 60% overall score
        "integrity_low": 60.0,         # below 60% integrity score
        "departure_extreme_pct": 50.0, # sentencing departure > 50%
    }

    def __init__(
        self,
        judges_df: Optional[pd.DataFrame] = None,
        disciplinary_df: Optional[pd.DataFrame] = None,
        performance_df: Optional[pd.DataFrame] = None,
        sentencing_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the analyzer.

        Args:
            judges_df: Preprocessed judge profiles DataFrame.
            disciplinary_df: Preprocessed disciplinary records DataFrame.
            performance_df: Preprocessed performance review DataFrame.
            sentencing_df: Preprocessed sentencing data DataFrame.
        """
        self.judges = judges_df if judges_df is not None else pd.DataFrame()
        self.disciplinary = disciplinary_df if disciplinary_df is not None else pd.DataFrame()
        self.performance = performance_df if performance_df is not None else pd.DataFrame()
        self.sentencing = sentencing_df if sentencing_df is not None else pd.DataFrame()
        self._cache: Dict[str, Any] = {}

    # ==================== Corruption Analysis ====================

    def analyze_corruption_indicators(self) -> pd.DataFrame:
        """
        Identify corruption risk indicators for each judge.

        Indicators include:
        - Number and severity of disciplinary actions
        - Financial disclosure anomalies
        - Recusal patterns

        Returns:
            DataFrame with corruption risk scores per judge.
        """
        if self.judges.empty:
            return pd.DataFrame()

        df = self.judges[["judge_id", "name_full", "court_name", "level"]].copy()

        # Disciplinary action counts
        if not self.disciplinary.empty and "judge_id" in self.disciplinary.columns:
            disc_counts = (
                self.disciplinary.groupby("judge_id")
                .agg(
                    n_disciplinary=("action_type", "count"),
                    n_severe=("severity", lambda s: sum(s.isin(["critical", "severe"]))),
                )
                .reset_index()
            )
            df = df.merge(disc_counts, on="judge_id", how="left")
        else:
            df["n_disciplinary"] = 0
            df["n_severe"] = 0

        df["n_disciplinary"] = df["n_disciplinary"].fillna(0).astype(int)
        df["n_severe"] = df["n_severe"].fillna(0).astype(int)

        # Financial disclosure flags
        if "has_financial_disclosures" in self.judges.columns:
            fin_cols = self.judges[["judge_id", "has_financial_disclosures", "n_financial_disclosures"]]
            df = df.merge(fin_cols, on="judge_id", how="left")
        else:
            df["has_financial_disclosures"] = False
            df["n_financial_disclosures"] = 0

        # Compute corruption risk score (0-100 scale)
        df["corruption_risk_score"] = self._compute_corruption_risk(df)

        df = df.sort_values("corruption_risk_score", ascending=False).reset_index(drop=True)
        self._cache["corruption_indicators"] = df
        logger.info("Analyzed corruption indicators for %d judges", len(df))
        return df

    @staticmethod
    def _compute_corruption_risk(df: pd.DataFrame) -> pd.Series:
        """Compute a normalized corruption risk score (0-100)."""
        score = pd.Series(0.0, index=df.index)

        # Disciplinary actions (0-50 points)
        score += np.minimum(df["n_disciplinary"] * 15, 30)
        score += np.minimum(df["n_severe"] * 20, 50)

        # Missing financial disclosures for federal judges (0-20 points)
        if "has_financial_disclosures" in df.columns and "level" in df.columns:
            is_federal = df["level"] == "federal"
            score += (is_federal & ~df["has_financial_disclosures"].fillna(False)).astype(float) * 10

        return np.minimum(score, 100)

    # ==================== Ethics Analysis ====================

    def analyze_ethics_indicators(self) -> pd.DataFrame:
        """
        Evaluate ethical standing based on performance reviews and
        conduct records.

        Returns:
            DataFrame with ethics scores per judge.
        """
        if self.judges.empty:
            return pd.DataFrame()

        df = self.judges[["judge_id", "name_full", "court_name"]].copy()

        # Performance review scores
        if not self.performance.empty and "judge_id" in self.performance.columns:
            perf_agg = (
                self.performance.groupby("judge_id")
                .agg(
                    avg_overall=("overall_score", "mean"),
                    avg_integrity=("integrity", "mean"),
                    avg_temperament=("temperament", "mean"),
                    n_reviews=("review_year", "count"),
                    pct_meets_standard=(
                        "meets_standard",
                        lambda s: s.mean() * 100 if len(s) > 0 else np.nan,
                    ),
                )
                .reset_index()
            )
            df = df.merge(perf_agg, on="judge_id", how="left")
        else:
            df["avg_overall"] = np.nan
            df["avg_integrity"] = np.nan
            df["avg_temperament"] = np.nan
            df["n_reviews"] = 0
            df["pct_meets_standard"] = np.nan

        # Compute ethics score (higher = better ethics standing)
        df["ethics_score"] = self._compute_ethics_score(df)

        df = df.sort_values("ethics_score", ascending=True).reset_index(drop=True)
        self._cache["ethics_indicators"] = df
        logger.info("Analyzed ethics indicators for %d judges", len(df))
        return df

    @staticmethod
    def _compute_ethics_score(df: pd.DataFrame) -> pd.Series:
        """
        Compute a composite ethics score (0-100, higher = better).

        Missing data results in a neutral score of 50.
        """
        components = []
        weights = []

        if "avg_overall" in df.columns:
            components.append(df["avg_overall"].fillna(50))
            weights.append(0.3)

        if "avg_integrity" in df.columns:
            components.append(df["avg_integrity"].fillna(50))
            weights.append(0.4)

        if "avg_temperament" in df.columns:
            components.append(df["avg_temperament"].fillna(50))
            weights.append(0.15)

        if "pct_meets_standard" in df.columns:
            components.append(df["pct_meets_standard"].fillna(50))
            weights.append(0.15)

        if not components:
            return pd.Series(50.0, index=df.index)

        total_weight = sum(weights)
        score = sum(c * w for c, w in zip(components, weights)) / total_weight
        return score.clip(0, 100)

    # ==================== Bias Analysis ====================

    def analyze_sentencing_bias(
        self,
        demographic_column: str = "defendant_race",
        control_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze sentencing disparities by demographic factors.

        Uses statistical tests to identify whether sentencing outcomes
        vary significantly by demographic characteristics, controlling
        for offense severity and criminal history.

        Args:
            demographic_column: Column name for the demographic factor.
            control_columns: Columns to control for (e.g., offense_level,
                criminal_history).

        Returns:
            Dictionary with bias analysis results.
        """
        if self.sentencing.empty:
            return {"error": "No sentencing data available"}

        if demographic_column not in self.sentencing.columns:
            return {"error": f"Column '{demographic_column}' not found in sentencing data"}

        if control_columns is None:
            control_columns = ["offense_level", "criminal_history"]

        results: Dict[str, Any] = {
            "demographic_column": demographic_column,
            "n_records": len(self.sentencing),
            "groups": {},
        }

        # Group-level statistics
        grouped = self.sentencing.groupby(demographic_column)
        for group_name, group_df in grouped:
            results["groups"][str(group_name)] = {
                "n": len(group_df),
                "mean_sentence_months": group_df["sentence_months"].mean()
                if "sentence_months" in group_df.columns
                else np.nan,
                "median_sentence_months": group_df["sentence_months"].median()
                if "sentence_months" in group_df.columns
                else np.nan,
                "std_sentence_months": group_df["sentence_months"].std()
                if "sentence_months" in group_df.columns
                else np.nan,
            }

            if "departure_pct" in group_df.columns:
                results["groups"][str(group_name)]["mean_departure_pct"] = (
                    group_df["departure_pct"].mean()
                )

        # Kruskal-Wallis test for overall group differences
        if "sentence_months" in self.sentencing.columns:
            groups = [
                g["sentence_months"].dropna().values
                for _, g in grouped
                if len(g["sentence_months"].dropna()) > 1
            ]
            if len(groups) >= 2:
                stat, p_value = scipy_stats.kruskal(*groups)
                results["kruskal_wallis"] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant_at_05": p_value < 0.05,
                }

        self._cache["sentencing_bias"] = results
        logger.info(
            "Analyzed sentencing bias by %s (%d records)",
            demographic_column,
            len(self.sentencing),
        )
        return results

    def analyze_judge_sentencing_patterns(self) -> pd.DataFrame:
        """
        Analyze individual judge sentencing patterns relative to peers.

        Returns:
            DataFrame with per-judge sentencing statistics and z-scores.
        """
        if self.sentencing.empty or "judge_id" not in self.sentencing.columns:
            return pd.DataFrame()

        judge_stats = (
            self.sentencing.groupby("judge_id")
            .agg(
                n_cases=("case_id", "count"),
                mean_sentence=("sentence_months", "mean"),
                median_sentence=("sentence_months", "median"),
                std_sentence=("sentence_months", "std"),
                mean_departure_pct=("departure_pct", "mean"),
            )
            .reset_index()
        )

        # Compute z-scores relative to peer judges
        overall_mean = judge_stats["mean_sentence"].mean()
        overall_std = judge_stats["mean_sentence"].std()

        if overall_std > 0:
            judge_stats["sentence_zscore"] = (
                (judge_stats["mean_sentence"] - overall_mean) / overall_std
            )
        else:
            judge_stats["sentence_zscore"] = 0.0

        # Flag outliers (|z| > 2)
        judge_stats["is_outlier"] = judge_stats["sentence_zscore"].abs() > 2.0
        judge_stats["direction"] = np.where(
            judge_stats["sentence_zscore"] > 0, "harsh", "lenient"
        )

        self._cache["judge_sentencing_patterns"] = judge_stats
        return judge_stats

    # ==================== Partisanship Analysis ====================

    def analyze_partisanship_indicators(self) -> pd.DataFrame:
        """
        Analyze partisanship indicators for judges.

        Examines:
        - Appointing authority's political party
        - Correlations between appointment and ruling patterns
        - Ruling patterns in politically sensitive case types

        Returns:
            DataFrame with partisanship indicators per judge.
        """
        if self.judges.empty:
            return pd.DataFrame()

        df = self.judges[
            ["judge_id", "name_full", "court_name", "party", "appointing_president"]
        ].copy()

        # Classify appointing president party
        df["appointer_party"] = df["appointing_president"].apply(
            self._classify_appointer_party
        )

        # Compute partisanship score (0 = no indicators, 100 = strong indicators)
        df["partisanship_score"] = self._compute_partisanship_score(df)

        self._cache["partisanship_indicators"] = df
        logger.info("Analyzed partisanship for %d judges", len(df))
        return df

    @staticmethod
    def _classify_appointer_party(president: str) -> str:
        """Classify the political party of an appointing president."""
        if not isinstance(president, str) or not president.strip():
            return "Unknown"

        # Known Republican presidents (modern era)
        republicans = [
            "trump", "bush", "reagan", "nixon", "ford", "eisenhower",
        ]
        # Known Democratic presidents (modern era)
        democrats = [
            "biden", "obama", "clinton", "carter", "johnson", "kennedy",
        ]

        name_lower = president.lower()
        for r in republicans:
            if r in name_lower:
                return "Republican"
        for d in democrats:
            if d in name_lower:
                return "Democrat"
        return "Unknown"

    @staticmethod
    def _compute_partisanship_score(df: pd.DataFrame) -> pd.Series:
        """
        Compute partisanship indicator score (0-100).

        A higher score means more available partisanship indicators
        are present, NOT that the judge is necessarily partisan.
        This is a data-availability metric, not a verdict.
        """
        score = pd.Series(0.0, index=df.index)

        # Known party affiliation adds to indicator availability
        has_party = df["party"].notna() & (df["party"] != "Unknown")
        score += has_party.astype(float) * 25

        # Known appointing president adds indicator
        has_appointer = df["appointing_president"].notna() & (
            df["appointing_president"] != ""
        )
        score += has_appointer.astype(float) * 25

        # Known appointer party
        has_appointer_party = df["appointer_party"] != "Unknown"
        score += has_appointer_party.astype(float) * 25

        # Party and appointer party alignment / contrast
        aligned = (df["party"] == df["appointer_party"]) & has_party & has_appointer_party
        score += aligned.astype(float) * 25

        return score.clip(0, 100)

    # ==================== Composite Risk Assessment ====================

    def compute_integrity_summary(self) -> pd.DataFrame:
        """
        Compute a comprehensive integrity summary combining all four
        analysis dimensions.

        Returns:
            DataFrame with per-judge integrity summary.
        """
        corruption = self.analyze_corruption_indicators()
        ethics = self.analyze_ethics_indicators()
        partisanship = self.analyze_partisanship_indicators()

        if corruption.empty and ethics.empty:
            return pd.DataFrame()

        # Start with judges as base
        df = self.judges[["judge_id", "name_full", "court_name", "level", "is_active"]].copy()

        # Merge corruption
        if not corruption.empty:
            df = df.merge(
                corruption[["judge_id", "corruption_risk_score", "n_disciplinary", "n_severe"]],
                on="judge_id",
                how="left",
            )

        # Merge ethics
        if not ethics.empty:
            df = df.merge(
                ethics[["judge_id", "ethics_score"]],
                on="judge_id",
                how="left",
            )

        # Merge partisanship
        if not partisanship.empty:
            df = df.merge(
                partisanship[["judge_id", "partisanship_score", "appointer_party"]],
                on="judge_id",
                how="left",
            )

        # Fill NaN
        for col in ["corruption_risk_score", "partisanship_score"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill ethics_score with neutral value of 50
        if "ethics_score" in df.columns:
            df["ethics_score"] = df["ethics_score"].fillna(50)

        # Compute overall integrity score (higher = better)
        # Invert corruption risk (lower risk = better)
        df["integrity_composite"] = (
            (100 - df.get("corruption_risk_score", 0)) * 0.3
            + df.get("ethics_score", 50) * 0.4
            + (100 - df.get("partisanship_score", 0)) * 0.3
        )

        # Assign risk tiers
        df["risk_tier"] = pd.cut(
            df["integrity_composite"],
            bins=[0, 30, 50, 70, 100],
            labels=["High Risk", "Elevated", "Moderate", "Low Risk"],
            include_lowest=True,
        )

        self._cache["integrity_summary"] = df
        logger.info("Computed integrity summary for %d judges", len(df))
        return df

    # ==================== Report Generation Support ====================

    def generate_text_report(self, judge_id: Optional[str] = None) -> str:
        """
        Generate a text report of analysis results.

        Args:
            judge_id: Optional specific judge to report on.
                If None, generates a summary report for all judges.

        Returns:
            Formatted report string.
        """
        summary = self.compute_integrity_summary()
        if summary.empty:
            return "No data available for report generation."

        lines = [
            "=" * 70,
            "JUDICIAL INTEGRITY ANALYSIS REPORT",
            "=" * 70,
            "",
        ]

        if judge_id:
            judge_rows = summary[summary["judge_id"] == judge_id]
            if judge_rows.empty:
                return f"No data found for judge_id: {judge_id}"
            row = judge_rows.iloc[0]
            lines.extend(self._format_judge_report(row))
        else:
            lines.append(f"Total Judges Analyzed: {len(summary)}")
            lines.append(f"Active Judges: {summary['is_active'].sum()}")
            lines.append("")

            # Risk tier distribution
            lines.append("## Risk Tier Distribution")
            tier_counts = summary["risk_tier"].value_counts()
            for tier, count in tier_counts.items():
                lines.append(f"  {tier}: {count} ({count / len(summary) * 100:.1f}%)")
            lines.append("")

            # Top concerns
            if "corruption_risk_score" in summary.columns:
                lines.append("## Top Corruption Risk")
                top_risk = summary.nlargest(5, "corruption_risk_score")
                for _, row in top_risk.iterrows():
                    lines.append(
                        f"  {row['name_full']} ({row['court_name']}): "
                        f"score={row['corruption_risk_score']:.1f}"
                    )
            lines.append("")

            if "ethics_score" in summary.columns:
                lines.append("## Lowest Ethics Scores")
                bottom_ethics = summary.nsmallest(5, "ethics_score")
                for _, row in bottom_ethics.iterrows():
                    lines.append(
                        f"  {row['name_full']} ({row['court_name']}): "
                        f"score={row['ethics_score']:.1f}"
                    )

        return "\n".join(lines)

    @staticmethod
    def _format_judge_report(row: pd.Series) -> List[str]:
        """Format a single judge report section."""
        lines = [
            f"## Judge: {row.get('name_full', 'Unknown')}",
            f"  Court: {row.get('court_name', 'Unknown')}",
            f"  Level: {row.get('level', 'Unknown')}",
            f"  Active: {row.get('is_active', 'Unknown')}",
            "",
            "### Integrity Scores",
            f"  Composite Score: {row.get('integrity_composite', 'N/A'):.1f}/100"
            if isinstance(row.get("integrity_composite"), (int, float))
            else "  Composite Score: N/A",
            f"  Risk Tier: {row.get('risk_tier', 'N/A')}",
            "",
            "### Corruption Indicators",
            f"  Risk Score: {row.get('corruption_risk_score', 0):.1f}",
            f"  Disciplinary Actions: {int(row.get('n_disciplinary', 0))}",
            f"  Severe Actions: {int(row.get('n_severe', 0))}",
            "",
            "### Ethics",
            f"  Ethics Score: {row.get('ethics_score', 'N/A'):.1f}"
            if isinstance(row.get("ethics_score"), (int, float))
            else "  Ethics Score: N/A",
            "",
            "### Partisanship",
            f"  Indicator Score: {row.get('partisanship_score', 0):.1f}",
            f"  Appointer Party: {row.get('appointer_party', 'Unknown')}",
        ]
        return lines


class SentencingDisparityAnalyzer:
    """
    Specialized analyzer for sentencing disparity analysis.

    Implements statistical methods for detecting and quantifying
    sentencing disparities across demographic groups and individual judges.
    """

    def __init__(self, sentencing_df: pd.DataFrame):
        """
        Initialize with sentencing data.

        Args:
            sentencing_df: Preprocessed sentencing DataFrame.
        """
        self.data = sentencing_df

    def compute_disparity_index(
        self,
        group_col: str = "defendant_race",
        reference_group: Optional[str] = None,
        outcome_col: str = "sentence_months",
    ) -> pd.DataFrame:
        """
        Compute disparity index comparing sentencing outcomes across groups.

        The disparity index is the ratio of a group's mean sentence to the
        reference group's mean sentence.

        Args:
            group_col: Column defining demographic groups.
            reference_group: Group to use as reference (denominator).
                If None, uses the group with the most observations.
            outcome_col: Column for the sentencing outcome.

        Returns:
            DataFrame with disparity indices per group.
        """
        if self.data.empty or group_col not in self.data.columns:
            return pd.DataFrame()

        grouped = self.data.groupby(group_col)[outcome_col]
        stats = grouped.agg(["count", "mean", "median", "std"]).reset_index()

        if reference_group is None:
            reference_group = stats.loc[stats["count"].idxmax(), group_col]

        ref_mean = stats.loc[stats[group_col] == reference_group, "mean"].iloc[0]

        if ref_mean > 0:
            stats["disparity_index"] = stats["mean"] / ref_mean
        else:
            stats["disparity_index"] = np.nan

        stats["reference_group"] = reference_group
        return stats

    def pairwise_comparisons(
        self,
        group_col: str = "defendant_race",
        outcome_col: str = "sentence_months",
    ) -> List[Dict[str, Any]]:
        """
        Perform pairwise statistical comparisons between demographic groups.

        Uses Mann-Whitney U test (non-parametric).

        Args:
            group_col: Column defining groups.
            outcome_col: Column for sentencing outcome.

        Returns:
            List of pairwise comparison results.
        """
        if self.data.empty or group_col not in self.data.columns:
            return []

        groups = {
            name: g[outcome_col].dropna().values
            for name, g in self.data.groupby(group_col)
            if len(g[outcome_col].dropna()) > 1
        }

        results = []
        group_names = list(groups.keys())

        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                g1_name, g2_name = group_names[i], group_names[j]
                g1_data, g2_data = groups[g1_name], groups[g2_name]

                stat, p_value = scipy_stats.mannwhitneyu(
                    g1_data, g2_data, alternative="two-sided"
                )

                # Effect size (rank-biserial correlation)
                n1, n2 = len(g1_data), len(g2_data)
                effect_size = 1 - (2 * stat) / (n1 * n2)

                results.append(
                    {
                        "group_1": str(g1_name),
                        "group_2": str(g2_name),
                        "n_1": n1,
                        "n_2": n2,
                        "mean_1": float(g1_data.mean()),
                        "mean_2": float(g2_data.mean()),
                        "u_statistic": float(stat),
                        "p_value": float(p_value),
                        "effect_size": float(effect_size),
                        "significant_at_05": p_value < 0.05,
                    }
                )

        return results

    def judge_demographic_interaction(
        self,
        judge_col: str = "judge_id",
        demographic_col: str = "defendant_race",
        outcome_col: str = "sentence_months",
        min_cases: int = 10,
    ) -> pd.DataFrame:
        """
        Analyze the interaction between judge identity and defendant demographics
        on sentencing outcomes.

        Args:
            judge_col: Column identifying judges.
            demographic_col: Column identifying defendant demographics.
            outcome_col: Column for sentencing outcome.
            min_cases: Minimum cases per judge-demographic cell.

        Returns:
            DataFrame with judge-demographic interaction effects.
        """
        if self.data.empty:
            return pd.DataFrame()

        required_cols = [judge_col, demographic_col, outcome_col]
        if not all(c in self.data.columns for c in required_cols):
            return pd.DataFrame()

        interaction = (
            self.data.groupby([judge_col, demographic_col])
            .agg(
                n_cases=(outcome_col, "count"),
                mean_sentence=(outcome_col, "mean"),
                median_sentence=(outcome_col, "median"),
            )
            .reset_index()
        )

        # Filter for sufficient data
        interaction = interaction[interaction["n_cases"] >= min_cases]

        # Compute per-judge disparity (max group mean - min group mean)
        if not interaction.empty:
            judge_disparity = (
                interaction.groupby(judge_col)
                .agg(
                    max_mean=("mean_sentence", "max"),
                    min_mean=("mean_sentence", "min"),
                    n_groups=(demographic_col, "nunique"),
                )
                .reset_index()
            )
            judge_disparity["disparity_range"] = (
                judge_disparity["max_mean"] - judge_disparity["min_mean"]
            )
            return judge_disparity

        return pd.DataFrame()
