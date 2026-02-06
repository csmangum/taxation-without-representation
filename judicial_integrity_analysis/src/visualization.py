"""
Visualization Module for Judicial Integrity Analysis.

Provides static and interactive visualizations for judicial integrity
analysis results, including risk dashboards, sentencing disparity charts,
and comparative judge profiles.
"""

import logging
from typing import Optional, List, Tuple
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logger = logging.getLogger(__name__)


class JudicialVisualizer:
    """
    Creates visualizations for judicial integrity analysis.

    Supports:
    - Risk tier distribution charts
    - Sentencing disparity plots
    - Judge comparison dashboards
    - Performance review heatmaps
    - Corruption indicator charts
    """

    # Color schemes
    RISK_COLORS = {
        "High Risk": "#D32F2F",
        "Elevated": "#FF9800",
        "Moderate": "#FDD835",
        "Low Risk": "#4CAF50",
    }

    PARTY_COLORS = {
        "Republican": "#E9141D",
        "Democrat": "#0015BC",
        "Independent": "#808080",
        "Nonpartisan": "#FFD700",
        "Unknown": "#C0C0C0",
    }

    DEMOGRAPHIC_PALETTE = sns.color_palette("Set2", 8)

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.

        Args:
            figsize: Default figure size for matplotlib plots.
        """
        self.figsize = figsize

    # ==================== Risk Overview ====================

    def plot_risk_tier_distribution(
        self,
        summary_df: pd.DataFrame,
        title: str = "Judicial Integrity Risk Tier Distribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the distribution of judges across risk tiers.

        Args:
            summary_df: DataFrame from JudicialIntegrityAnalyzer.compute_integrity_summary().
            title: Plot title.
            save_path: Optional path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        tier_counts = summary_df["risk_tier"].value_counts()
        colors = [self.RISK_COLORS.get(t, "#808080") for t in tier_counts.index]
        axes[0].pie(
            tier_counts.values,
            labels=tier_counts.index,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0].set_title("Risk Tier Distribution")

        # Bar chart by court level
        if "level" in summary_df.columns and "risk_tier" in summary_df.columns:
            ct = pd.crosstab(summary_df["level"], summary_df["risk_tier"])
            ct.plot(
                kind="bar",
                ax=axes[1],
                color=[self.RISK_COLORS.get(c, "#808080") for c in ct.columns],
                stacked=True,
            )
            axes[1].set_title("Risk Tiers by Court Level")
            axes[1].set_xlabel("Court Level")
            axes[1].set_ylabel("Number of Judges")
            axes[1].legend(title="Risk Tier")
            axes[1].tick_params(axis="x", rotation=45)

        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved figure to %s", save_path)

        return fig

    def plot_integrity_scores(
        self,
        summary_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Judge Integrity Composite Scores",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot integrity composite scores for judges.

        Args:
            summary_df: DataFrame with integrity_composite column.
            top_n: Number of judges to show (lowest scores).
            title: Plot title.
            save_path: Optional path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        df = summary_df.nsmallest(top_n, "integrity_composite").copy()
        df = df.sort_values("integrity_composite")

        colors = [
            self.RISK_COLORS.get(str(tier), "#808080")
            for tier in df["risk_tier"]
        ]

        ax.barh(range(len(df)), df["integrity_composite"], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(
            [f"{row['name_full']}\n({row['court_name']})" for _, row in df.iterrows()],
            fontsize=8,
        )
        ax.set_xlabel("Integrity Composite Score (0-100)")
        ax.set_title(title, fontsize=14)
        ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="Midpoint")
        ax.set_xlim(0, 100)

        # Add legend
        patches = [
            mpatches.Patch(color=color, label=tier)
            for tier, color in self.RISK_COLORS.items()
        ]
        ax.legend(handles=patches, loc="lower right", title="Risk Tier")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved figure to %s", save_path)

        return fig

    # ==================== Corruption Analysis Plots ====================

    def plot_corruption_indicators(
        self,
        corruption_df: pd.DataFrame,
        title: str = "Corruption Risk Indicators",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot corruption risk indicators.

        Args:
            corruption_df: DataFrame from analyze_corruption_indicators().
            title: Plot title.
            save_path: Optional path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Distribution of corruption risk scores
        axes[0].hist(
            corruption_df["corruption_risk_score"],
            bins=20,
            color="steelblue",
            edgecolor="white",
            alpha=0.8,
        )
        axes[0].set_xlabel("Corruption Risk Score")
        axes[0].set_ylabel("Number of Judges")
        axes[0].set_title("Distribution of Corruption Risk Scores")
        axes[0].axvline(
            x=corruption_df["corruption_risk_score"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {corruption_df['corruption_risk_score'].mean():.1f}",
        )
        axes[0].legend()

        # Disciplinary action counts
        if "n_disciplinary" in corruption_df.columns:
            disc_counts = corruption_df["n_disciplinary"].value_counts().sort_index()
            axes[1].bar(
                disc_counts.index,
                disc_counts.values,
                color="coral",
                edgecolor="white",
            )
            axes[1].set_xlabel("Number of Disciplinary Actions")
            axes[1].set_ylabel("Number of Judges")
            axes[1].set_title("Disciplinary Action Distribution")

        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ==================== Sentencing Disparity Plots ====================

    def plot_sentencing_disparity(
        self,
        sentencing_df: pd.DataFrame,
        group_col: str = "defendant_race",
        outcome_col: str = "sentence_months",
        title: str = "Sentencing Outcomes by Demographic Group",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot sentencing outcome distributions by demographic group.

        Args:
            sentencing_df: Sentencing DataFrame.
            group_col: Demographic grouping column.
            outcome_col: Sentencing outcome column.
            title: Plot title.
            save_path: Optional path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Box plot
        groups = sentencing_df[group_col].unique()
        data_by_group = [
            sentencing_df.loc[sentencing_df[group_col] == g, outcome_col].dropna()
            for g in groups
        ]
        bp = axes[0].boxplot(
            data_by_group,
            labels=groups,
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], self.DEMOGRAPHIC_PALETTE):
            patch.set_facecolor(color)
        axes[0].set_xlabel(group_col.replace("_", " ").title())
        axes[0].set_ylabel(outcome_col.replace("_", " ").title())
        axes[0].set_title("Sentencing Distribution")
        axes[0].tick_params(axis="x", rotation=45)

        # Violin plot
        if len(sentencing_df) > 0:
            sns.violinplot(
                data=sentencing_df,
                x=group_col,
                y=outcome_col,
                ax=axes[1],
                palette="Set2",
                inner="quartile",
            )
            axes[1].set_title("Sentencing Distribution (Violin)")
            axes[1].tick_params(axis="x", rotation=45)

        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_judge_sentencing_comparison(
        self,
        judge_patterns_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Judge Sentencing Patterns (Z-Scores)",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot judge sentencing z-scores relative to peers.

        Args:
            judge_patterns_df: DataFrame from analyze_judge_sentencing_patterns().
            top_n: Number of outlier judges to show.
            title: Plot title.
            save_path: Optional path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Show top outliers (most extreme z-scores)
        df = judge_patterns_df.reindex(
            judge_patterns_df["sentence_zscore"].abs().nlargest(top_n).index
        ).sort_values("sentence_zscore")

        colors = [
            "#D32F2F" if z > 0 else "#1565C0"
            for z in df["sentence_zscore"]
        ]

        ax.barh(range(len(df)), df["sentence_zscore"], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["judge_id"], fontsize=8)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.axvline(x=2, color="red", linestyle="--", alpha=0.5, label="z=+2 (harsh)")
        ax.axvline(x=-2, color="blue", linestyle="--", alpha=0.5, label="z=-2 (lenient)")
        ax.set_xlabel("Sentencing Z-Score")
        ax.set_title(title, fontsize=14)
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ==================== Partisanship Plots ====================

    def plot_partisanship_by_appointer(
        self,
        partisanship_df: pd.DataFrame,
        title: str = "Judges by Appointing Party",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot distribution of judges by appointing party.

        Args:
            partisanship_df: DataFrame from analyze_partisanship_indicators().
            title: Plot title.
            save_path: Optional path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if "appointer_party" not in partisanship_df.columns:
            ax.text(0.5, 0.5, "No appointer data available", ha="center", va="center")
            return fig

        counts = partisanship_df["appointer_party"].value_counts()
        colors = [self.PARTY_COLORS.get(p, "#808080") for p in counts.index]

        ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
        ax.set_xlabel("Appointing Party")
        ax.set_ylabel("Number of Judges")
        ax.set_title(title, fontsize=14)

        for i, (party, count) in enumerate(counts.items()):
            ax.text(i, count + 0.3, str(count), ha="center", fontweight="bold")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ==================== Performance Review Plots ====================

    def plot_performance_heatmap(
        self,
        performance_df: pd.DataFrame,
        title: str = "Judicial Performance Review Scores",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot heatmap of performance review scores.

        Args:
            performance_df: Performance review DataFrame.
            title: Plot title.
            save_path: Optional path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        score_cols = [
            "overall_score",
            "legal_ability",
            "integrity",
            "temperament",
            "communication",
            "administrative",
        ]
        available_cols = [c for c in score_cols if c in performance_df.columns]

        if not available_cols or "judge_id" not in performance_df.columns:
            ax.text(0.5, 0.5, "Insufficient data for heatmap", ha="center", va="center")
            return fig

        pivot = performance_df.groupby("judge_id")[available_cols].mean()

        if len(pivot) > 0:
            sns.heatmap(
                pivot,
                cmap="RdYlGn",
                center=50,
                ax=ax,
                annot=len(pivot) < 30,
                fmt=".1f",
                xticklabels=[c.replace("_", " ").title() for c in available_cols],
            )
            ax.set_title(title, fontsize=14)
            ax.set_ylabel("Judge ID")
        else:
            ax.text(0.5, 0.5, "No performance data available", ha="center", va="center")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    # ==================== Dashboard ====================

    def create_analysis_dashboard(
        self,
        analyzer,
        output_dir: str = "output/figures",
        prefix: str = "",
    ) -> List[str]:
        """
        Create a complete dashboard of analysis visualizations.

        Args:
            analyzer: JudicialIntegrityAnalyzer instance.
            output_dir: Directory to save figures.
            prefix: Filename prefix.

        Returns:
            List of saved file paths.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved_files = []

        # Integrity summary
        summary = analyzer.compute_integrity_summary()
        if not summary.empty:
            path = f"{output_dir}/{prefix}risk_tiers.png"
            self.plot_risk_tier_distribution(summary, save_path=path)
            saved_files.append(path)

            path = f"{output_dir}/{prefix}integrity_scores.png"
            self.plot_integrity_scores(summary, save_path=path)
            saved_files.append(path)

        # Corruption indicators
        corruption = analyzer.analyze_corruption_indicators()
        if not corruption.empty:
            path = f"{output_dir}/{prefix}corruption_indicators.png"
            self.plot_corruption_indicators(corruption, save_path=path)
            saved_files.append(path)

        # Partisanship
        partisanship = analyzer.analyze_partisanship_indicators()
        if not partisanship.empty:
            path = f"{output_dir}/{prefix}partisanship.png"
            self.plot_partisanship_by_appointer(partisanship, save_path=path)
            saved_files.append(path)

        # Performance reviews
        if analyzer.performance is not None and not analyzer.performance.empty:
            path = f"{output_dir}/{prefix}performance_heatmap.png"
            self.plot_performance_heatmap(analyzer.performance, save_path=path)
            saved_files.append(path)

        # Sentencing analysis
        if analyzer.sentencing is not None and not analyzer.sentencing.empty:
            path = f"{output_dir}/{prefix}sentencing_disparity.png"
            self.plot_sentencing_disparity(analyzer.sentencing, save_path=path)
            saved_files.append(path)

            patterns = analyzer.analyze_judge_sentencing_patterns()
            if not patterns.empty:
                path = f"{output_dir}/{prefix}judge_sentencing_zscore.png"
                self.plot_judge_sentencing_comparison(patterns, save_path=path)
                saved_files.append(path)

        logger.info("Created %d visualizations in %s", len(saved_files), output_dir)
        return saved_files
