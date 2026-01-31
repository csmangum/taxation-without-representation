"""
Visualization Module for Congressional Voting Networks.

Provides static and interactive visualizations for network analysis results.
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Optional interactive visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)


class NetworkVisualizer:
    """
    Creates visualizations for congressional voting networks.
    
    Supports:
    - Static network plots (matplotlib)
    - Interactive network plots (plotly)
    - Time series of metrics
    - Heatmaps and distributions
    """
    
    # Party color scheme
    PARTY_COLORS = {
        'Democrat': '#0015BC',      # Blue
        'Republican': '#E9141D',    # Red
        'Independent': '#808080',   # Gray
        'Other': '#FFD700',         # Gold
        'Unknown': '#C0C0C0',       # Silver
    }
    
    def __init__(
        self,
        network: Optional[nx.Graph] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Initialize visualizer.
        
        Args:
            network: Optional NetworkX graph to visualize.
            figsize: Default figure size for matplotlib plots.
        """
        self.network = network
        self.figsize = figsize
        self._pos = None
    
    def set_network(self, network: nx.Graph):
        """Set or update the network to visualize."""
        self.network = network
        self._pos = None
    
    def _compute_layout(
        self,
        layout: str = "spring",
        seed: int = 42
    ) -> Dict:
        """
        Compute node positions for the network.
        
        Args:
            layout: Layout algorithm - "spring", "kamada_kawai", "spectral", "circular".
            seed: Random seed for reproducible layouts.
            
        Returns:
            Dictionary mapping nodes to (x, y) positions.
        """
        if self.network is None:
            raise ValueError("No network set")
        
        if layout == "spring":
            pos = nx.spring_layout(self.network, seed=seed, k=1/np.sqrt(len(self.network)))
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.network)
        elif layout == "spectral":
            pos = nx.spectral_layout(self.network)
        elif layout == "circular":
            pos = nx.circular_layout(self.network)
        elif layout == "party_split":
            pos = self._compute_party_split_layout()
        else:
            raise ValueError(f"Unknown layout: {layout}")
        
        self._pos = pos
        return pos
    
    def _compute_party_split_layout(self) -> Dict:
        """Compute layout with parties on opposite sides."""
        parties = {}
        for node in self.network.nodes():
            party = self.network.nodes[node].get('party_group', 'Other')
            if party not in parties:
                parties[party] = []
            parties[party].append(node)
        
        pos = {}
        party_positions = {
            'Democrat': -1,
            'Republican': 1,
            'Independent': 0,
            'Other': 0,
        }
        
        for party, nodes in parties.items():
            x_base = party_positions.get(party, 0)
            n = len(nodes)
            
            for i, node in enumerate(nodes):
                y = (i - n/2) / max(n, 1) * 2
                x = x_base + np.random.uniform(-0.2, 0.2)
                pos[node] = (x, y)
        
        return pos
    
    def _get_node_colors(self, color_by: str = "party") -> List:
        """Get colors for nodes based on attribute."""
        colors = []
        
        for node in self.network.nodes():
            if color_by == "party":
                party = self.network.nodes[node].get('party_group', 'Unknown')
                colors.append(self.PARTY_COLORS.get(party, '#808080'))
            elif color_by == "nominate":
                score = self.network.nodes[node].get('nominate_dim1', 0)
                if pd.isna(score):
                    score = 0
                # Map [-1, 1] to blue-red colormap
                colors.append(plt.cm.coolwarm((score + 1) / 2))
            elif color_by == "community":
                comm = self.network.nodes[node].get('community', 0)
                colors.append(plt.cm.tab20(comm % 20))
            else:
                colors.append('#808080')
        
        return colors
    
    def plot_network(
        self,
        layout: str = "spring",
        color_by: str = "party",
        node_size_by: Optional[str] = None,
        title: str = "Congressional Voting Network",
        show_labels: bool = False,
        edge_alpha: float = 0.3,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Create static network visualization.
        
        Args:
            layout: Layout algorithm.
            color_by: Node coloring - "party", "nominate", "community".
            node_size_by: Optional attribute for node sizing.
            title: Plot title.
            show_labels: Whether to show node labels.
            edge_alpha: Edge transparency.
            save_path: Optional path to save figure.
            ax: Optional matplotlib axes.
            
        Returns:
            Matplotlib figure.
        """
        if self.network is None:
            raise ValueError("No network set")
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        else:
            fig = ax.get_figure()
        
        # Compute layout
        pos = self._compute_layout(layout)
        
        # Get node colors
        node_colors = self._get_node_colors(color_by)
        
        # Get node sizes
        if node_size_by:
            sizes = []
            for node in self.network.nodes():
                val = self.network.nodes[node].get(node_size_by, 1)
                sizes.append(val * 100 + 50)
        else:
            sizes = [100] * len(self.network.nodes())
        
        # Get edge weights for width
        edge_weights = []
        for u, v, d in self.network.edges(data=True):
            edge_weights.append(d.get('weight', 1.0))
        
        # Normalize edge weights for visualization
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [w / max_weight * 2 for w in edge_weights]
        else:
            edge_widths = [1] * len(list(self.network.edges()))
        
        # Draw network
        nx.draw_networkx_edges(
            self.network, pos, ax=ax,
            alpha=edge_alpha,
            width=edge_widths,
            edge_color='gray'
        )
        
        nx.draw_networkx_nodes(
            self.network, pos, ax=ax,
            node_color=node_colors,
            node_size=sizes,
            alpha=0.8
        )
        
        if show_labels:
            labels = {n: self.network.nodes[n].get('bioname', n)[:15] 
                     for n in self.network.nodes()}
            nx.draw_networkx_labels(
                self.network, pos, labels=labels, ax=ax,
                font_size=6
            )
        
        # Add legend for parties
        if color_by == "party":
            patches = [
                mpatches.Patch(color=color, label=party)
                for party, color in self.PARTY_COLORS.items()
                if any(self.network.nodes[n].get('party_group') == party 
                      for n in self.network.nodes())
            ]
            ax.legend(handles=patches, loc='upper left')
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_network_interactive(
        self,
        layout: str = "spring",
        color_by: str = "party",
        title: str = "Congressional Voting Network"
    ):
        """
        Create interactive network visualization using Plotly.
        
        Args:
            layout: Layout algorithm.
            color_by: Node coloring attribute.
            title: Plot title.
            
        Returns:
            Plotly figure.
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for interactive plots")
        
        if self.network is None:
            raise ValueError("No network set")
        
        pos = self._compute_layout(layout)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for u, v in self.network.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = [pos[n][0] for n in self.network.nodes()]
        node_y = [pos[n][1] for n in self.network.nodes()]
        
        node_colors = self._get_node_colors(color_by)
        
        node_text = []
        for node in self.network.nodes():
            attrs = self.network.nodes[node]
            text = f"Name: {attrs.get('bioname', 'Unknown')}<br>"
            text += f"Party: {attrs.get('party_group', 'Unknown')}<br>"
            text += f"State: {attrs.get('state_abbrev', 'XX')}<br>"
            text += f"Chamber: {attrs.get('chamber', 'Unknown')}"
            node_text.append(text)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_colors,
                size=10,
                line=dict(width=1, color='white')
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        return fig
    
    def plot_polarization_over_time(
        self,
        temporal_data: pd.DataFrame,
        metrics: List[str] = ['polarization', 'assortativity'],
        title: str = "Congressional Polarization Over Time",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot polarization metrics over time.
        
        Args:
            temporal_data: DataFrame with 'congress' column and metric columns.
            metrics: List of metrics to plot.
            title: Plot title.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for metric in metrics:
            if metric in temporal_data.columns:
                ax.plot(
                    temporal_data['congress'],
                    temporal_data[metric],
                    marker='o',
                    label=metric.replace('_', ' ').title(),
                    linewidth=2
                )
        
        ax.set_xlabel('Congress Number', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_party_cohesion_over_time(
        self,
        temporal_data: pd.DataFrame,
        title: str = "Party Cohesion Over Time",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot party cohesion metrics over time.
        
        Args:
            temporal_data: DataFrame with cohesion columns.
            title: Plot title.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'dem_cohesion' in temporal_data.columns:
            ax.plot(
                temporal_data['congress'],
                temporal_data['dem_cohesion'],
                color=self.PARTY_COLORS['Democrat'],
                marker='o',
                label='Democrat',
                linewidth=2
            )
        
        if 'rep_cohesion' in temporal_data.columns:
            ax.plot(
                temporal_data['congress'],
                temporal_data['rep_cohesion'],
                color=self.PARTY_COLORS['Republican'],
                marker='o',
                label='Republican',
                linewidth=2
            )
        
        ax.set_xlabel('Congress Number', fontsize=12)
        ax.set_ylabel('Cohesion Score', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_centrality_distribution(
        self,
        centrality_df: pd.DataFrame,
        centrality_type: str = 'degree',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of centrality by party.
        
        Args:
            centrality_df: DataFrame from NetworkAnalyzer.compute_all_centralities().
            centrality_type: Which centrality to plot.
            title: Plot title.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if title is None:
            title = f"{centrality_type.title()} Centrality Distribution by Party"
        
        for party in ['Democrat', 'Republican', 'Independent']:
            if 'party_group' in centrality_df.columns:
                data = centrality_df[centrality_df['party_group'] == party][centrality_type]
                if len(data) > 0:
                    ax.hist(
                        data,
                        bins=20,
                        alpha=0.5,
                        color=self.PARTY_COLORS[party],
                        label=party
                    )
        
        ax.set_xlabel(f'{centrality_type.title()} Centrality', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_similarity_heatmap(
        self,
        similarity_matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        party_labels: Optional[List[str]] = None,
        title: str = "Legislator Voting Similarity",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of voting similarity matrix.
        
        Args:
            similarity_matrix: Square similarity matrix.
            labels: Optional node labels.
            party_labels: Optional party labels for coloring.
            title: Plot title.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Sort by party if labels provided
        if party_labels is not None:
            order = np.argsort(party_labels)
            similarity_matrix = similarity_matrix[order][:, order]
            party_labels = [party_labels[i] for i in order]
        
        sns.heatmap(
            similarity_matrix,
            cmap='coolwarm',
            center=0.5,
            ax=ax,
            xticklabels=False,
            yticklabels=False
        )
        
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_community_structure(
        self,
        communities: Dict[Any, int],
        title: str = "Community Structure",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot network colored by detected communities.
        
        Args:
            communities: Dictionary mapping nodes to community IDs.
            title: Plot title.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        # Set community as node attribute
        nx.set_node_attributes(self.network, communities, 'community')
        
        return self.plot_network(
            color_by='community',
            title=title,
            save_path=save_path
        )
    
    def plot_degree_distribution(
        self,
        degrees: np.ndarray,
        counts: np.ndarray,
        title: str = "Degree Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot degree distribution.
        
        Args:
            degrees: Array of degree values.
            counts: Array of counts for each degree.
            title: Plot title.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Linear scale
        axes[0].bar(degrees, counts, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Degree', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title(f'{title} (Linear)', fontsize=14)
        
        # Log-log scale
        axes[1].scatter(degrees, counts, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Degree (log)', fontsize=12)
        axes[1].set_ylabel('Count (log)', fontsize=12)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_title(f'{title} (Log-Log)', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_nominate_scatter(
        self,
        color_by: str = "party",
        title: str = "DW-NOMINATE Ideological Positions",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot legislators in DW-NOMINATE ideological space.
        
        Args:
            color_by: Coloring attribute.
            title: Plot title.
            save_path: Optional path to save figure.
            
        Returns:
            Matplotlib figure.
        """
        if self.network is None:
            raise ValueError("No network set")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for node in self.network.nodes():
            attrs = self.network.nodes[node]
            x = attrs.get('nominate_dim1', np.nan)
            y = attrs.get('nominate_dim2', np.nan)
            
            if not (pd.isna(x) or pd.isna(y)):
                party = attrs.get('party_group', 'Unknown')
                color = self.PARTY_COLORS.get(party, '#808080')
                ax.scatter(x, y, c=color, alpha=0.6, s=50)
        
        # Add legend
        patches = [
            mpatches.Patch(color=color, label=party)
            for party, color in self.PARTY_COLORS.items()
            if any(self.network.nodes[n].get('party_group') == party 
                  for n in self.network.nodes())
        ]
        ax.legend(handles=patches, loc='upper left')
        
        ax.set_xlabel('DW-NOMINATE Dimension 1 (Liberal-Conservative)', fontsize=12)
        ax.set_ylabel('DW-NOMINATE Dimension 2', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_analysis_dashboard(
        self,
        analyzer,
        temporal_data: Optional[pd.DataFrame] = None,
        output_dir: str = "output/figures",
        prefix: str = ""
    ) -> List[str]:
        """
        Create a complete dashboard of analysis visualizations.
        
        Args:
            analyzer: NetworkAnalyzer instance.
            temporal_data: Optional temporal analysis data.
            output_dir: Directory to save figures.
            prefix: Filename prefix.
            
        Returns:
            List of saved file paths.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved_files = []
        
        # Network visualization
        path = f"{output_dir}/{prefix}network.png"
        self.plot_network(save_path=path)
        saved_files.append(path)
        
        # Party-split layout
        path = f"{output_dir}/{prefix}network_party_split.png"
        self.plot_network(layout="party_split", save_path=path, title="Network by Party")
        saved_files.append(path)
        
        # Community structure
        communities = analyzer.detect_communities_louvain()
        path = f"{output_dir}/{prefix}communities.png"
        self.plot_community_structure(communities, save_path=path)
        saved_files.append(path)
        
        # Centrality distributions
        centralities = analyzer.compute_all_centralities()
        for ctype in ['degree', 'betweenness', 'eigenvector']:
            path = f"{output_dir}/{prefix}centrality_{ctype}.png"
            self.plot_centrality_distribution(centralities, ctype, save_path=path)
            saved_files.append(path)
        
        # Degree distribution
        degrees, counts = analyzer.get_degree_distribution()
        path = f"{output_dir}/{prefix}degree_distribution.png"
        self.plot_degree_distribution(degrees, counts, save_path=path)
        saved_files.append(path)
        
        # NOMINATE scatter
        path = f"{output_dir}/{prefix}nominate_positions.png"
        self.plot_nominate_scatter(save_path=path)
        saved_files.append(path)
        
        # Temporal plots if data available
        if temporal_data is not None:
            path = f"{output_dir}/{prefix}polarization_trend.png"
            self.plot_polarization_over_time(temporal_data, save_path=path)
            saved_files.append(path)
            
            path = f"{output_dir}/{prefix}cohesion_trend.png"
            self.plot_party_cohesion_over_time(temporal_data, save_path=path)
            saved_files.append(path)
        
        logger.info(f"Created {len(saved_files)} visualizations in {output_dir}")
        return saved_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example with synthetic network
    G = nx.Graph()
    
    # Add nodes
    for i in range(20):
        G.add_node(f"dem_{i}", party_group="Democrat", bioname=f"Democrat {i}",
                   nominate_dim1=-0.5 - np.random.random() * 0.3,
                   nominate_dim2=np.random.random() * 0.5 - 0.25)
    for i in range(20):
        G.add_node(f"rep_{i}", party_group="Republican", bioname=f"Republican {i}",
                   nominate_dim1=0.5 + np.random.random() * 0.3,
                   nominate_dim2=np.random.random() * 0.5 - 0.25)
    
    # Add edges
    for i in range(20):
        for j in range(i+1, 20):
            G.add_edge(f"dem_{i}", f"dem_{j}", weight=0.7 + np.random.random() * 0.3)
            G.add_edge(f"rep_{i}", f"rep_{j}", weight=0.7 + np.random.random() * 0.3)
    
    for i in range(5):
        G.add_edge(f"dem_{i}", f"rep_{i}", weight=0.3 + np.random.random() * 0.2)
    
    viz = NetworkVisualizer(G)
    viz.plot_network(title="Test Network")
    viz.plot_nominate_scatter()
    plt.show()
