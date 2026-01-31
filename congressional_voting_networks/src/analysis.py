"""
Network Analysis Module for Congressional Voting Networks.

Provides methods for computing centrality metrics, community detection,
polarization measures, and temporal analysis of congressional networks.
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """
    Analyzes congressional voting networks using graph theory metrics.
    
    Computes:
    - Centrality measures (degree, betweenness, eigenvector, closeness)
    - Community detection (Louvain modularity)
    - Polarization metrics (assortativity, cross-party edges)
    - Network statistics (density, clustering, path lengths)
    """
    
    def __init__(self, network: nx.Graph):
        """
        Initialize analyzer with a network.
        
        Args:
            network: NetworkX Graph to analyze.
        """
        self.network = network
        self._metrics_cache = {}
    
    # ==================== Centrality Measures ====================
    
    def compute_degree_centrality(self, weighted: bool = True) -> Dict[str, float]:
        """
        Compute degree centrality for all nodes.
        
        Args:
            weighted: If True, use weighted degree (sum of edge weights).
            
        Returns:
            Dictionary mapping node IDs to centrality values.
        """
        if weighted and nx.is_weighted(self.network):
            # Weighted degree centrality
            centrality = {}
            for node in self.network.nodes():
                weighted_degree = sum(
                    d.get('weight', 1) 
                    for _, _, d in self.network.edges(node, data=True)
                )
                centrality[node] = weighted_degree
            # Normalize
            max_val = max(centrality.values()) if centrality else 1
            centrality = {k: v / max_val for k, v in centrality.items()}
        else:
            centrality = nx.degree_centrality(self.network)
        
        self._metrics_cache['degree_centrality'] = centrality
        return centrality
    
    def compute_betweenness_centrality(
        self, 
        normalized: bool = True,
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute betweenness centrality for all nodes.
        
        Args:
            normalized: If True, normalize by possible paths.
            k: If set, use k-sample approximation for large networks.
            
        Returns:
            Dictionary mapping node IDs to centrality values.
        """
        if k is not None:
            centrality = nx.betweenness_centrality(
                self.network, normalized=normalized, k=k
            )
        else:
            centrality = nx.betweenness_centrality(
                self.network, normalized=normalized
            )
        
        self._metrics_cache['betweenness_centrality'] = centrality
        return centrality
    
    def compute_eigenvector_centrality(
        self, 
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Compute eigenvector centrality for all nodes.
        
        Args:
            max_iter: Maximum iterations for power iteration.
            tol: Convergence tolerance.
            
        Returns:
            Dictionary mapping node IDs to centrality values.
        """
        try:
            centrality = nx.eigenvector_centrality(
                self.network, max_iter=max_iter, tol=tol
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed to converge, using numpy fallback")
            centrality = nx.eigenvector_centrality_numpy(self.network)
        
        self._metrics_cache['eigenvector_centrality'] = centrality
        return centrality
    
    def compute_closeness_centrality(self) -> Dict[str, float]:
        """
        Compute closeness centrality for all nodes.
        
        Returns:
            Dictionary mapping node IDs to centrality values.
        """
        centrality = nx.closeness_centrality(self.network)
        self._metrics_cache['closeness_centrality'] = centrality
        return centrality
    
    def compute_pagerank(
        self, 
        alpha: float = 0.85,
        max_iter: int = 100
    ) -> Dict[str, float]:
        """
        Compute PageRank centrality.
        
        Args:
            alpha: Damping parameter.
            max_iter: Maximum iterations.
            
        Returns:
            Dictionary mapping node IDs to PageRank values.
        """
        pr = nx.pagerank(self.network, alpha=alpha, max_iter=max_iter)
        self._metrics_cache['pagerank'] = pr
        return pr
    
    def compute_all_centralities(self) -> pd.DataFrame:
        """
        Compute all centrality measures and return as DataFrame.
        
        Returns:
            DataFrame with node IDs as index and centrality columns.
        """
        logger.info("Computing all centrality measures...")
        
        degree = self.compute_degree_centrality()
        betweenness = self.compute_betweenness_centrality()
        eigenvector = self.compute_eigenvector_centrality()
        closeness = self.compute_closeness_centrality()
        pagerank = self.compute_pagerank()
        
        df = pd.DataFrame({
            'degree': degree,
            'betweenness': betweenness,
            'eigenvector': eigenvector,
            'closeness': closeness,
            'pagerank': pagerank
        })
        
        # Add node attributes
        for node in df.index:
            attrs = self.network.nodes[node]
            for key, value in attrs.items():
                if key not in df.columns:
                    df.loc[node, key] = value
        
        return df
    
    def get_top_legislators(
        self, 
        centrality_type: str = 'degree',
        top_n: int = 10,
        by_party: bool = False
    ) -> pd.DataFrame:
        """
        Get top legislators by centrality measure.
        
        Args:
            centrality_type: Type of centrality measure.
            top_n: Number of top legislators to return.
            by_party: If True, return top N per party.
            
        Returns:
            DataFrame with top legislators.
        """
        cache_key = f'{centrality_type}_centrality'
        if cache_key not in self._metrics_cache:
            method = getattr(self, f'compute_{centrality_type}_centrality')
            method()
        
        centrality = self._metrics_cache[cache_key]
        
        df = pd.DataFrame({'centrality': centrality})
        
        # Add node attributes
        for node in df.index:
            attrs = self.network.nodes[node]
            df.loc[node, 'bioname'] = attrs.get('bioname', 'Unknown')
            df.loc[node, 'party_group'] = attrs.get('party_group', 'Unknown')
            df.loc[node, 'state_abbrev'] = attrs.get('state_abbrev', 'XX')
            df.loc[node, 'chamber'] = attrs.get('chamber', 'Unknown')
        
        df = df.sort_values('centrality', ascending=False)
        
        if by_party:
            result = df.groupby('party_group').head(top_n)
        else:
            result = df.head(top_n)
        
        return result
    
    # ==================== Community Detection ====================
    
    def detect_communities_louvain(
        self, 
        resolution: float = 1.0,
        random_state: int = 42
    ) -> Dict[str, int]:
        """
        Detect communities using Louvain modularity optimization.
        
        Args:
            resolution: Resolution parameter (higher = more communities).
            random_state: Random seed for reproducibility.
            
        Returns:
            Dictionary mapping node IDs to community labels.
        """
        communities = community_louvain.best_partition(
            self.network, 
            resolution=resolution,
            random_state=random_state
        )
        
        self._metrics_cache['communities'] = communities
        logger.info(f"Detected {len(set(communities.values()))} communities")
        return communities
    
    def get_community_summary(self) -> pd.DataFrame:
        """
        Get summary of detected communities.
        
        Returns:
            DataFrame with community statistics.
        """
        if 'communities' not in self._metrics_cache:
            self.detect_communities_louvain()
        
        communities = self._metrics_cache['communities']
        
        # Group by community
        community_data = defaultdict(list)
        for node, comm in communities.items():
            community_data[comm].append(node)
        
        summary = []
        for comm, nodes in community_data.items():
            subgraph = self.network.subgraph(nodes)
            
            # Party breakdown
            party_counts = defaultdict(int)
            for node in nodes:
                party = self.network.nodes[node].get('party_group', 'Unknown')
                party_counts[party] += 1
            
            dominant_party = max(party_counts.items(), key=lambda x: x[1])[0]
            party_purity = max(party_counts.values()) / len(nodes) if nodes else 0
            
            summary.append({
                'community': comm,
                'size': len(nodes),
                'density': nx.density(subgraph),
                'dominant_party': dominant_party,
                'party_purity': party_purity,
                'party_breakdown': dict(party_counts)
            })
        
        return pd.DataFrame(summary).sort_values('size', ascending=False)
    
    def compute_modularity(self) -> float:
        """
        Compute modularity score of current community partition.
        
        Returns:
            Modularity score.
        """
        if 'communities' not in self._metrics_cache:
            self.detect_communities_louvain()
        
        communities = self._metrics_cache['communities']
        modularity = community_louvain.modularity(communities, self.network)
        
        self._metrics_cache['modularity'] = modularity
        return modularity
    
    def compare_communities_to_parties(self) -> Dict[str, Any]:
        """
        Compare detected communities to party affiliations.
        
        Returns:
            Dictionary with comparison metrics.
        """
        if 'communities' not in self._metrics_cache:
            self.detect_communities_louvain()
        
        communities = self._metrics_cache['communities']
        
        # Get party labels
        party_labels = {
            node: self.network.nodes[node].get('party_group', 'Unknown')
            for node in self.network.nodes()
        }
        
        # Compute NMI between communities and parties
        from sklearn.metrics import normalized_mutual_info_score
        
        comm_list = [communities[n] for n in self.network.nodes()]
        party_list = [party_labels[n] for n in self.network.nodes()]
        
        nmi = normalized_mutual_info_score(party_list, comm_list)
        
        return {
            'nmi': nmi,
            'n_communities': len(set(communities.values())),
            'n_parties': len(set(party_labels.values())),
        }
    
    # ==================== Polarization Metrics ====================
    
    def compute_party_assortativity(self) -> float:
        """
        Compute assortativity coefficient by party.
        
        High assortativity indicates legislators connect mainly within party.
        
        Returns:
            Assortativity coefficient (-1 to 1).
        """
        # Get party labels
        party_labels = {
            node: self.network.nodes[node].get('party_group', 'Unknown')
            for node in self.network.nodes()
        }
        
        # Set as node attribute
        nx.set_node_attributes(self.network, party_labels, 'party_attr')
        
        assortativity = nx.attribute_assortativity_coefficient(
            self.network, 'party_attr'
        )
        
        self._metrics_cache['party_assortativity'] = assortativity
        return assortativity
    
    def compute_cross_party_edge_ratio(self) -> Dict[str, float]:
        """
        Compute ratio of cross-party edges.
        
        Returns:
            Dictionary with edge statistics.
        """
        total_edges = 0
        cross_party_edges = 0
        within_party_edges = 0
        
        for u, v in self.network.edges():
            party_u = self.network.nodes[u].get('party_group', 'Unknown')
            party_v = self.network.nodes[v].get('party_group', 'Unknown')
            
            total_edges += 1
            if party_u != party_v:
                cross_party_edges += 1
            else:
                within_party_edges += 1
        
        result = {
            'total_edges': total_edges,
            'cross_party_edges': cross_party_edges,
            'within_party_edges': within_party_edges,
            'cross_party_ratio': cross_party_edges / total_edges if total_edges > 0 else 0,
        }
        
        self._metrics_cache['edge_stats'] = result
        return result
    
    def compute_party_cohesion(self) -> Dict[str, float]:
        """
        Compute cohesion within each party (average edge weight within party).
        
        Returns:
            Dictionary mapping party to cohesion score.
        """
        party_edges = defaultdict(list)
        
        for u, v, d in self.network.edges(data=True):
            party_u = self.network.nodes[u].get('party_group', 'Unknown')
            party_v = self.network.nodes[v].get('party_group', 'Unknown')
            
            if party_u == party_v:
                weight = d.get('weight', 1.0)
                party_edges[party_u].append(weight)
        
        cohesion = {}
        for party, weights in party_edges.items():
            cohesion[party] = np.mean(weights) if weights else 0
        
        self._metrics_cache['party_cohesion'] = cohesion
        return cohesion
    
    def compute_polarization_score(self) -> float:
        """
        Compute overall polarization score.
        
        Combines assortativity and cross-party metrics.
        Higher score = more polarized.
        
        Returns:
            Polarization score (0 to 1).
        """
        assortativity = self.compute_party_assortativity()
        edge_stats = self.compute_cross_party_edge_ratio()
        
        # Normalize assortativity to [0, 1]
        norm_assortativity = (assortativity + 1) / 2
        
        # Invert cross-party ratio (less cross = more polarized)
        polarization = (norm_assortativity + (1 - edge_stats['cross_party_ratio'])) / 2
        
        self._metrics_cache['polarization_score'] = polarization
        return polarization
    
    # ==================== Network Statistics ====================
    
    def compute_network_statistics(self) -> Dict[str, Any]:
        """
        Compute basic network statistics.
        
        Returns:
            Dictionary with network metrics.
        """
        G = self.network
        
        stats = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
        }
        
        if nx.is_connected(G):
            stats['diameter'] = nx.diameter(G)
            stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            # Use largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            stats['diameter'] = nx.diameter(subgraph)
            stats['avg_shortest_path'] = nx.average_shortest_path_length(subgraph)
            stats['n_components'] = nx.number_connected_components(G)
            stats['largest_component_size'] = len(largest_cc)
        
        stats['avg_clustering'] = nx.average_clustering(G)
        stats['transitivity'] = nx.transitivity(G)
        
        # Degree statistics
        degrees = [d for _, d in G.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = max(degrees)
        stats['min_degree'] = min(degrees)
        
        self._metrics_cache['network_stats'] = stats
        return stats
    
    def get_degree_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get degree distribution.
        
        Returns:
            Tuple of (degrees, counts).
        """
        degrees = [d for _, d in self.network.degree()]
        unique, counts = np.unique(degrees, return_counts=True)
        return unique, counts
    
    # ==================== Full Analysis ====================
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete network analysis.
        
        Returns:
            Dictionary with all analysis results.
        """
        logger.info("Running full network analysis...")
        
        results = {
            'network_stats': self.compute_network_statistics(),
            'polarization_score': self.compute_polarization_score(),
            'party_assortativity': self.compute_party_assortativity(),
            'party_cohesion': self.compute_party_cohesion(),
            'edge_stats': self.compute_cross_party_edge_ratio(),
            'modularity': self.compute_modularity(),
            'community_comparison': self.compare_communities_to_parties(),
        }
        
        return results
    
    def generate_report(self) -> str:
        """
        Generate text report of analysis results.
        
        Returns:
            Formatted report string.
        """
        results = self.run_full_analysis()
        
        report = []
        report.append("=" * 60)
        report.append("CONGRESSIONAL VOTING NETWORK ANALYSIS REPORT")
        report.append("=" * 60)
        
        report.append("\n## Network Statistics")
        stats = results['network_stats']
        report.append(f"  Nodes (Legislators): {stats['n_nodes']}")
        report.append(f"  Edges (Connections): {stats['n_edges']}")
        report.append(f"  Density: {stats['density']:.4f}")
        report.append(f"  Average Degree: {stats['avg_degree']:.2f}")
        report.append(f"  Clustering Coefficient: {stats['avg_clustering']:.4f}")
        
        report.append("\n## Polarization Metrics")
        report.append(f"  Polarization Score: {results['polarization_score']:.4f}")
        report.append(f"  Party Assortativity: {results['party_assortativity']:.4f}")
        report.append(f"  Cross-Party Edge Ratio: {results['edge_stats']['cross_party_ratio']:.4f}")
        
        report.append("\n## Party Cohesion")
        for party, cohesion in results['party_cohesion'].items():
            report.append(f"  {party}: {cohesion:.4f}")
        
        report.append("\n## Community Detection")
        report.append(f"  Modularity: {results['modularity']:.4f}")
        report.append(f"  Communities Detected: {results['community_comparison']['n_communities']}")
        report.append(f"  NMI with Parties: {results['community_comparison']['nmi']:.4f}")
        
        return "\n".join(report)


def analyze_temporal_networks(
    networks: Dict[int, nx.Graph]
) -> pd.DataFrame:
    """
    Analyze a series of temporal networks and track metrics over time.
    
    Args:
        networks: Dictionary mapping Congress numbers to graphs.
        
    Returns:
        DataFrame with metrics per Congress.
    """
    results = []
    
    for congress, G in sorted(networks.items()):
        logger.info(f"Analyzing Congress {congress}...")
        
        analyzer = NetworkAnalyzer(G)
        
        try:
            row = {
                'congress': congress,
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'density': nx.density(G),
                'avg_clustering': nx.average_clustering(G),
                'polarization': analyzer.compute_polarization_score(),
                'assortativity': analyzer.compute_party_assortativity(),
                'modularity': analyzer.compute_modularity(),
            }
            
            cohesion = analyzer.compute_party_cohesion()
            row['dem_cohesion'] = cohesion.get('Democrat', np.nan)
            row['rep_cohesion'] = cohesion.get('Republican', np.nan)
            
            results.append(row)
            
        except Exception as e:
            logger.warning(f"Error analyzing Congress {congress}: {e}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create a simple test network
    G = nx.Graph()
    
    # Add some nodes with party attributes
    for i in range(10):
        G.add_node(f"dem_{i}", party_group="Democrat", bioname=f"Dem {i}")
    for i in range(10):
        G.add_node(f"rep_{i}", party_group="Republican", bioname=f"Rep {i}")
    
    # Add edges (within party more likely)
    for i in range(10):
        for j in range(i+1, 10):
            G.add_edge(f"dem_{i}", f"dem_{j}", weight=0.8 + np.random.random() * 0.2)
            G.add_edge(f"rep_{i}", f"rep_{j}", weight=0.8 + np.random.random() * 0.2)
    
    # Few cross-party edges
    for i in range(3):
        G.add_edge(f"dem_{i}", f"rep_{i}", weight=0.3 + np.random.random() * 0.2)
    
    analyzer = NetworkAnalyzer(G)
    print(analyzer.generate_report())
