"""Tests for the analysis module."""

import pytest
import numpy as np
import networkx as nx

from congressional_voting_networks.src.analysis import NetworkAnalyzer


@pytest.fixture
def sample_polarized_network():
    """Create a sample polarized network for testing."""
    G = nx.Graph()
    
    # Add Democrat nodes
    for i in range(10):
        G.add_node(f"dem_{i}", 
                   party_group="Democrat", 
                   bioname=f"Democrat {i}",
                   party_name="Democrat",
                   state_abbrev="CA",
                   chamber="House",
                   nominate_dim1=-0.5 - np.random.random() * 0.3,
                   nominate_dim2=np.random.random() * 0.5 - 0.25)
    
    # Add Republican nodes
    for i in range(10):
        G.add_node(f"rep_{i}", 
                   party_group="Republican", 
                   bioname=f"Republican {i}",
                   party_name="Republican",
                   state_abbrev="TX",
                   chamber="House",
                   nominate_dim1=0.5 + np.random.random() * 0.3,
                   nominate_dim2=np.random.random() * 0.5 - 0.25)
    
    # Add strong within-party edges
    for i in range(10):
        for j in range(i+1, 10):
            G.add_edge(f"dem_{i}", f"dem_{j}", weight=0.8 + np.random.random() * 0.2)
            G.add_edge(f"rep_{i}", f"rep_{j}", weight=0.8 + np.random.random() * 0.2)
    
    # Add few weak cross-party edges
    for i in range(3):
        G.add_edge(f"dem_{i}", f"rep_{i}", weight=0.3 + np.random.random() * 0.2)
    
    return G


@pytest.fixture
def sample_bipartisan_network():
    """Create a less polarized network for testing."""
    G = nx.Graph()
    
    # Add Democrat nodes
    for i in range(5):
        G.add_node(f"dem_{i}", party_group="Democrat", bioname=f"Democrat {i}")
    
    # Add Republican nodes
    for i in range(5):
        G.add_node(f"rep_{i}", party_group="Republican", bioname=f"Republican {i}")
    
    # Add many cross-party edges
    for i in range(5):
        for j in range(5):
            G.add_edge(f"dem_{i}", f"rep_{j}", weight=0.6 + np.random.random() * 0.2)
    
    return G


class TestNetworkAnalyzer:
    """Test cases for NetworkAnalyzer."""
    
    def test_init(self, sample_polarized_network):
        """Test analyzer initialization."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        assert analyzer.network is not None
    
    # ==================== Centrality Tests ====================
    
    def test_degree_centrality(self, sample_polarized_network):
        """Test degree centrality computation."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        centrality = analyzer.compute_degree_centrality()
        
        assert len(centrality) == 20
        assert all(0 <= v <= 1 for v in centrality.values())
    
    def test_betweenness_centrality(self, sample_polarized_network):
        """Test betweenness centrality computation."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        centrality = analyzer.compute_betweenness_centrality()
        
        assert len(centrality) == 20
        assert all(0 <= v <= 1 for v in centrality.values())
    
    def test_eigenvector_centrality(self, sample_polarized_network):
        """Test eigenvector centrality computation."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        centrality = analyzer.compute_eigenvector_centrality()
        
        assert len(centrality) == 20
        assert all(v >= 0 for v in centrality.values())
    
    def test_closeness_centrality(self, sample_polarized_network):
        """Test closeness centrality computation."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        centrality = analyzer.compute_closeness_centrality()
        
        assert len(centrality) == 20
        assert all(0 <= v <= 1 for v in centrality.values())
    
    def test_pagerank(self, sample_polarized_network):
        """Test PageRank computation."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        pr = analyzer.compute_pagerank()
        
        assert len(pr) == 20
        assert pytest.approx(sum(pr.values()), rel=0.01) == 1.0
    
    def test_compute_all_centralities(self, sample_polarized_network):
        """Test computing all centralities."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        df = analyzer.compute_all_centralities()
        
        assert len(df) == 20
        assert 'degree' in df.columns
        assert 'betweenness' in df.columns
        assert 'eigenvector' in df.columns
        assert 'closeness' in df.columns
        assert 'pagerank' in df.columns
    
    def test_get_top_legislators(self, sample_polarized_network):
        """Test getting top legislators by centrality."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        top = analyzer.get_top_legislators(centrality_type='degree', top_n=5)
        
        assert len(top) == 5
        assert 'bioname' in top.columns
        assert 'centrality' in top.columns
    
    def test_get_top_legislators_by_party(self, sample_polarized_network):
        """Test getting top legislators per party."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        top = analyzer.get_top_legislators(centrality_type='degree', top_n=3, by_party=True)
        
        # Should have 3 per party = 6 total
        assert len(top) == 6
    
    # ==================== Community Detection Tests ====================
    
    def test_detect_communities_louvain(self, sample_polarized_network):
        """Test Louvain community detection."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        communities = analyzer.detect_communities_louvain()
        
        assert len(communities) == 20
        # In a polarized network, we expect ~2 communities
        assert len(set(communities.values())) >= 2
    
    def test_get_community_summary(self, sample_polarized_network):
        """Test community summary."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        summary = analyzer.get_community_summary()
        
        assert len(summary) >= 2
        assert 'size' in summary.columns
        assert 'dominant_party' in summary.columns
        assert 'party_purity' in summary.columns
    
    def test_compute_modularity(self, sample_polarized_network):
        """Test modularity computation."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        modularity = analyzer.compute_modularity()
        
        # Modularity should be between -0.5 and 1
        assert -0.5 <= modularity <= 1
        # A polarized network should have high modularity
        assert modularity > 0.3
    
    def test_compare_communities_to_parties(self, sample_polarized_network):
        """Test community-party comparison."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        comparison = analyzer.compare_communities_to_parties()
        
        assert 'nmi' in comparison
        assert 'n_communities' in comparison
        assert 'n_parties' in comparison
        # NMI should be between 0 and 1
        assert 0 <= comparison['nmi'] <= 1
    
    # ==================== Polarization Tests ====================
    
    def test_party_assortativity_polarized(self, sample_polarized_network):
        """Test party assortativity for polarized network."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        assortativity = analyzer.compute_party_assortativity()
        
        # Polarized network should have high assortativity
        assert assortativity > 0.5
    
    def test_party_assortativity_bipartisan(self, sample_bipartisan_network):
        """Test party assortativity for bipartisan network."""
        analyzer = NetworkAnalyzer(sample_bipartisan_network)
        assortativity = analyzer.compute_party_assortativity()
        
        # Bipartisan network should have low or negative assortativity
        assert assortativity < 0.3
    
    def test_cross_party_edge_ratio(self, sample_polarized_network):
        """Test cross-party edge ratio computation."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        stats = analyzer.compute_cross_party_edge_ratio()
        
        assert 'total_edges' in stats
        assert 'cross_party_edges' in stats
        assert 'within_party_edges' in stats
        assert 'cross_party_ratio' in stats
        
        # Ratio should be between 0 and 1
        assert 0 <= stats['cross_party_ratio'] <= 1
        # Polarized network should have low cross-party ratio
        assert stats['cross_party_ratio'] < 0.2
    
    def test_party_cohesion(self, sample_polarized_network):
        """Test party cohesion computation."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        cohesion = analyzer.compute_party_cohesion()
        
        assert 'Democrat' in cohesion
        assert 'Republican' in cohesion
        # Both parties should have high cohesion in polarized network
        assert cohesion['Democrat'] > 0.7
        assert cohesion['Republican'] > 0.7
    
    def test_polarization_score(self, sample_polarized_network):
        """Test overall polarization score."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        score = analyzer.compute_polarization_score()
        
        assert 0 <= score <= 1
        # Polarized network should have high score
        assert score > 0.6
    
    # ==================== Network Statistics Tests ====================
    
    def test_network_statistics(self, sample_polarized_network):
        """Test network statistics computation."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        stats = analyzer.compute_network_statistics()
        
        assert stats['n_nodes'] == 20
        assert stats['n_edges'] > 0
        assert 0 <= stats['density'] <= 1
        assert stats['avg_degree'] > 0
        assert 0 <= stats['avg_clustering'] <= 1
    
    def test_degree_distribution(self, sample_polarized_network):
        """Test degree distribution."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        degrees, counts = analyzer.get_degree_distribution()
        
        assert len(degrees) > 0
        assert len(counts) == len(degrees)
        assert sum(counts) == 20  # Total nodes
    
    # ==================== Full Analysis Tests ====================
    
    def test_run_full_analysis(self, sample_polarized_network):
        """Test running full analysis."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        results = analyzer.run_full_analysis()
        
        assert 'network_stats' in results
        assert 'polarization_score' in results
        assert 'party_assortativity' in results
        assert 'party_cohesion' in results
        assert 'modularity' in results
    
    def test_generate_report(self, sample_polarized_network):
        """Test report generation."""
        analyzer = NetworkAnalyzer(sample_polarized_network)
        report = analyzer.generate_report()
        
        assert isinstance(report, str)
        assert 'NETWORK ANALYSIS REPORT' in report
        assert 'Polarization' in report
        assert 'Party Cohesion' in report


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
