"""Tests for the network builder module."""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import networkx as nx

from congressional_voting_networks.src.network_builder import CongressionalNetworkBuilder


@pytest.fixture
def sample_network_data():
    """Create sample data for network building."""
    # Create a simple vote matrix (5 legislators, 10 votes)
    # Democrats (0-2) vote similarly, Republicans (3-4) vote similarly
    np.random.seed(42)
    
    n_legs = 5
    n_votes = 10
    
    matrix = np.zeros((n_legs, n_votes))
    
    # Democrats tend to vote Yea (1)
    matrix[0, :] = [1, 1, 1, 1, 1, -1, -1, 1, 1, 1]
    matrix[1, :] = [1, 1, 1, 1, 1, -1, -1, 1, 1, 1]
    matrix[2, :] = [1, 1, 1, 1, -1, -1, -1, 1, 1, 1]
    
    # Republicans tend to vote Nay (-1)
    matrix[3, :] = [-1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
    matrix[4, :] = [-1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
    
    sparse_matrix = csr_matrix(matrix)
    
    legislator_ids = np.array(['dem_1', 'dem_2', 'dem_3', 'rep_1', 'rep_2'])
    
    legislator_info = pd.DataFrame({
        'member_congress_id': legislator_ids,
        'bioname': ['Democrat 1', 'Democrat 2', 'Democrat 3', 'Republican 1', 'Republican 2'],
        'party_code': [100, 100, 100, 200, 200],
        'party_name': ['Democrat', 'Democrat', 'Democrat', 'Republican', 'Republican'],
        'party_group': ['Democrat', 'Democrat', 'Democrat', 'Republican', 'Republican'],
        'state_abbrev': ['CA', 'NY', 'IL', 'TX', 'FL'],
        'chamber': ['House', 'House', 'House', 'House', 'House'],
        'congress': [117, 117, 117, 117, 117],
        'nominate_dim1': [-0.5, -0.4, -0.3, 0.5, 0.6],
        'nominate_dim2': [0.1, 0.2, 0.1, -0.1, -0.2],
    })
    
    return sparse_matrix, legislator_ids, legislator_info


class TestCongressionalNetworkBuilder:
    """Test cases for CongressionalNetworkBuilder."""
    
    def test_init(self, sample_network_data):
        """Test builder initialization."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        
        assert builder.vote_matrix is not None
        assert len(builder.legislator_ids) == 5
        assert builder.legislator_info is not None
    
    def test_compute_similarity_matrix_cosine(self, sample_network_data):
        """Test cosine similarity computation."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        
        sim_matrix = builder.compute_similarity_matrix(method='cosine')
        
        assert sim_matrix.shape == (5, 5)
        # Diagonal should be 1 (self-similarity)
        np.testing.assert_array_almost_equal(np.diag(sim_matrix), np.ones(5))
        # Democrats should be similar to each other
        assert sim_matrix[0, 1] > 0.8
        # Democrats and Republicans should be dissimilar
        assert sim_matrix[0, 3] < 0.5
    
    def test_compute_similarity_matrix_agreement(self, sample_network_data):
        """Test agreement similarity computation."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        
        sim_matrix = builder.compute_similarity_matrix(method='agreement')
        
        assert sim_matrix.shape == (5, 5)
        # All values should be between 0 and 1
        assert np.all(sim_matrix >= 0)
        assert np.all(sim_matrix <= 1)
    
    def test_build_similarity_network(self, sample_network_data):
        """Test similarity network construction."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        
        G = builder.build_similarity_network(similarity_threshold=0.5)
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 5
        assert G.number_of_edges() > 0
        
        # Check node attributes
        node = list(G.nodes())[0]
        assert 'party_group' in G.nodes[node]
        assert 'bioname' in G.nodes[node]
    
    def test_network_edge_weights(self, sample_network_data):
        """Test that edges have proper weights."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        
        G = builder.build_similarity_network(similarity_threshold=0.0)
        
        for u, v, d in G.edges(data=True):
            assert 'weight' in d
            assert 0 <= d['weight'] <= 1
    
    def test_party_members_connected(self, sample_network_data):
        """Test that same-party members are connected."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        
        G = builder.build_similarity_network(similarity_threshold=0.7)
        
        # Democrats should be connected
        assert G.has_edge('dem_1', 'dem_2')
        # Republicans should be connected
        assert G.has_edge('rep_1', 'rep_2')
    
    def test_build_bipartite_network(self, sample_network_data):
        """Test bipartite network construction."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        
        B = builder.build_bipartite_network()
        
        assert isinstance(B, nx.Graph)
        # Should have legislators + rollcalls nodes
        assert B.number_of_nodes() == 5 + 10
        
        # Check bipartite attribute
        leg_nodes = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 0]
        roll_nodes = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 1]
        assert len(leg_nodes) == 5
        assert len(roll_nodes) == 10
    
    def test_build_party_network(self, sample_network_data):
        """Test party-level network construction."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        builder.compute_similarity_matrix()
        
        G = builder.build_party_network()
        
        assert isinstance(G, nx.Graph)
        # Should have 2 party nodes
        assert G.number_of_nodes() == 2
        assert 'Democrat' in G.nodes()
        assert 'Republican' in G.nodes()
    
    def test_get_subgraph_by_party(self, sample_network_data):
        """Test extracting subgraph by party."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        builder.build_similarity_network(similarity_threshold=0.5)
        
        dem_subgraph = builder.get_subgraph_by_party(['Democrat'])
        
        assert dem_subgraph.number_of_nodes() == 3
        for node in dem_subgraph.nodes():
            assert dem_subgraph.nodes[node]['party_group'] == 'Democrat'
    
    def test_get_subgraph_by_chamber(self, sample_network_data):
        """Test extracting subgraph by chamber."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        builder.build_similarity_network(similarity_threshold=0.5)
        
        house_subgraph = builder.get_subgraph_by_chamber('House')
        
        # All nodes are House in our test data
        assert house_subgraph.number_of_nodes() == 5
    
    def test_top_k_edges(self, sample_network_data):
        """Test network with top k edges per node."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        
        G = builder.build_similarity_network(
            similarity_threshold=0.0,
            top_k_edges=2
        )
        
        # Each node should have at most 2 edges (but edges are shared)
        assert G.number_of_edges() <= 5 * 2


class TestNetworkIO:
    """Test network save/load functionality."""
    
    def test_save_and_load_graphml(self, sample_network_data, tmp_path):
        """Test saving and loading GraphML format."""
        matrix, leg_ids, leg_info = sample_network_data
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        G = builder.build_similarity_network(similarity_threshold=0.5)
        
        # Save
        filepath = tmp_path / "test_network.graphml"
        builder.save_network(str(filepath), format='graphml')
        
        # Load
        G_loaded = CongressionalNetworkBuilder.load_network(str(filepath), format='graphml')
        
        assert G_loaded.number_of_nodes() == G.number_of_nodes()
        assert G_loaded.number_of_edges() == G.number_of_edges()


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
