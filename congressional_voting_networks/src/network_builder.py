"""
Network Construction Module for Congressional Voting Networks.

Builds NetworkX graphs from preprocessed voting data, with various
network types including co-voting similarity networks and bipartite graphs.
"""

import logging
from typing import Optional, Dict, List, Tuple, Union
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

logger = logging.getLogger(__name__)


class CongressionalNetworkBuilder:
    """
    Constructs network graphs from congressional voting data.
    
    Supports multiple network types:
    - Co-voting similarity networks (weighted, legislators as nodes)
    - Bipartite legislator-rollcall networks
    - Party-aggregated networks
    - Temporal snapshots
    """
    
    def __init__(
        self,
        vote_matrix: csr_matrix,
        legislator_ids: np.ndarray,
        legislator_info: pd.DataFrame
    ):
        """
        Initialize network builder.
        
        Args:
            vote_matrix: Sparse matrix (legislators x rollcalls) with vote values.
            legislator_ids: Array of legislator identifiers.
            legislator_info: DataFrame with legislator attributes (party, state, etc.).
        """
        self.vote_matrix = vote_matrix
        self.legislator_ids = legislator_ids
        self.legislator_info = legislator_info
        
        # Ensure legislator_info is indexed correctly
        if 'member_congress_id' in legislator_info.columns:
            self.legislator_info = legislator_info.set_index('member_congress_id')
        
        self.similarity_matrix = None
        self.network = None
    
    def compute_similarity_matrix(
        self,
        method: str = "cosine",
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute pairwise similarity between legislators based on voting patterns.
        
        Args:
            method: Similarity method - "cosine", "agreement", or "correlation".
            normalize: Whether to normalize output to [0, 1].
            
        Returns:
            Square similarity matrix (n_legislators x n_legislators).
        """
        logger.info(f"Computing {method} similarity matrix...")
        
        if method == "cosine":
            # Cosine similarity handles sparse matrices efficiently
            sim_matrix = cosine_similarity(self.vote_matrix)
        
        elif method == "agreement":
            # Agreement rate: proportion of matching votes (excluding abstains)
            sim_matrix = self._compute_agreement_matrix()
        
        elif method == "correlation":
            # Pearson correlation
            dense_matrix = self.vote_matrix.toarray()
            sim_matrix = np.corrcoef(dense_matrix)
            # Handle NaN from constant rows
            sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        if normalize and method != "agreement":
            # Normalize to [0, 1] range
            sim_matrix = (sim_matrix + 1) / 2
        
        self.similarity_matrix = sim_matrix
        logger.info(f"Similarity matrix shape: {sim_matrix.shape}")
        return sim_matrix
    
    def _compute_agreement_matrix(self) -> np.ndarray:
        """Compute agreement rate matrix (proportion of same votes)."""
        n = self.vote_matrix.shape[0]
        matrix = self.vote_matrix.toarray()
        
        agreement = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                votes_i = matrix[i]
                votes_j = matrix[j]
                
                # Only count votes where both legislators voted
                mask = (votes_i != 0) & (votes_j != 0)
                if mask.sum() > 0:
                    agree = (votes_i[mask] == votes_j[mask]).sum()
                    agreement[i, j] = agree / mask.sum()
                    agreement[j, i] = agreement[i, j]
                else:
                    agreement[i, j] = 0
                    agreement[j, i] = 0
        
        np.fill_diagonal(agreement, 1.0)
        return agreement
    
    def build_similarity_network(
        self,
        similarity_threshold: float = 0.5,
        method: str = "cosine",
        include_all_nodes: bool = True,
        top_k_edges: Optional[int] = None
    ) -> nx.Graph:
        """
        Build weighted undirected network based on voting similarity.
        
        Args:
            similarity_threshold: Minimum similarity to create an edge.
            method: Similarity computation method.
            include_all_nodes: If True, include isolates (no edges above threshold).
            top_k_edges: If set, only keep top k edges per node.
            
        Returns:
            NetworkX Graph with weighted edges.
        """
        if self.similarity_matrix is None or True:  # Always recompute for now
            self.compute_similarity_matrix(method=method)
        
        G = nx.Graph()
        
        # Add nodes with attributes
        for i, leg_id in enumerate(self.legislator_ids):
            node_attrs = self._get_node_attributes(leg_id)
            G.add_node(leg_id, **node_attrs)
        
        # Add edges based on similarity
        n = len(self.legislator_ids)
        edge_count = 0
        
        for i in range(n):
            if top_k_edges:
                # Get top k neighbors for this node
                similarities = self.similarity_matrix[i].copy()
                similarities[i] = -1  # Exclude self
                top_indices = np.argsort(similarities)[-top_k_edges:]
                
                for j in top_indices:
                    if similarities[j] >= similarity_threshold:
                        leg_i = self.legislator_ids[i]
                        leg_j = self.legislator_ids[j]
                        if not G.has_edge(leg_i, leg_j):
                            G.add_edge(leg_i, leg_j, weight=similarities[j])
                            edge_count += 1
            else:
                for j in range(i + 1, n):
                    sim = self.similarity_matrix[i, j]
                    if sim >= similarity_threshold:
                        leg_i = self.legislator_ids[i]
                        leg_j = self.legislator_ids[j]
                        G.add_edge(leg_i, leg_j, weight=sim)
                        edge_count += 1
        
        # Remove isolates if not wanted
        if not include_all_nodes:
            isolates = list(nx.isolates(G))
            G.remove_nodes_from(isolates)
            logger.info(f"Removed {len(isolates)} isolated nodes")
        
        self.network = G
        logger.info(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _get_node_attributes(self, leg_id: str) -> Dict:
        """Get node attributes for a legislator."""
        if leg_id not in self.legislator_info.index:
            return {'id': str(leg_id)}
        
        row = self.legislator_info.loc[leg_id]
        
        # Helper to convert numpy types to native Python types
        def to_native(val, default):
            if pd.isna(val):
                return default
            if hasattr(val, 'item'):
                return val.item()
            return val
        
        attrs = {
            'id': str(leg_id),
            'bioname': str(to_native(row.get('bioname', 'Unknown'), 'Unknown')),
            'party_code': int(to_native(row.get('party_code', 0), 0)),
            'party_name': str(to_native(row.get('party_name', 'Unknown'), 'Unknown')),
            'party_group': str(to_native(row.get('party_group', 'Other'), 'Other')),
            'state_abbrev': str(to_native(row.get('state_abbrev', 'XX'), 'XX')),
            'chamber': str(to_native(row.get('chamber', 'Unknown'), 'Unknown')),
            'congress': int(to_native(row.get('congress', 0), 0)),
            'nominate_dim1': float(to_native(row.get('nominate_dim1', 0.0), 0.0)),
            'nominate_dim2': float(to_native(row.get('nominate_dim2', 0.0), 0.0)),
        }
        
        return attrs
    
    def build_bipartite_network(self) -> nx.Graph:
        """
        Build bipartite network with legislators and rollcalls as nodes.
        
        Returns:
            NetworkX bipartite Graph.
        """
        B = nx.Graph()
        
        # Add legislator nodes (bipartite=0)
        for leg_id in self.legislator_ids:
            attrs = self._get_node_attributes(leg_id)
            attrs['bipartite'] = 0
            B.add_node(leg_id, **attrs)
        
        # Add rollcall nodes (bipartite=1)
        n_rollcalls = self.vote_matrix.shape[1]
        for j in range(n_rollcalls):
            rollcall_id = f"rollcall_{j}"
            B.add_node(rollcall_id, bipartite=1)
        
        # Add edges for each vote
        matrix = self.vote_matrix.tocoo()
        for i, j, v in zip(matrix.row, matrix.col, matrix.data):
            if v != 0:  # Only add edges for actual votes
                leg_id = self.legislator_ids[i]
                rollcall_id = f"rollcall_{j}"
                B.add_edge(leg_id, rollcall_id, vote=int(v))
        
        logger.info(f"Built bipartite network: {len(self.legislator_ids)} legislators, "
                   f"{n_rollcalls} rollcalls, {B.number_of_edges()} edges")
        return B
    
    def build_party_network(
        self,
        similarity_threshold: float = 0.0
    ) -> nx.Graph:
        """
        Build network aggregated at party level.
        
        Args:
            similarity_threshold: Minimum similarity for edges.
            
        Returns:
            NetworkX Graph with parties as nodes.
        """
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        # Group legislators by party
        parties = {}
        for i, leg_id in enumerate(self.legislator_ids):
            if leg_id in self.legislator_info.index:
                party = self.legislator_info.loc[leg_id, 'party_group']
            else:
                party = 'Unknown'
            
            if party not in parties:
                parties[party] = []
            parties[party].append(i)
        
        G = nx.Graph()
        
        # Add party nodes
        for party, indices in parties.items():
            G.add_node(party, size=len(indices))
        
        # Compute inter-party average similarities
        party_list = list(parties.keys())
        for i, party_i in enumerate(party_list):
            for j, party_j in enumerate(party_list):
                if i < j:
                    indices_i = parties[party_i]
                    indices_j = parties[party_j]
                    
                    # Average similarity between parties
                    similarities = []
                    for ii in indices_i:
                        for jj in indices_j:
                            similarities.append(self.similarity_matrix[ii, jj])
                    
                    avg_sim = np.mean(similarities) if similarities else 0
                    
                    if avg_sim >= similarity_threshold:
                        G.add_edge(party_i, party_j, weight=avg_sim)
        
        logger.info(f"Built party network with {G.number_of_nodes()} parties")
        return G
    
    def project_bipartite_network(
        self,
        bipartite_graph: nx.Graph,
        project_to: str = "legislators"
    ) -> nx.Graph:
        """
        Project bipartite network to single-mode network.
        
        Args:
            bipartite_graph: Bipartite graph from build_bipartite_network.
            project_to: "legislators" or "rollcalls".
            
        Returns:
            Projected single-mode NetworkX Graph.
        """
        # Get the two node sets
        leg_nodes = {n for n, d in bipartite_graph.nodes(data=True) if d.get('bipartite') == 0}
        roll_nodes = {n for n, d in bipartite_graph.nodes(data=True) if d.get('bipartite') == 1}
        
        if project_to == "legislators":
            nodes = leg_nodes
        else:
            nodes = roll_nodes
        
        # Use NetworkX bipartite projection (weighted)
        from networkx.algorithms import bipartite
        projected = bipartite.weighted_projected_graph(bipartite_graph, nodes)
        
        logger.info(f"Projected bipartite to {project_to}: {projected.number_of_nodes()} nodes, "
                   f"{projected.number_of_edges()} edges")
        return projected
    
    def get_subgraph_by_party(
        self,
        parties: List[str]
    ) -> nx.Graph:
        """
        Extract subgraph containing only specified parties.
        
        Args:
            parties: List of party names/groups to include.
            
        Returns:
            Subgraph containing only legislators from specified parties.
        """
        if self.network is None:
            raise ValueError("Network not built. Call build_similarity_network first.")
        
        nodes_to_keep = [
            n for n, d in self.network.nodes(data=True)
            if d.get('party_group') in parties or d.get('party_name') in parties
        ]
        
        return self.network.subgraph(nodes_to_keep).copy()
    
    def get_subgraph_by_chamber(
        self,
        chamber: str
    ) -> nx.Graph:
        """
        Extract subgraph containing only specified chamber.
        
        Args:
            chamber: "House" or "Senate".
            
        Returns:
            Subgraph containing only legislators from specified chamber.
        """
        if self.network is None:
            raise ValueError("Network not built. Call build_similarity_network first.")
        
        nodes_to_keep = [
            n for n, d in self.network.nodes(data=True)
            if d.get('chamber') == chamber
        ]
        
        return self.network.subgraph(nodes_to_keep).copy()
    
    def save_network(
        self,
        filepath: str,
        format: str = "graphml"
    ):
        """
        Save network to file.
        
        Args:
            filepath: Output file path.
            format: Output format - "graphml", "gexf", "adjlist", "edgelist".
        """
        if self.network is None:
            raise ValueError("Network not built. Call build_similarity_network first.")
        
        if format == "graphml":
            nx.write_graphml(self.network, filepath)
        elif format == "gexf":
            nx.write_gexf(self.network, filepath)
        elif format == "adjlist":
            nx.write_adjlist(self.network, filepath)
        elif format == "edgelist":
            nx.write_weighted_edgelist(self.network, filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved network to {filepath}")
    
    @staticmethod
    def load_network(
        filepath: str,
        format: str = "graphml"
    ) -> nx.Graph:
        """
        Load network from file.
        
        Args:
            filepath: Input file path.
            format: Input format - "graphml", "gexf", "adjlist", "edgelist".
            
        Returns:
            NetworkX Graph.
        """
        if format == "graphml":
            G = nx.read_graphml(filepath)
        elif format == "gexf":
            G = nx.read_gexf(filepath)
        elif format == "adjlist":
            G = nx.read_adjlist(filepath)
        elif format == "edgelist":
            G = nx.read_weighted_edgelist(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return G


def build_congress_network(
    vote_matrix: csr_matrix,
    legislator_ids: np.ndarray,
    legislator_info: pd.DataFrame,
    similarity_threshold: float = 0.5,
    method: str = "cosine"
) -> nx.Graph:
    """
    Convenience function to build a congressional network.
    
    Args:
        vote_matrix: Sparse vote matrix.
        legislator_ids: Array of legislator IDs.
        legislator_info: DataFrame with legislator attributes.
        similarity_threshold: Edge threshold.
        method: Similarity method.
        
    Returns:
        NetworkX Graph.
    """
    builder = CongressionalNetworkBuilder(vote_matrix, legislator_ids, legislator_info)
    return builder.build_similarity_network(
        similarity_threshold=similarity_threshold,
        method=method
    )


def build_temporal_networks(
    preprocessor,
    congress_range: Tuple[int, int],
    similarity_threshold: float = 0.5,
    chamber: Optional[str] = None
) -> Dict[int, nx.Graph]:
    """
    Build networks for each Congress in a range.
    
    Args:
        preprocessor: VoteDataPreprocessor instance.
        congress_range: Tuple of (start, end) Congress numbers.
        similarity_threshold: Edge threshold.
        chamber: Optional chamber filter.
        
    Returns:
        Dictionary mapping Congress numbers to graphs.
    """
    networks = {}
    
    for congress in range(congress_range[0], congress_range[1] + 1):
        logger.info(f"Building network for Congress {congress}...")
        
        matrix, leg_ids, roll_ids = preprocessor.create_vote_matrix(
            congress=congress,
            chamber=chamber
        )
        
        if matrix is None or matrix.shape[0] < 2:
            logger.warning(f"Insufficient data for Congress {congress}, skipping")
            continue
        
        leg_info = preprocessor.get_legislator_info(leg_ids)
        
        builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
        G = builder.build_similarity_network(similarity_threshold=similarity_threshold)
        
        networks[congress] = G
    
    return networks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    from data_acquisition import VoteviewDataLoader
    from preprocessing import VoteDataPreprocessor
    
    loader = VoteviewDataLoader()
    
    # Load data for recent Congress
    members = loader.load_members(congress_range=(117, 117))
    rollcalls = loader.load_rollcalls(congress_range=(117, 117))
    votes = loader.load_votes(congress_range=(117, 117))
    
    preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
    preprocessor.preprocess_all()
    
    matrix, leg_ids, _ = preprocessor.create_vote_matrix(congress=117)
    leg_info = preprocessor.get_legislator_info(leg_ids)
    
    builder = CongressionalNetworkBuilder(matrix, leg_ids, leg_info)
    G = builder.build_similarity_network(similarity_threshold=0.6)
    
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
