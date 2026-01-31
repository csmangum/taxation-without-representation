"""
Data Preprocessing Module for Congressional Voting Networks.

Handles cleaning, filtering, and transformation of voting record data
into formats suitable for network analysis.
"""

import logging
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from pathlib import Path

logger = logging.getLogger(__name__)


class VoteDataPreprocessor:
    """
    Preprocesses Voteview data for network analysis.
    
    Transforms raw CSV data into cleaned DataFrames and vote matrices
    suitable for computing legislator similarities.
    """
    
    # Vote code mappings from Voteview documentation
    VOTE_CODES = {
        # Yea votes (1-3)
        1: 1,   # Yea
        2: 1,   # Paired Yea
        3: 1,   # Announced Yea
        # Nay votes (4-6)
        4: -1,  # Nay
        5: -1,  # Paired Nay
        6: -1,  # Announced Nay
        # Other (7-9) - treated as abstain/not voting
        7: 0,   # Present (not voting)
        8: 0,   # Not Present
        9: 0,   # Not a Member
    }
    
    # Party code mappings (major parties)
    PARTY_CODES = {
        100: "Democrat",
        200: "Republican",
        # Historical parties
        1: "Federalist",
        13: "Democratic-Republican",
        22: "Adams",
        25: "National Republican",
        26: "Anti-Masonic",
        29: "Whig",
        34: "Whig & Democrat",
        37: "Constitutional Unionist",
        44: "Nullifier",
        46: "States Rights",
        108: "Anti-Lecompton Democrat",
        112: "Conservative Democrat",
        114: "Readjuster",
        117: "Silver Republican",
        203: "Unconditional Unionist",
        206: "Unionist",
        208: "Liberal Republican",
        213: "Progressive Republican",
        300: "Free Soil",
        310: "American",
        326: "National Greenbacker",
        340: "Populist",
        347: "Prohibitionist",
        354: "Silver",
        355: "Union",
        356: "Union Labor",
        370: "Progressive",
        380: "Socialist",
        402: "Liberal",
        522: "American Labor",
        537: "Farmer-Labor",
        328: "Independent",
        329: "Independent Democrat",
        331: "Independent Republican",
    }
    
    def __init__(self, members_df: pd.DataFrame, 
                 rollcalls_df: pd.DataFrame, 
                 votes_df: pd.DataFrame):
        """
        Initialize preprocessor with raw data.
        
        Args:
            members_df: Raw members DataFrame from Voteview.
            rollcalls_df: Raw rollcalls DataFrame from Voteview.
            votes_df: Raw votes DataFrame from Voteview.
        """
        self.members_raw = members_df.copy()
        self.rollcalls_raw = rollcalls_df.copy()
        self.votes_raw = votes_df.copy()
        
        # Processed data (populated by preprocessing methods)
        self.members = None
        self.rollcalls = None
        self.votes = None
        self.vote_matrix = None
        self.legislator_ids = None
        self.rollcall_ids = None
    
    def preprocess_members(self) -> pd.DataFrame:
        """
        Clean and enhance member data.
        
        Returns:
            Cleaned members DataFrame with party names and standardized fields.
        """
        df = self.members_raw.copy()
        
        # Map party codes to names
        df['party_name'] = df['party_code'].map(self.PARTY_CODES)
        df['party_name'] = df['party_name'].fillna('Other')
        
        # Create simplified party grouping (for modern analysis)
        df['party_group'] = df['party_name'].apply(self._simplify_party)
        
        # Clean state codes
        df['state_abbrev'] = df['state_abbrev'].fillna('XX')
        
        # Ensure DW-NOMINATE scores are numeric
        df['nominate_dim1'] = pd.to_numeric(df['nominate_dim1'], errors='coerce')
        df['nominate_dim2'] = pd.to_numeric(df['nominate_dim2'], errors='coerce')
        
        # Create unique identifier combining ICPSR and congress
        df['member_congress_id'] = df['icpsr'].astype(str) + '_' + df['congress'].astype(str)
        
        # Sort by congress and chamber
        df = df.sort_values(['congress', 'chamber', 'party_code'])
        
        self.members = df
        logger.info(f"Preprocessed {len(df)} member records")
        return df
    
    def _simplify_party(self, party_name: str) -> str:
        """Simplify party name to major groupings."""
        if 'Democrat' in party_name or party_name in ['Populist', 'Silver']:
            return 'Democrat'
        elif 'Republican' in party_name or party_name in ['Whig', 'Federalist']:
            return 'Republican'
        elif party_name in ['Independent', 'Independent Democrat', 'Independent Republican']:
            return 'Independent'
        else:
            return 'Other'
    
    def preprocess_rollcalls(self) -> pd.DataFrame:
        """
        Clean and enhance rollcall data.
        
        Returns:
            Cleaned rollcalls DataFrame.
        """
        df = self.rollcalls_raw.copy()
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        
        # Create unique rollcall identifier
        df['rollcall_id'] = (df['congress'].astype(str) + '_' + 
                            df['chamber'] + '_' + 
                            df['rollnumber'].astype(str))
        
        # Clean vote question
        df['vote_question'] = df['vote_question'].fillna('Unknown')
        
        self.rollcalls = df
        logger.info(f"Preprocessed {len(df)} rollcall records")
        return df
    
    def preprocess_votes(self) -> pd.DataFrame:
        """
        Clean and transform vote data.
        
        Returns:
            Cleaned votes DataFrame with standardized vote values.
        """
        df = self.votes_raw.copy()
        
        # Map vote codes to standardized values (1=Yea, -1=Nay, 0=Other)
        df['vote_value'] = df['cast_code'].map(self.VOTE_CODES)
        df['vote_value'] = df['vote_value'].fillna(0)
        
        # Create unique identifiers
        df['rollcall_id'] = (df['congress'].astype(str) + '_' + 
                            df['chamber'] + '_' + 
                            df['rollnumber'].astype(str))
        df['member_congress_id'] = df['icpsr'].astype(str) + '_' + df['congress'].astype(str)
        
        self.votes = df
        logger.info(f"Preprocessed {len(df)} vote records")
        return df
    
    def preprocess_all(self) -> Dict[str, pd.DataFrame]:
        """
        Run all preprocessing steps.
        
        Returns:
            Dictionary with cleaned dataframes.
        """
        return {
            'members': self.preprocess_members(),
            'rollcalls': self.preprocess_rollcalls(),
            'votes': self.preprocess_votes(),
        }
    
    def create_vote_matrix(
        self,
        congress: Optional[int] = None,
        congress_range: Optional[Tuple[int, int]] = None,
        chamber: Optional[str] = None,
        min_votes: int = 10
    ) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """
        Create sparse vote matrix (legislators x rollcalls).
        
        Args:
            congress: Single Congress number to filter.
            congress_range: Tuple (start, end) Congress range.
            chamber: "House" or "Senate" to filter.
            min_votes: Minimum votes a legislator must have to be included.
            
        Returns:
            Tuple of (sparse_matrix, legislator_ids, rollcall_ids)
        """
        if self.votes is None:
            self.preprocess_all()
        
        votes = self.votes.copy()
        
        # Apply filters
        if congress is not None:
            votes = votes[votes['congress'] == congress]
        elif congress_range is not None:
            votes = votes[(votes['congress'] >= congress_range[0]) & 
                         (votes['congress'] <= congress_range[1])]
        
        if chamber is not None:
            votes = votes[votes['chamber'] == chamber]
        
        if len(votes) == 0:
            logger.warning("No votes match the filter criteria")
            return None, None, None
        
        # Get unique legislators and rollcalls
        legislator_ids = votes['member_congress_id'].unique()
        rollcall_ids = votes['rollcall_id'].unique()
        
        # Create mappings for matrix indices
        leg_to_idx = {leg: idx for idx, leg in enumerate(legislator_ids)}
        roll_to_idx = {roll: idx for idx, roll in enumerate(rollcall_ids)}
        
        # Build sparse matrix
        n_legs = len(legislator_ids)
        n_rolls = len(rollcall_ids)
        
        matrix = lil_matrix((n_legs, n_rolls), dtype=np.int8)
        
        for _, row in votes.iterrows():
            leg_idx = leg_to_idx[row['member_congress_id']]
            roll_idx = roll_to_idx[row['rollcall_id']]
            matrix[leg_idx, roll_idx] = row['vote_value']
        
        # Convert to CSR for efficient operations
        matrix = csr_matrix(matrix)
        
        # Filter legislators with too few votes
        if min_votes > 0:
            vote_counts = np.abs(matrix).sum(axis=1).A1
            valid_mask = vote_counts >= min_votes
            matrix = matrix[valid_mask]
            legislator_ids = legislator_ids[valid_mask]
        
        self.vote_matrix = matrix
        self.legislator_ids = legislator_ids
        self.rollcall_ids = rollcall_ids
        
        logger.info(f"Created vote matrix: {matrix.shape[0]} legislators x {matrix.shape[1]} rollcalls")
        return matrix, legislator_ids, rollcall_ids
    
    def get_legislator_info(self, member_congress_ids: np.ndarray) -> pd.DataFrame:
        """
        Get legislator information for given member-congress IDs.
        
        Args:
            member_congress_ids: Array of member_congress_id values.
            
        Returns:
            DataFrame with legislator info indexed by member_congress_id.
        """
        if self.members is None:
            self.preprocess_members()
        
        info = self.members[self.members['member_congress_id'].isin(member_congress_ids)]
        return info.set_index('member_congress_id')
    
    def compute_agreement_rate(
        self,
        votes1: np.ndarray,
        votes2: np.ndarray,
        exclude_zeros: bool = True
    ) -> float:
        """
        Compute agreement rate between two vote vectors.
        
        Args:
            votes1: First vote vector (1=Yea, -1=Nay, 0=Other).
            votes2: Second vote vector.
            exclude_zeros: If True, only count votes where both legislators voted.
            
        Returns:
            Agreement rate (0.0 to 1.0).
        """
        if exclude_zeros:
            # Only consider votes where both legislators voted (not 0)
            mask = (votes1 != 0) & (votes2 != 0)
            if mask.sum() == 0:
                return np.nan
            votes1 = votes1[mask]
            votes2 = votes2[mask]
        
        agreements = (votes1 == votes2).sum()
        total = len(votes1)
        
        return agreements / total if total > 0 else np.nan
    
    def save_processed_data(self, output_dir: str = "data/processed"):
        """
        Save processed dataframes to parquet files.
        
        Args:
            output_dir: Directory to save processed files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.members is not None:
            self.members.to_parquet(output_path / "members.parquet")
            logger.info(f"Saved members to {output_path / 'members.parquet'}")
        
        if self.rollcalls is not None:
            self.rollcalls.to_parquet(output_path / "rollcalls.parquet")
            logger.info(f"Saved rollcalls to {output_path / 'rollcalls.parquet'}")
        
        if self.votes is not None:
            self.votes.to_parquet(output_path / "votes.parquet")
            logger.info(f"Saved votes to {output_path / 'votes.parquet'}")
    
    @classmethod
    def load_processed_data(cls, processed_dir: str = "data/processed") -> 'VoteDataPreprocessor':
        """
        Load previously processed data from parquet files.
        
        Args:
            processed_dir: Directory containing processed parquet files.
            
        Returns:
            VoteDataPreprocessor instance with loaded data.
        """
        path = Path(processed_dir)
        
        members = pd.read_parquet(path / "members.parquet")
        rollcalls = pd.read_parquet(path / "rollcalls.parquet")
        votes = pd.read_parquet(path / "votes.parquet")
        
        preprocessor = cls(members, rollcalls, votes)
        preprocessor.members = members
        preprocessor.rollcalls = rollcalls
        preprocessor.votes = votes
        
        return preprocessor


def preprocess_congress_data(
    members_df: pd.DataFrame,
    rollcalls_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    congress: Optional[int] = None,
    chamber: Optional[str] = None
) -> Tuple[csr_matrix, np.ndarray, pd.DataFrame]:
    """
    Convenience function to preprocess data and create vote matrix.
    
    Args:
        members_df: Raw members DataFrame.
        rollcalls_df: Raw rollcalls DataFrame.
        votes_df: Raw votes DataFrame.
        congress: Optional Congress number to filter.
        chamber: Optional chamber to filter.
        
    Returns:
        Tuple of (vote_matrix, legislator_ids, legislator_info)
    """
    preprocessor = VoteDataPreprocessor(members_df, rollcalls_df, votes_df)
    preprocessor.preprocess_all()
    
    matrix, leg_ids, _ = preprocessor.create_vote_matrix(
        congress=congress,
        chamber=chamber
    )
    
    leg_info = preprocessor.get_legislator_info(leg_ids)
    
    return matrix, leg_ids, leg_info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    from .data_acquisition import VoteviewDataLoader
    
    loader = VoteviewDataLoader()
    
    # Load a small subset for testing
    members = loader.load_members(congress_range=(116, 118))
    rollcalls = loader.load_rollcalls(congress_range=(116, 118))
    votes = loader.load_votes(congress_range=(116, 118))
    
    preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
    preprocessor.preprocess_all()
    
    # Create vote matrix for 117th Congress
    matrix, leg_ids, roll_ids = preprocessor.create_vote_matrix(congress=117)
    
    print(f"Vote matrix shape: {matrix.shape}")
    print(f"Number of legislators: {len(leg_ids)}")
    print(f"Number of rollcalls: {len(roll_ids)}")
