"""Tests for the preprocessing module."""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, '..')

from src.preprocessing import VoteDataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample test data."""
    members = pd.DataFrame({
        'icpsr': [1, 2, 3, 4, 5],
        'congress': [117, 117, 117, 117, 117],
        'chamber': ['House', 'House', 'House', 'Senate', 'Senate'],
        'party_code': [100, 100, 200, 200, 328],
        'state_abbrev': ['CA', 'NY', 'TX', 'FL', 'ME'],
        'bioname': ['Dem 1', 'Dem 2', 'Rep 1', 'Rep 2', 'Ind 1'],
        'nominate_dim1': [-0.5, -0.4, 0.5, 0.6, 0.0],
        'nominate_dim2': [0.1, 0.2, -0.1, -0.2, 0.0],
    })
    
    rollcalls = pd.DataFrame({
        'congress': [117, 117, 117, 117, 117],
        'chamber': ['House', 'House', 'House', 'Senate', 'Senate'],
        'rollnumber': [1, 2, 3, 1, 2],
        'date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-01', '2021-01-02'],
        'vote_question': ['Passage', 'Amendment', 'Motion', 'Passage', 'Amendment'],
    })
    
    votes = pd.DataFrame({
        'congress': [117] * 15,
        'chamber': ['House'] * 9 + ['Senate'] * 6,
        'icpsr': [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4, 5, 4, 5],
        'rollnumber': [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 2, 2, 1, 2],
        'cast_code': [1, 1, 6, 1, 1, 6, 6, 6, 1, 1, 1, 6, 1, 1, 6],
    })
    
    return members, rollcalls, votes


class TestVoteDataPreprocessor:
    """Test cases for VoteDataPreprocessor."""
    
    def test_init(self, sample_data):
        """Test preprocessor initialization."""
        members, rollcalls, votes = sample_data
        preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
        
        assert preprocessor.members_raw is not None
        assert preprocessor.rollcalls_raw is not None
        assert preprocessor.votes_raw is not None
    
    def test_preprocess_members(self, sample_data):
        """Test member preprocessing."""
        members, rollcalls, votes = sample_data
        preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
        
        result = preprocessor.preprocess_members()
        
        assert 'party_name' in result.columns
        assert 'party_group' in result.columns
        assert 'member_congress_id' in result.columns
        assert result['party_name'].iloc[0] == 'Democrat'
        assert result['party_name'].iloc[2] == 'Republican'
    
    def test_preprocess_rollcalls(self, sample_data):
        """Test rollcall preprocessing."""
        members, rollcalls, votes = sample_data
        preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
        
        result = preprocessor.preprocess_rollcalls()
        
        assert 'rollcall_id' in result.columns
        assert 'year' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
    
    def test_preprocess_votes(self, sample_data):
        """Test vote preprocessing."""
        members, rollcalls, votes = sample_data
        preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
        
        result = preprocessor.preprocess_votes()
        
        assert 'vote_value' in result.columns
        assert 'rollcall_id' in result.columns
        assert 'member_congress_id' in result.columns
        
        # Check vote mapping
        assert set(result['vote_value'].unique()).issubset({-1, 0, 1})
    
    def test_create_vote_matrix(self, sample_data):
        """Test vote matrix creation."""
        members, rollcalls, votes = sample_data
        preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
        preprocessor.preprocess_all()
        
        matrix, leg_ids, roll_ids = preprocessor.create_vote_matrix(
            congress=117,
            min_votes=1
        )
        
        assert isinstance(matrix, csr_matrix)
        assert len(leg_ids) > 0
        assert len(roll_ids) > 0
        assert matrix.shape[0] == len(leg_ids)
        assert matrix.shape[1] == len(roll_ids)
    
    def test_create_vote_matrix_chamber_filter(self, sample_data):
        """Test vote matrix with chamber filter."""
        members, rollcalls, votes = sample_data
        preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
        preprocessor.preprocess_all()
        
        matrix_house, _, _ = preprocessor.create_vote_matrix(
            congress=117, chamber='House', min_votes=1
        )
        matrix_senate, _, _ = preprocessor.create_vote_matrix(
            congress=117, chamber='Senate', min_votes=1
        )
        
        # House should have 3 members, Senate should have 2
        assert matrix_house.shape[0] == 3
        assert matrix_senate.shape[0] == 2
    
    def test_get_legislator_info(self, sample_data):
        """Test getting legislator info."""
        members, rollcalls, votes = sample_data
        preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
        preprocessor.preprocess_all()
        
        matrix, leg_ids, _ = preprocessor.create_vote_matrix(congress=117, min_votes=1)
        info = preprocessor.get_legislator_info(leg_ids)
        
        assert len(info) > 0
        assert 'party_name' in info.columns
        assert 'bioname' in info.columns
    
    def test_compute_agreement_rate(self, sample_data):
        """Test agreement rate computation."""
        members, rollcalls, votes = sample_data
        preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
        
        votes1 = np.array([1, 1, -1, 0])
        votes2 = np.array([1, -1, -1, 0])
        
        rate = preprocessor.compute_agreement_rate(votes1, votes2)
        
        # 2 out of 3 agreements (excluding zeros)
        assert rate == pytest.approx(2/3, rel=0.01)
    
    def test_vote_code_mapping(self, sample_data):
        """Test that vote codes are correctly mapped."""
        members, rollcalls, votes = sample_data
        preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
        
        # Check the mapping
        assert preprocessor.VOTE_CODES[1] == 1   # Yea
        assert preprocessor.VOTE_CODES[4] == -1  # Nay
        assert preprocessor.VOTE_CODES[6] == -1  # Announced Nay
        assert preprocessor.VOTE_CODES[9] == 0   # Not a Member


class TestPartyCodeMapping:
    """Test party code mappings."""
    
    def test_major_parties(self, sample_data):
        """Test major party code mappings."""
        members, rollcalls, votes = sample_data
        preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
        
        assert preprocessor.PARTY_CODES[100] == 'Democrat'
        assert preprocessor.PARTY_CODES[200] == 'Republican'
        assert preprocessor.PARTY_CODES[328] == 'Independent'
    
    def test_simplify_party(self, sample_data):
        """Test party simplification."""
        members, rollcalls, votes = sample_data
        preprocessor = VoteDataPreprocessor(members, rollcalls, votes)
        
        assert preprocessor._simplify_party('Democrat') == 'Democrat'
        assert preprocessor._simplify_party('Republican') == 'Republican'
        assert preprocessor._simplify_party('Independent') == 'Independent'
        assert preprocessor._simplify_party('Whig') == 'Republican'
        assert preprocessor._simplify_party('Populist') == 'Democrat'


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
