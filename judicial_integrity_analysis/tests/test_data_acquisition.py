"""Tests for the data acquisition module."""

import pytest
from unittest.mock import patch, MagicMock

from judicial_integrity_analysis.src.data_acquisition import (
    CourtListenerClient,
    JudicialConductScraper,
    SentencingDataLoader,
    JudicialDataAggregator,
)


class TestCourtListenerClient:
    """Test cases for CourtListenerClient."""

    def test_init_without_token(self):
        """Test initialization without API token."""
        client = CourtListenerClient(data_dir="/tmp/test_data")
        assert client.api_token is None
        assert "Authorization" not in client._session.headers

    def test_init_with_token(self):
        """Test initialization with API token."""
        client = CourtListenerClient(
            api_token="test_token_123",
            data_dir="/tmp/test_data",
        )
        assert client.api_token == "test_token_123"
        assert client._session.headers["Authorization"] == "Token test_token_123"

    def test_base_url(self):
        """Test that base URL is set correctly."""
        client = CourtListenerClient()
        assert "courtlistener.com" in client.BASE_URL

    def test_az_court_identifiers(self):
        """Test Arizona court identifiers are defined."""
        assert "azd" in CourtListenerClient.AZ_FEDERAL_COURTS
        assert "ariz" in CourtListenerClient.AZ_STATE_COURTS

    @patch("judicial_integrity_analysis.src.data_acquisition.requests.Session")
    def test_get_request(self, mock_session_cls):
        """Test basic GET request handling."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"id": 1}]}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = CourtListenerClient(rate_limit_delay=0)
        client._session = mock_session

        result = client._get("/people/", params={"court": "azd"})
        assert "results" in result

    @patch("judicial_integrity_analysis.src.data_acquisition.requests.Session")
    def test_search_judges(self, mock_session_cls):
        """Test judge search functionality."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "name_first": "John", "name_last": "Smith"},
                {"id": 2, "name_first": "Jane", "name_last": "Doe"},
            ],
            "next": None,
        }
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = CourtListenerClient(rate_limit_delay=0)
        client._session = mock_session

        results = client.search_judges(court="azd")
        assert len(results) == 2

    @patch("judicial_integrity_analysis.src.data_acquisition.requests.Session")
    def test_search_judges_with_max_results(self, mock_session_cls):
        """Test that max_results limits output."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"id": i} for i in range(10)],
            "next": None,
        }
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = CourtListenerClient(rate_limit_delay=0)
        client._session = mock_session

        results = client.search_judges(court="azd", max_results=5)
        assert len(results) == 5


class TestJudicialConductScraper:
    """Test cases for JudicialConductScraper."""

    def test_init(self, tmp_path):
        """Test scraper initialization."""
        scraper = JudicialConductScraper(data_dir=str(tmp_path))
        assert scraper.data_dir.exists()

    def test_az_sources_defined(self):
        """Test that AZ sources are defined."""
        assert "conduct_commission" in JudicialConductScraper.AZ_SOURCES
        assert "performance_review" in JudicialConductScraper.AZ_SOURCES

    def test_disciplinary_placeholder(self, tmp_path):
        """Test that disciplinary action fetch returns placeholder."""
        scraper = JudicialConductScraper(data_dir=str(tmp_path))
        result = scraper.fetch_az_disciplinary_actions()
        assert isinstance(result, list)

    def test_performance_review_placeholder(self, tmp_path):
        """Test that performance review fetch returns placeholder."""
        scraper = JudicialConductScraper(data_dir=str(tmp_path))
        result = scraper.fetch_az_performance_reviews()
        assert isinstance(result, list)


class TestSentencingDataLoader:
    """Test cases for SentencingDataLoader."""

    def test_init(self, tmp_path):
        """Test loader initialization."""
        loader = SentencingDataLoader(data_dir=str(tmp_path))
        assert loader.data_dir.exists()

    def test_ussc_download_placeholder(self, tmp_path):
        """Test that USSC download returns placeholder."""
        loader = SentencingDataLoader(data_dir=str(tmp_path))
        result = loader.download_ussc_data()
        assert result is None  # Placeholder


class TestJudicialDataAggregator:
    """Test cases for JudicialDataAggregator."""

    def test_init(self, tmp_path):
        """Test aggregator initialization."""
        agg = JudicialDataAggregator(
            state="az",
            data_dir=str(tmp_path),
        )
        assert agg.state == "az"
        assert agg.raw_dir.exists()
        assert agg.processed_dir.exists()

    def test_state_normalization(self, tmp_path):
        """Test that state code is normalized to lowercase."""
        agg = JudicialDataAggregator(state="AZ", data_dir=str(tmp_path))
        assert agg.state == "az"

    def test_collect_state_judges_placeholder(self, tmp_path):
        """Test that state judge collection is a placeholder."""
        agg = JudicialDataAggregator(state="az", data_dir=str(tmp_path))
        result = agg.collect_state_judges()
        assert isinstance(result, list)

    def test_collect_disciplinary_unknown_state(self, tmp_path):
        """Test disciplinary collection for unsupported state."""
        agg = JudicialDataAggregator(state="xx", data_dir=str(tmp_path))
        result = agg.collect_disciplinary_records()
        assert isinstance(result, list)

    @patch.object(CourtListenerClient, "search_judges")
    @patch.object(CourtListenerClient, "get_judge_positions")
    @patch.object(CourtListenerClient, "get_judge_financial_disclosures")
    def test_collect_federal_judges(
        self, mock_disclosures, mock_positions, mock_search, tmp_path
    ):
        """Test federal judge collection with mocked API."""
        mock_search.return_value = [
            {"id": 1, "name_first": "Test", "name_last": "Judge"},
        ]
        mock_positions.return_value = [{"position_type": "Judge"}]
        mock_disclosures.return_value = []

        agg = JudicialDataAggregator(state="az", data_dir=str(tmp_path))
        result = agg.collect_federal_judges()

        assert len(result) == 1
        assert "positions" in result[0]

    @patch.object(CourtListenerClient, "search_judges")
    @patch.object(CourtListenerClient, "get_judge_positions")
    @patch.object(CourtListenerClient, "get_judge_financial_disclosures")
    def test_collect_all(
        self, mock_disclosures, mock_positions, mock_search, tmp_path
    ):
        """Test full data collection."""
        mock_search.return_value = []
        mock_positions.return_value = []
        mock_disclosures.return_value = []

        agg = JudicialDataAggregator(state="az", data_dir=str(tmp_path))
        result = agg.collect_all()

        assert "federal_judges" in result
        assert "state_judges" in result
        assert "disciplinary_records" in result
        assert "performance_reviews" in result
        assert "collection_timestamp" in result
        assert result["state"] == "az"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
