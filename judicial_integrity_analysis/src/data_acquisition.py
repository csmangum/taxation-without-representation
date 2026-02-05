"""
Data Acquisition Module for Judicial Integrity Analysis.

Fetches judicial data from public APIs and databases including:
- CourtListener API (federal opinions, judges, financial disclosures)
- PACER (federal court dockets)
- State judicial performance review sites
- U.S. Sentencing Commission data
- Judicial conduct commission records
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CourtListenerClient:
    """
    Client for the CourtListener REST API (free.law).

    CourtListener provides access to millions of legal opinions,
    judge profiles, oral arguments, and financial disclosures.

    API docs: https://www.courtlistener.com/api/rest-info/
    """

    BASE_URL = "https://www.courtlistener.com/api/rest/v4"

    # Federal court identifiers for Arizona
    AZ_FEDERAL_COURTS = {
        "azd": "District of Arizona",
    }

    # Arizona state court identifiers
    AZ_STATE_COURTS = {
        "ariz": "Arizona Supreme Court",
        "arizctapp": "Arizona Court of Appeals",
    }

    def __init__(
        self,
        api_token: Optional[str] = None,
        data_dir: str = "data/raw",
        rate_limit_delay: float = 1.0,
    ):
        """
        Initialize the CourtListener API client.

        Args:
            api_token: CourtListener API token. If None, uses unauthenticated
                access (lower rate limits).
            data_dir: Directory to store downloaded data files.
            rate_limit_delay: Seconds to wait between API requests.
        """
        self.api_token = api_token
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self._session = requests.Session()
        if api_token:
            self._session.headers["Authorization"] = f"Token {api_token}"

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Make an authenticated GET request to the CourtListener API.

        Args:
            endpoint: API endpoint path (e.g., "/judges/").
            params: Query parameters.
            timeout: Request timeout in seconds.

        Returns:
            JSON response as dictionary.

        Raises:
            requests.RequestException: On network or HTTP errors.
        """
        url = f"{self.BASE_URL}{endpoint}"
        time.sleep(self.rate_limit_delay)

        response = self._session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _get_paginated(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all pages of a paginated API response.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.
            max_results: Maximum number of results to return.

        Returns:
            List of result dictionaries.
        """
        results: List[Dict[str, Any]] = []
        params = params or {}

        page_url: Optional[str] = f"{self.BASE_URL}{endpoint}"
        while page_url:
            time.sleep(self.rate_limit_delay)
            response = self._session.get(page_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            results.extend(data.get("results", []))

            if max_results and len(results) >= max_results:
                results = results[:max_results]
                break

            page_url = data.get("next")
            params = {}  # params already encoded in next URL

        logger.info(f"Fetched {len(results)} results from {endpoint}")
        return results

    # ==================== Judge Data ====================

    def search_judges(
        self,
        name: Optional[str] = None,
        court: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for judges in the CourtListener database.

        Args:
            name: Judge name or partial name.
            court: Court identifier (e.g., "azd" for District of Arizona).
            date_start: Start date for appointment (YYYY-MM-DD).
            date_end: End date for appointment (YYYY-MM-DD).
            max_results: Maximum number of results.

        Returns:
            List of judge records.
        """
        params: Dict[str, Any] = {}
        if name:
            params["name_last__startswith"] = name
        if court:
            params["court"] = court
        if date_start:
            params["date_nominated__gte"] = date_start
        if date_end:
            params["date_nominated__lte"] = date_end

        return self._get_paginated("/people/", params=params, max_results=max_results)

    def get_judge_detail(self, judge_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific judge.

        Args:
            judge_id: CourtListener judge ID.

        Returns:
            Judge detail dictionary.
        """
        return self._get(f"/people/{judge_id}/")

    def get_judge_positions(self, judge_id: int) -> List[Dict[str, Any]]:
        """
        Get judicial positions held by a judge.

        Args:
            judge_id: CourtListener judge ID.

        Returns:
            List of position records.
        """
        return self._get_paginated(
            "/positions/", params={"person": judge_id}
        )

    def get_judge_financial_disclosures(
        self, judge_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get financial disclosure records for a judge.

        Args:
            judge_id: CourtListener judge ID.

        Returns:
            List of financial disclosure records.
        """
        return self._get_paginated(
            "/financial-disclosures/", params={"person": judge_id}
        )

    # ==================== Opinion / Case Data ====================

    def search_opinions(
        self,
        court: Optional[str] = None,
        judge: Optional[int] = None,
        date_filed_after: Optional[str] = None,
        date_filed_before: Optional[str] = None,
        type_filter: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for court opinions.

        Args:
            court: Court identifier.
            judge: Judge ID.
            date_filed_after: Filter opinions filed after this date (YYYY-MM-DD).
            date_filed_before: Filter opinions filed before this date.
            type_filter: Opinion type filter (e.g., "010combined" for lead opinions).
            max_results: Maximum number of results.

        Returns:
            List of opinion records.
        """
        params: Dict[str, Any] = {}
        if court:
            params["cluster__docket__court"] = court
        if judge:
            params["author"] = judge
        if date_filed_after:
            params["cluster__date_filed__gte"] = date_filed_after
        if date_filed_before:
            params["cluster__date_filed__lte"] = date_filed_before
        if type_filter:
            params["type"] = type_filter

        return self._get_paginated(
            "/opinions/", params=params, max_results=max_results
        )

    def search_dockets(
        self,
        court: Optional[str] = None,
        judge: Optional[str] = None,
        date_filed_after: Optional[str] = None,
        date_filed_before: Optional[str] = None,
        nature_of_suit: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for court dockets.

        Args:
            court: Court identifier.
            judge: Assigned judge name.
            date_filed_after: Filter dockets filed after this date.
            date_filed_before: Filter dockets filed before this date.
            nature_of_suit: Nature of suit code.
            max_results: Maximum number of results.

        Returns:
            List of docket records.
        """
        params: Dict[str, Any] = {}
        if court:
            params["court"] = court
        if judge:
            params["assigned_to__name_last__startswith"] = judge
        if date_filed_after:
            params["date_filed__gte"] = date_filed_after
        if date_filed_before:
            params["date_filed__lte"] = date_filed_before
        if nature_of_suit:
            params["nature_of_suit"] = nature_of_suit

        return self._get_paginated(
            "/dockets/", params=params, max_results=max_results
        )

    # ==================== Court / Jurisdiction Data ====================

    def get_courts(self, jurisdiction: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get court information.

        Args:
            jurisdiction: Filter by jurisdiction type (e.g., "FD" for federal district).

        Returns:
            List of court records.
        """
        params: Dict[str, Any] = {}
        if jurisdiction:
            params["jurisdiction"] = jurisdiction
        return self._get_paginated("/courts/", params=params)

    def get_court_detail(self, court_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a court.

        Args:
            court_id: Court identifier (e.g., "azd").

        Returns:
            Court detail dictionary.
        """
        return self._get(f"/courts/{court_id}/")


class JudicialConductScraper:
    """
    Scraper for state judicial conduct commission data.

    Collects disciplinary actions, public complaints, and performance
    review data from state-level judicial oversight bodies.
    """

    # Known URLs for Arizona judicial oversight bodies
    AZ_SOURCES = {
        "conduct_commission": "https://www.azcourts.gov/cjc/",
        "performance_review": "https://www.azcourts.gov/jpr/",
        "judicial_branch": "https://www.azcourts.gov/",
    }

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the scraper.

        Args:
            data_dir: Directory to store scraped data.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "JudicialIntegrityResearch/0.1 (Academic Research)"
        )

    def fetch_az_disciplinary_actions(self) -> List[Dict[str, Any]]:
        """
        Fetch disciplinary action records from the Arizona Commission
        on Judicial Conduct.

        Returns:
            List of disciplinary action records.

        Note:
            This is a placeholder for the actual scraping logic.
            Implementation depends on the structure of the target website
            at the time of use. The website may require BeautifulSoup or
            Selenium for proper parsing.
        """
        logger.info("Fetching AZ disciplinary actions...")
        logger.warning(
            "Disciplinary action scraping requires site-specific implementation. "
            "Check %s for current structure.",
            self.AZ_SOURCES["conduct_commission"],
        )
        return []

    def fetch_az_performance_reviews(self) -> List[Dict[str, Any]]:
        """
        Fetch judicial performance review data from the Arizona
        Commission on Judicial Performance Review.

        Returns:
            List of performance review records.

        Note:
            This is a placeholder. The JPR publishes PDF reports that
            require pdfplumber for extraction.
        """
        logger.info("Fetching AZ performance reviews...")
        logger.warning(
            "Performance review scraping requires site-specific implementation. "
            "Check %s for current structure.",
            self.AZ_SOURCES["performance_review"],
        )
        return []


class SentencingDataLoader:
    """
    Loader for sentencing statistics from the U.S. Sentencing Commission
    and state-level sentencing reports.
    """

    USSC_BASE_URL = "https://www.ussc.gov"

    # Known dataset endpoints (these are illustrative; actual URLs may differ)
    USSC_DATASETS = {
        "annual_report": "/research/datafiles/commission-datafiles",
        "district_data": "/research/datafiles/district-datafiles",
    }

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the sentencing data loader.

        Args:
            data_dir: Directory to store downloaded data.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_ussc_data(
        self,
        dataset: str = "annual_report",
        year: Optional[int] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Download sentencing data from the U.S. Sentencing Commission.

        Args:
            dataset: Dataset type key.
            year: Fiscal year of the data.
            force: Re-download even if file exists.

        Returns:
            Path to downloaded file, or None on failure.

        Note:
            USSC data files are often in SAS or Excel format and require
            specific parsers. This is a placeholder for the download logic.
        """
        logger.info("USSC data download: placeholder - check %s for available files", self.USSC_BASE_URL)
        return None


class JudicialDataAggregator:
    """
    High-level aggregator that combines data from multiple sources
    into unified judge profiles for analysis.
    """

    def __init__(
        self,
        state: str = "az",
        data_dir: str = "data",
        courtlistener_token: Optional[str] = None,
    ):
        """
        Initialize the aggregator for a specific state.

        Args:
            state: Two-letter state code (lowercase).
            data_dir: Base data directory.
            courtlistener_token: Optional CourtListener API token.
        """
        self.state = state.lower()
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / self.state / "raw"
        self.processed_dir = self.data_dir / self.state / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.courtlistener = CourtListenerClient(
            api_token=courtlistener_token,
            data_dir=str(self.raw_dir / "courtlistener"),
        )
        self.conduct_scraper = JudicialConductScraper(
            data_dir=str(self.raw_dir / "conduct"),
        )
        self.sentencing_loader = SentencingDataLoader(
            data_dir=str(self.raw_dir / "sentencing"),
        )

    def collect_federal_judges(
        self,
        court: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect federal judge data for the state.

        Args:
            court: Specific court identifier. Defaults to state's federal district.
            max_results: Maximum number of judges to fetch.

        Returns:
            List of judge records with positions and disclosures.
        """
        if court is None:
            court = f"{self.state}d"  # e.g., "azd" for District of Arizona

        judges = self.courtlistener.search_judges(
            court=court, max_results=max_results
        )

        enriched_judges = []
        for judge in tqdm(judges, desc="Enriching judge data"):
            judge_id = judge.get("id")
            if judge_id:
                try:
                    judge["positions"] = self.courtlistener.get_judge_positions(
                        judge_id
                    )
                    judge["financial_disclosures"] = (
                        self.courtlistener.get_judge_financial_disclosures(judge_id)
                    )
                except requests.RequestException as e:
                    logger.warning("Failed to enrich judge %s: %s", judge_id, e)
            enriched_judges.append(judge)

        logger.info("Collected %d federal judges for %s", len(enriched_judges), self.state.upper())
        return enriched_judges

    def collect_state_judges(self) -> List[Dict[str, Any]]:
        """
        Collect state-level judge data.

        Returns:
            List of state judge records.

        Note:
            State judge data collection varies significantly by state.
            This implementation provides a framework; specific scraping
            logic must be adapted per state.
        """
        logger.info(
            "State judge collection for %s requires state-specific implementation.",
            self.state.upper(),
        )
        return []

    def collect_disciplinary_records(self) -> List[Dict[str, Any]]:
        """
        Collect judicial disciplinary / conduct records.

        Returns:
            List of disciplinary records.
        """
        if self.state == "az":
            return self.conduct_scraper.fetch_az_disciplinary_actions()
        logger.warning("No disciplinary scraper for state: %s", self.state)
        return []

    def collect_performance_reviews(self) -> List[Dict[str, Any]]:
        """
        Collect judicial performance review data.

        Returns:
            List of performance review records.
        """
        if self.state == "az":
            return self.conduct_scraper.fetch_az_performance_reviews()
        logger.warning("No performance review scraper for state: %s", self.state)
        return []

    def collect_all(self, max_judges: Optional[int] = None) -> Dict[str, Any]:
        """
        Collect all available data for the state.

        Args:
            max_judges: Maximum number of federal judges to fetch.

        Returns:
            Dictionary with all collected data keyed by source type.
        """
        return {
            "federal_judges": self.collect_federal_judges(max_results=max_judges),
            "state_judges": self.collect_state_judges(),
            "disciplinary_records": self.collect_disciplinary_records(),
            "performance_reviews": self.collect_performance_reviews(),
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "state": self.state,
        }
