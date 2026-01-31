"""
Data Acquisition Module for Congressional Voting Networks.

Downloads and manages voting record data from Voteview.com (UCLA Political Science).
"""

import logging
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)


class VoteviewDataLoader:
    """
    Handles downloading and loading of Voteview.com congressional voting data.
    
    Data includes:
    - Members: Legislator metadata (party, state, DW-NOMINATE scores)
    - Rollcalls: Vote metadata (date, bill, outcome)
    - Votes: Individual vote records linking legislators to rollcalls
    """
    
    # Base URLs for Voteview data
    BASE_URL = "https://voteview.com/static/data/out"
    
    # Dataset file mappings
    DATASETS = {
        "members": "members/HSall_members.csv",
        "rollcalls": "rollcalls/HSall_rollcalls.csv",
        "votes": "votes/HSall_votes.csv",
    }
    
    # Per-congress URL patterns (for smaller downloads)
    CONGRESS_URL_PATTERN = {
        "members": "members/H{congress:03d}_members.csv",
        "rollcalls": "rollcalls/H{congress:03d}_rollcalls.csv",
        "votes": "votes/H{congress:03d}_votes.csv",
    }
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store downloaded data files.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded dataframes
        self._cache = {}
    
    def _download_file(
        self, 
        url: str, 
        dest_path: Path, 
        force: bool = False,
        chunk_size: int = 8192
    ) -> bool:
        """
        Download a file from URL to destination path.
        
        Args:
            url: Source URL to download from.
            dest_path: Local path to save the file.
            force: If True, re-download even if file exists.
            chunk_size: Download chunk size in bytes.
            
        Returns:
            True if download was successful, False otherwise.
        """
        if dest_path.exists() and not force:
            logger.info(f"File already exists: {dest_path}")
            return True
        
        logger.info(f"Downloading: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                          desc=dest_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Downloaded successfully: {dest_path}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False
    
    def download_all_data(self, force: bool = False) -> dict:
        """
        Download all main datasets (full historical data).
        
        Args:
            force: If True, re-download even if files exist.
            
        Returns:
            Dictionary mapping dataset names to file paths.
        """
        downloaded = {}
        
        for name, path in self.DATASETS.items():
            url = f"{self.BASE_URL}/{path}"
            dest_path = self.data_dir / f"HSall_{name}.csv"
            
            if self._download_file(url, dest_path, force):
                downloaded[name] = dest_path
            else:
                logger.warning(f"Failed to download {name} dataset")
        
        return downloaded
    
    def download_congress_data(
        self, 
        congress: int, 
        chamber: str = "HS",
        force: bool = False
    ) -> dict:
        """
        Download data for a specific Congress number.
        
        Args:
            congress: Congress number (1-119+).
            chamber: "H" for House, "S" for Senate, "HS" for both.
            force: If True, re-download even if files exist.
            
        Returns:
            Dictionary mapping dataset names to file paths.
        """
        downloaded = {}
        
        for name, pattern in self.CONGRESS_URL_PATTERN.items():
            # Adjust pattern for chamber
            file_pattern = pattern.replace("H{congress", f"{chamber}{{congress")
            path = file_pattern.format(congress=congress)
            url = f"{self.BASE_URL}/{path}"
            dest_path = self.data_dir / f"{chamber}{congress:03d}_{name}.csv"
            
            if self._download_file(url, dest_path, force):
                downloaded[name] = dest_path
        
        return downloaded
    
    def load_members(
        self, 
        congress_range: Optional[tuple] = None,
        chamber: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load members dataset with optional filtering.
        
        Args:
            congress_range: Optional tuple (start, end) to filter by Congress.
            chamber: Optional "House" or "Senate" to filter by chamber.
            
        Returns:
            DataFrame with member data.
        """
        cache_key = f"members_{congress_range}_{chamber}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self.data_dir / "HSall_members.csv"
        if not file_path.exists():
            logger.info("Members file not found, downloading...")
            self.download_all_data()
        
        df = pd.read_csv(file_path)
        
        # Apply filters
        if congress_range:
            df = df[(df['congress'] >= congress_range[0]) & 
                    (df['congress'] <= congress_range[1])]
        
        if chamber:
            df = df[df['chamber'] == chamber]
        
        self._cache[cache_key] = df
        return df
    
    def load_rollcalls(
        self, 
        congress_range: Optional[tuple] = None,
        chamber: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load rollcalls dataset with optional filtering.
        
        Args:
            congress_range: Optional tuple (start, end) to filter by Congress.
            chamber: Optional "House" or "Senate" to filter by chamber.
            
        Returns:
            DataFrame with rollcall data.
        """
        cache_key = f"rollcalls_{congress_range}_{chamber}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self.data_dir / "HSall_rollcalls.csv"
        if not file_path.exists():
            logger.info("Rollcalls file not found, downloading...")
            self.download_all_data()
        
        df = pd.read_csv(file_path)
        
        # Apply filters
        if congress_range:
            df = df[(df['congress'] >= congress_range[0]) & 
                    (df['congress'] <= congress_range[1])]
        
        if chamber:
            df = df[df['chamber'] == chamber]
        
        self._cache[cache_key] = df
        return df
    
    def load_votes(
        self, 
        congress_range: Optional[tuple] = None,
        chamber: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load votes dataset with optional filtering.
        
        Note: This is the largest file (~10M+ rows). Use congress_range 
        for memory efficiency.
        
        Args:
            congress_range: Optional tuple (start, end) to filter by Congress.
            chamber: Optional "House" or "Senate" to filter by chamber.
            
        Returns:
            DataFrame with vote data.
        """
        cache_key = f"votes_{congress_range}_{chamber}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self.data_dir / "HSall_votes.csv"
        if not file_path.exists():
            logger.info("Votes file not found, downloading...")
            self.download_all_data()
        
        # For large files, use chunked reading if filtering
        if congress_range or chamber:
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=500000):
                if congress_range:
                    chunk = chunk[(chunk['congress'] >= congress_range[0]) & 
                                  (chunk['congress'] <= congress_range[1])]
                if chamber:
                    chunk = chunk[chunk['chamber'] == chamber]
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path)
        
        self._cache[cache_key] = df
        return df
    
    def get_congress_info(self) -> pd.DataFrame:
        """
        Get summary information about available Congresses.
        
        Returns:
            DataFrame with congress number, date range, and vote counts.
        """
        rollcalls = self.load_rollcalls()
        
        info = rollcalls.groupby('congress').agg({
            'rollnumber': 'count',
            'date': ['min', 'max']
        }).reset_index()
        
        info.columns = ['congress', 'num_rollcalls', 'start_date', 'end_date']
        return info
    
    def clear_cache(self):
        """Clear the in-memory cache of loaded dataframes."""
        self._cache.clear()
        logger.info("Cache cleared")


def download_data(data_dir: str = "data/raw", force: bool = False) -> dict:
    """
    Convenience function to download all Voteview data.
    
    Args:
        data_dir: Directory to store downloaded data.
        force: If True, re-download even if files exist.
        
    Returns:
        Dictionary mapping dataset names to file paths.
    """
    loader = VoteviewDataLoader(data_dir)
    return loader.download_all_data(force)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Download data
    loader = VoteviewDataLoader()
    files = loader.download_all_data()
    
    print(f"\nDownloaded files: {files}")
    
    # Show summary
    info = loader.get_congress_info()
    print(f"\nCongress summary:\n{info.tail(10)}")
