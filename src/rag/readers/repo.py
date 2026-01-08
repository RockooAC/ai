from os import cpu_count, path
from pathlib import Path
from typing import Dict, Optional, Set, List
from xmlrpc.client import boolean
from gitingest import ingest

from src.rag.common import setup_logger

class RepoReader:
    def __init__(
        self,
        repo_path: str,
        output_dir: str,
        log_file: Optional[Path] = None,
        debug: bool = False,
    ):
        """
        Initialize the repository reader.

        Args:
            repo_path: Path to the repository root
            output_dir: Directory where digest files will be saved
            debug: Enable debug logging
            max_workers: Maximum number of worker processes
            log_file: Optional path to log file
        """
        self.repo_path = path.expanduser(repo_path)
        self.output_dir = path.expanduser(output_dir)
        self.logger = setup_logger("RepoReader", debug=debug, log_file=log_file)

    def process_repo(
        self,
        exclude_patterns: Optional[List[str]]
    ) -> dict[str, str]:
        """
        Process a single directory and generate its digest file.
        """
        if not Path(self.output_dir).exists():
            self.logger.warning(f"Directory not found: {self.output_dir}. Creating new one..")
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Processing directory: {self.repo_path}")
        digest_file_path = Path(self.output_dir) / "repo.digest"

        try:
            ingest(
                source=self.repo_path,
                include_submodules=True,
                output=str(digest_file_path),
                exclude_patterns=exclude_patterns,
                include_gitignored=False
            )

        except Exception as e:
            self.logger.error(f"Error ingesting directory {self.repo_path}: {e}", exc_info=True)
            return {}
        
        
        if not digest_file_path.exists() or digest_file_path.stat().st_size == 0:
            self.logger.warning(f"Digest file empty or missing: {digest_file_path}")
            return {}
            
        return {"repo.digest": str(digest_file_path)}

