"""
Usage:
    python digest.py -d "~/wrk/git/rg" -o "/tmp/" -l cpp
"""

import argparse
import os
from pathlib import Path

from src.rag.readers.repo import RepoReader
from src.rag.common import setup_logger
from src.rag.config import REPOSITORY_CONFIG

# Set up the main logger for the script
logger = setup_logger("Digest", debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate digest files for specific programming languages in a repository."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="~/wrk/git/rg",
        help="Root directory of the repository (default: ~/wrk/git/rg).",
    )
    parser.add_argument(
        "-o", 
        "--output", 
        type=str, 
        required=True, 
        help="Output directory for digest files."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging.",
    )
    args = parser.parse_args()
    directory = os.path.expanduser(args.directory)

    reader = RepoReader(
        repo_path=args.directory,
        output_dir=args.output,
        debug=args.debug, 
    )
    
    reader.process_repo(
        exclude_patterns=REPOSITORY_CONFIG["RG_EXCLUDE_PATTERNS"]
    )