#!/usr/bin/python3

import re
import logging
import subprocess
from enum import Enum
import tree_sitter as TS
from tree_sitter import Query
import tree_sitter_cpp as TSCpp
from dataclasses import dataclass
from typing import List, AnyStr, Tuple
from queries import CPP_DECLS_CALLS_QUERY


class Processing(Enum):
    SINGLE = "SINGLE"
    TREE_SITTER = "TREE-SITTER"


class DiffType(Enum):
    ADD = "ADD"
    REMOVE = "REMOVE"
    MODIFY = "MODIFY"


@dataclass
class DiffFacts:
    content: str  # git diff raw
    filename: str  # tree-sitter chunks
    namespace: str  # tree-sitter chunks
    added_headers: List[str]  # git diff parsing
    removed_headers: List[str]  # git diff parsing
    old_paths: List[str]  # git diff raw
    new_paths: List[str]  # git diff raw
    added_methods: List[str]  # tree-sitter queries
    removed_methods: List[str]  # tree-sitter queries
    static_identifiers: List[str]  # tree-sitter queries
    identifiers: List[str]  # Helper LLM
    joined_diffs: List[Tuple[str, str]]  # Keep method name next to diff
    chunks: List[str]  # source code split with tree-sitter chunks that match diff
    processing: Processing  # git diff parsing
    type: DiffType  # git diff kind ( add, remove, modify )


@dataclass
class FileSpecification:
    filename: str
    path: str
    source: str
    chunks: List["CppReader"]  # Assigned data type with no import
    diffs: List[DiffFacts]
    headers: List[str]
    prompts: List[Tuple[str, str, List[str]]]
    responses: List[Tuple[int, str]]


class DiffAnalyzer:
    HEADER_RE = re.compile(r'^\s*#\s*include\s*([<"])([^">]+)[">]')

    def __init__(self, branch: str, parent_branch: str, repository_path: str, cpp_parser):
        self.branch = branch
        self.parent_branch = parent_branch
        self.repository_path = repository_path
        self.cpp_parser = cpp_parser
        self.excludes = ("std::", "if", "else", "return")

        self.__setup_logger__()

    def __setup_logger__(self) -> logging:
        self.logger = logging
        self.logger.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.DEBUG
        )

    def extract_diff_facts(self, diff: str, filename: str) -> DiffFacts:
        added_headers, removed_headers = self.check_diff_for_headers(diff=diff)
        added_paths, removed_paths = self.check_diff_for_renames()
        added_methods, removed_methods = self.check_diff_for_added_removed_methods(diff=diff)

        return DiffFacts(
            content=diff,
            filename=filename,
            namespace="",  # FIXME -> to be verified if will be extracted from tree-sitter chunks, if not, remove
            added_headers=added_headers,
            removed_headers=removed_headers,
            old_paths=removed_paths,
            new_paths=added_paths,
            added_methods=added_methods,
            removed_methods=removed_methods,
            static_identifiers=self.extract_identifiers_from_code(diff=diff),
            identifiers=[],
            joined_diffs=[],
            chunks=[],
            processing=Processing.TREE_SITTER,
            type=self.check_diff_for_processing_method(diff=diff)  # FIXME -> currently not used, left for further development
        )

    def check_diff_for_headers(self, diff: AnyStr) -> Tuple:
        added_headers, removed_headers = list(), list()
        for ln in diff.splitlines():
            if ln.startswith('+++') or ln.startswith('---'):
                continue
            m = self.HEADER_RE.search(ln[1:])
            if m:
                sym = m.group(2)
                if ln.startswith('+') and '"' in ln:
                    added_headers.append(sym)
                elif ln.startswith('-') and '"' in ln:
                    removed_headers.append(sym)
                continue

        self.logger.info(f"Extracted headers: {list(set(added_headers))}")
        return added_headers, removed_headers

    def check_diff_for_renames(self) -> Tuple:
        res = subprocess.run(
            ["git", "diff", "--name-status", "-M", f"{self.parent_branch}..{self.branch}"],
            cwd=self.repository_path,
            text=True,
            capture_output=True,
            check=True
        ).stdout.splitlines()

        old_paths, new_paths = [], []
        for row in res:
            if row.startswith('R'):
                _, old, new = row.split('\t')
                old_paths.append(old)
                new_paths.append(new)

        return new_paths, old_paths

    def check_diff_for_single_file(self, diff_context: int, file: str) -> AnyStr:
        return subprocess.run(
            ["git", "diff", f"-U{diff_context}", f"{self.parent_branch}..{self.branch}", "--", file],
            check=True,
            capture_output=True,
            text=True,
            cwd=self.repository_path
        ).stdout

    def check_diff_for_changed_files(self) -> List[AnyStr]:
        return subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMRT", f"{self.parent_branch}..{self.branch}"],
            check=True,
            capture_output=True,
            text=True,
            cwd=self.repository_path
        ).stdout.splitlines()

    @staticmethod
    def split_diff_into_added_and_removed(diff: AnyStr, raw_output: bool = False) -> Tuple:
        added_lines = []
        removed_lines = []
        for line in diff.splitlines():
            if line.startswith(("diff ", "index ", "---", "+++", "@@")):
                continue

            if line.startswith("+"):
                added_lines.append(line[1:])
            elif line.startswith("-"):
                removed_lines.append(line[1:])

        if raw_output:
            return added_lines, removed_lines
        else:
            return "\n".join(added_lines), "\n".join(removed_lines)

    def check_diff_for_added_removed_methods(self, diff) -> Tuple:
        added, removed = self.split_diff_into_added_and_removed(diff=diff)
        return self.extract_identifiers_from_code(diff=added), self.extract_identifiers_from_code(diff=removed)

    def check_diff_for_processing_method(self, diff: str) -> DiffType:
        adds, dels = self.split_diff_into_added_and_removed(diff=diff, raw_output=True)
        if len(adds) > 0 and len(dels) == 0:
            return DiffType.ADD
        elif len(dels) > 0 and len(adds) == 0:
            return DiffType.REMOVE
        elif len(adds) > 0 and len(dels) > 0:
            return DiffType.MODIFY

    def extract_identifiers_from_code(self, diff: AnyStr) -> List[AnyStr]:
        query = Query(TS.Language(TSCpp.language()), CPP_DECLS_CALLS_QUERY)
        names = []
        for key, val in query.captures(self.cpp_parser.parse(diff).root_node).items():
            for v in val:
                names.extend(diff[v.start_byte:v.end_byte].split("::"))
        self.logger.info(f"Extracted static identifiers: {list(set(names))}")
        return list(set(names))

    def filter_identifiers(self, identifiers: List[AnyStr]) -> List[AnyStr]:
        return [i for i in identifiers if isinstance(i, str) and not any(e in i for e in self.excludes)]

    @staticmethod
    def check_diff_for_includes(diff: str) -> List[str]:
        return re.findall(
            pattern=r'^[+\-\s]*#include\s*[<"]([^">]+)[">]',
            string=diff,
            flags=re.MULTILINE
        )
