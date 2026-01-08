#!/usr/bin/python3

import re
import os
import json
import logging
import requests
import subprocess
from typing import List, AnyStr, Union, Dict


class Utils:
    def __init__(self):
        self.chunk_limit = 2048
        self.text_to_chunk_const = 1.3
        self.filename_regex = r'(.*\.(h|cpp|cmake|py|sh|yml|yaml|go)$|^CMakeLists\.txt$)'
        self.pipelines_url = "http://10.255.240.156:9099"
        self.pipelines_api_key = os.getenv("PIPELINES_KEY", "<placeholder>")
        self.pipeline_name = "pipelineCodeAssistant"
        self.gitlab_url = "https://drm-gitlab.redlabs.pl"
        self.diff_header = "diff --git"
        self.no_issues_indicator = "No issues found [TO BE EXCLUDED]"
        self.golang_extension = ".*\.go"
        self.cpp_extension = r'(.*\.(h|cpp|cmake)$|^CMakeLists\.txt$)'
        self.__setup_logger__()

    def __setup_logger__(self) -> logging:
        self.logger = logging
        self.logger.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.DEBUG
        )

    @staticmethod
    def gerrit_mapping(team: AnyStr):
        if team == "zos":
            return "https://git.atendesoftware.pl", "atds%2Fdrm%2Fzos%2Fred%2Dgalaxy%2Fcore"
        else:
            return "https://gerrit.redgelabs.com", "dsb%2Fdataplane"

    @staticmethod
    def generic_category() -> Dict[AnyStr, Dict]:
        return {
            "SINGLE": {},
            "TREE-SITTER": {}
        }

    @staticmethod
    def get_changed_files(parent_branch: str, compare_branch: str, repo_path: str) -> List[AnyStr]:
        """
        Returns a list of file paths changed between two Git branches.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACMRT", f"{parent_branch}..{compare_branch}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=repo_path
            ).stdout.strip()

            return result.split('\n') if result else []
        except subprocess.CalledProcessError as e:
            return []

    @staticmethod
    def check_diff_for_file(diff_context: int, parent_branch: str, compare_branch: str, file: str, repo_path: str) -> AnyStr:
        """
        Prints the diff for single file.
        """
        return subprocess.run(
            ["git", "diff", f"-U{diff_context}", f"{parent_branch}..{compare_branch}", "--", file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=repo_path
        ).stdout

    def extract_single_diffs(self, diff_text: AnyStr) -> List:
        parts = re.split(r'(?=^@@)', diff_text, flags=re.MULTILINE)
        return [part.strip() for part in parts if part.strip() and self.diff_header not in part]

    def check_chunksize(self, diff_string: AnyStr) -> Union[int, float]:
        """
        If calculated chunksize is larger than limit, current file will be processed by tree-sitter
        """
        return len(diff_string.split()) * self.text_to_chunk_const

    def prompt_template_filler_generic(self, diff: AnyStr) -> AnyStr:
        return f"""You are a Senior Software Engineer reviewing a git diff.
        Identify:
        • real bugs, undefined behavior, missed edge cases  
        • concurrency errors, unsafe memory, incorrect logic  

        Do **NOT**:
        • Invent advice just to fill space 
        • Mention testing or performance (unless the diff directly affects them)  
        • Summarise the change — only comment on concrete technical issues

        ---

        ### Response format (STRICT)

        * If you find **any issue at all**, return concrete, actionable observations – **do NOT include the phrase "{self.no_issues_indicator}"**.
        * If the diff is completely clean, respond with **exactly** the single line below – nothing before or after, no punctuation, no formatting:

        {self.no_issues_indicator}

        ---

        Git Diff:
        {diff}
        """.strip()

    def prompt_template_filler_golang(self, diff: AnyStr) -> AnyStr:
        return f"""You are a Senior Go (Golang) Developer reviewing a git diff.
        Identify:
        • real bugs, nil dereferences, missed edge cases  
        • concurrency errors (goroutines, channels, race conditions), incorrect logic  
        • idiomatic Go improvements (e.g. error handling, use of interfaces, slice handling) **only if clearly useful**


        Do **NOT**:
        • Invent advice just to fill space 
        • Mention testing or performance (unless the diff directly affects them)  
        • Summarise the change — only comment on concrete technical issues

        ---

        ### Response format (STRICT)

        * If you find **any issue at all**, return concrete, actionable observations – **do NOT include the phrase "{self.no_issues_indicator}"**.
        * If the diff is completely clean, respond with **exactly** the single line below – nothing before or after, no punctuation, no formatting:

        {self.no_issues_indicator}

        ---

        Git Diff:
        {diff}
        """.strip()

    def prompt_template_filler(self, diff: AnyStr) -> AnyStr:
        return f"""You are a Senior C++ Developer reviewing a git diff.
        Identify:
        • real bugs, undefined behavior, missed edge cases  
        • concurrency errors, unsafe memory, incorrect logic  
        • idiomatic C++ improvements **only if truly useful**

        Do **NOT**:
        • Invent advice just to fill space 
        • Mention testing or performance (unless the diff directly affects them)  
        • Summarise the change — only comment on concrete technical issues

        ---

        ### Response format (STRICT)

        * If you find **any issue at all**, return concrete, actionable observations – **do NOT include the phrase "{self.no_issues_indicator}"**.
        * If the diff is completely clean, respond with **exactly** the single line below – nothing before or after, no punctuation, no formatting:

        {self.no_issues_indicator}

        ---

        Git Diff:
        {diff}
        """.strip()

    @staticmethod
    def prompt_template_filler_chunk(code: AnyStr) -> AnyStr:
        return f"""
        You are reviewing a C++ function or class as a senior developer.

        Your task:
        - Identify bugs, logic errors, or undefined behavior
        - Flag unsafe memory usage, data races, or API misuse
        - Point out violations of C++ best practices only if truly meaningful

        Important instructions:
        - Lines starting with '+' were added.
        - Lines starting with '-' were removed.
        - Lines without '+' or '-' are context only and should **not** be commented on.
        - Focus your review exclusively on added or removed lines, even if surrounded by context.
        - Do not summarize unchanged code.
        - Do not comment just to fill space.

        Code diff with 25 lines of context before and after each change:
        {code}

        If no issues exist in the modified lines, respond exactly with:
        No issues found.
        """

    def prompt_template_mapping(self, file: AnyStr):
        if re.match(self.cpp_extension, file):
            return self.prompt_template_filler
        elif re.match(self.golang_extension, file):
            return self.prompt_template_filler_golang
        else:
            return self.prompt_template_filler_generic

    def decide_filenames_regex(self, filenames: List[AnyStr]) -> List[AnyStr]:
        return [f for f in filenames if re.match(self.filename_regex, f)]

    @staticmethod
    def extract_changed_lines_from_diff(diff: AnyStr) -> List:
        changed_lines = set()
        current_new_line = 0

        for line in diff.splitlines():
            if line.startswith('@@'):
                match = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    current_new_line = int(match.group(1))
            elif line.startswith('+') and not line.startswith('+++'):
                changed_lines.add(current_new_line)
                current_new_line += 1
            elif not line.startswith('-'):
                current_new_line += 1

        return sorted(list(changed_lines))

    def stream_pipeline_response(self, prompt: AnyStr) -> AnyStr:
        output = ""
        with requests.post(
                url=f"{self.pipelines_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.pipelines_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.pipeline_name,
                    "stream": True,
                    "messages": [{"role": "user", "content": prompt}]
                },
                stream=True
        ) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    output += delta["content"]
        return output

    @staticmethod
    def extract_git_modified_line_from_diff(diff_prompt: AnyStr, context_lines: int = 10) -> int:
        match = re.search(r'\@\@ -\d+(?:,\d+)? \+(\d+)', diff_prompt)
        if match:
            new_start_line = int(match.group(1))
            return new_start_line + context_lines
        return 1

    @property
    def gitlab_repository_id_mapping(self) -> Dict[AnyStr, int]:
        return {
            "rm-cdns-go": 764,
            "rm-origin-vod-go": 652,
            "rg-xk6": 509,
            "rg-cdn-scaler-agent-go": 434,
            "rg-libs-go": 433,
            "rm-scte-proxy": 1015
        }

    def get_merge_request_hash_details(self, project_id: int, mr_id: int, token: AnyStr) -> Dict:
        return requests.get(
            url=f"{self.gitlab_url}/api/v4/projects/{project_id}/merge_requests/{mr_id}",
            headers={
                "PRIVATE-TOKEN": token
            }
        ).json()
