#!/usr/bin/python3

import re
import os
import sys
import time
import json
import logging
import requests
import subprocess
from typing import List, AnyStr, Union, Dict, NoReturn, Tuple


class Utils:
    def __init__(self):
        self.chunk_limit = 10  # FIXME -> modify is assumption of handling all diff in tree-sitter fails
        self.gerrit_port = 29418
        self.text_to_chunk_const = 1.3
        self.diff_header = "diff --git"
        self.golang_extension = ".*\.go"
        self.qdrant_url = "http://10.255.240.18:6333"
        self.pipeline_name = "pipelineReviewAssistant"
        self.gitlab_url = "https://drm-gitlab.redlabs.pl"
        self.pipelines_url = "http://10.255.240.156:9099"
        self.no_issues_indicator = "No issues found [TO BE EXCLUDED]"
        self.cpp_extension = r'(.*\.(h|cpp|cmake)$|^CMakeLists\.txt$)'
        self.no_issues_indicators = ["[TO BE EXCLUDED]", "Empty Response"]
        self.pipelines_api_key = os.getenv("PIPELINES_KEY", "<placeholder>")
        self.filename_regex = r'(.*\.(h|cpp|cmake)$|^CMakeLists\.txt$)'
        # self.filename_regex = r'(.*\.(h|cpp|cmake|py|sh|yml|yaml|go)$|^CMakeLists\.txt$)'

        self.__setup_logger__()

    def __setup_logger__(self) -> logging:
        self.logger = logging
        self.logger.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.DEBUG
        )

    @staticmethod
    def gerrit_mapping(team: AnyStr) -> Tuple:
        if team == "zos":
            return "https://git.atendesoftware.pl", "atds%2Fdrm%2Fzos%2Fred%2Dgalaxy%2Fcore"
        else:
            return "https://gerrit.redgelabs.com", "dsb%2Fdataplane"

    def get_changed_files(self, parent_branch: str, compare_branch: str, repo_path: str) -> List[AnyStr]:
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
            self.logger.info(f"Modified files from git diff: {result}")
            return result.split('\n') if result else []
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Getting modified files from diff error: {e}")
            sys.exit(1)

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

        ### If you find problems
        Return bullet-point comments – **do NOT mention the sentinel**.
        
        ### If there are **no** problems
        Respond with **exactly** the fenced block below – no text before or after:
    
        ```sentinel
        {self.no_issues_indicator}
        ```

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

        ### If you find problems
        Return bullet-point comments – **do NOT mention the sentinel**.
        
        ### If there are **no** problems
        Respond with **exactly** the fenced block below – no text before or after:

        ```sentinel
        {self.no_issues_indicator}
        ```

        ---

        Git Diff:
        {diff}
        """.strip()

    def prompt_template_filler(self, diff: AnyStr) -> AnyStr:
        return f"""You are a Senior C++ Developer reviewing a git diff.
        Your job is to find real engineering issues — not stylistic suggestions or summaries.

        Identify:
        • real bugs, undefined behavior, missed edge cases
        • concurrency errors, unsafe memory, incorrect logic
        • idiomatic C++ improvements **only if they fix a real issue or improve clarity**

        Do **NOT**:
        • Invent advice just to fill space
        • Mention testing or performance (unless the diff directly affects them)
        • Summarise the change — only comment on concrete technical issues

        ---

        ### If you find problems
        Return bullet-point comments – **do NOT mention the sentinel**.

        ### If there are **no** problems
        Respond with **exactly** the fenced block below – no text before or after:

        ```sentinel
        {self.no_issues_indicator}
        ```

        ---

        Git Diff:
        {diff}
        """.strip()


    def prompt_template_filler_chunk(self, code: AnyStr) -> AnyStr:
        return f"""
        You are reviewing a C++ function or class as a senior developer.

        TASK:
        - Only identify real, explicit technical bugs:
        * Crashes, undefined behavior
        * Memory leaks, use-after-free, double free
        * Data races or concurrency issues
        * Incorrect logic causing wrong results
        - Ignore all stylistic, idiomatic, or optional improvements

        IMPORTANT:
        - Lines starting with '+' are added, '-' are removed
        - Lines without '+' or '-' are context and should NOT be commented on
        - Only comment on explicit issues in added or removed lines

        DO NOT:
        - Comment on style, naming, formatting, or idiomatic usage
        - Suggest testing  or documentation improvements
        - Summarize what the code does

        ---
        
        ### Response format (STRICT)
        * If you find any issue at all, return concrete, actionable observations – do NOT include the phrase "{self.no_issues_indicator}".
        * If the code is completely clean, respond with **exactly** the single line below – nothing before or after, no punctuation, no formatting:
        
        {self.no_issues_indicator}

        ---
        Code to review:
        ```
        {code}
        ```
        """



    def prompt_template_mapping(self, file: AnyStr):
        if re.match(self.cpp_extension, file):
            return self.prompt_template_filler
        elif re.match(self.golang_extension, file):
            return self.prompt_template_filler_golang
        else:
            return self.prompt_template_filler_generic

    @staticmethod
    def decide_qdrant_tmp_collection_name(commit_message) -> AnyStr:
        match = re.search(r'\[(\d{4})\].*', commit_message)
        if match:
            return f"gerrit_review_{match.group(1)}"
        else:
            return f"gerrit_review_{time.time()}"

    def decide_filename_regex(self, filename: AnyStr) -> bool:
        return bool(re.match(self.filename_regex, filename))

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

    @staticmethod
    def extract_no_context_lines(diff: str) -> AnyStr:
        matches = re.findall(r'^(?:\+|-)(?!\+\+\+|---).+', diff, flags=re.MULTILINE)
        return '\n'.join(matches)

    def stream_pipeline_response(self,
                                 prompt: AnyStr,
                                 commit_collection_name: Union[AnyStr, None] = None,
                                 raw_diff: Union[AnyStr, None] = None,
                                 headers: Union[AnyStr, None] = None
                                 ) -> AnyStr:
        output = ""
        body = {
            "model": self.pipeline_name,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
            "env": {}
        }
        if commit_collection_name:
            body["env"]["COMMIT_COLLECTION"] = commit_collection_name

        if raw_diff:
            body["env"]["RAW_DIFF"] = raw_diff

        if headers:
            body["env"]["HEADERS"] = headers

        with requests.post(
                url=f"{self.pipelines_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.pipelines_api_key}",
                    "Content-Type": "application/json"
                },
                json=body,
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

    def pipeline_response(self,
                          prompt: AnyStr,
                          commit_collection_name: Union[AnyStr, None] = None,
                          raw_diff: Union[AnyStr, None] = None,
                          file_name: Union[AnyStr, None] = None,
                          max_attempts: int = 2,
                          static_identifiers: Union[List, None] = None,
                          headers: Union[List, None] = None
                          ) -> AnyStr:
        body = {
            "model": self.pipeline_name,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "env": {}
        }

        if commit_collection_name:
            body["env"]["COMMIT_COLLECTION"] = commit_collection_name
        if raw_diff:
            body["env"]["RAW_DIFF"] = raw_diff
        if file_name:
            body["env"]["FILE_NAME"] = file_name
        if static_identifiers:
            body["env"]["STATIC_IDENTIFIERS"] = static_identifiers
        if headers:
            body["env"]["HEADERS"] = headers

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    url=f"{self.pipelines_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.pipelines_api_key}",
                        "Content-Type": "application/json"
                    },
                    json=body,
                    timeout=180
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                self.logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    break
                time.sleep(5)
        self.logger.error(f"Pipeline request failed after {max_attempts} attempts.")
        return self.no_issues_indicator

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

    def remove_temporary_qdrant_collection(self, collection_name: AnyStr) -> NoReturn:
        response = requests.delete(
            f"{self.qdrant_url}/collections/{collection_name}",
            timeout=30
        )
        self.logger.info(f"Removed qdrant collection: {collection_name} with HTTP code -> {response.status_code}")