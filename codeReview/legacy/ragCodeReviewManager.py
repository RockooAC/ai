#!/usr/bin/python3


import sys
import time
import requests
import argparse
from utils import Utils
from typing import NoReturn


def parser():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--repo_path",
        type=str,
        required=True,
        help="Provide path to markdown documentation"
    )
    args.add_argument(
        "--commit_message",
        type=str,
        required=True,
        help="Git change commit message"
    )
    args.add_argument(
        "--change_number",
        required=True,
        type=str,
        metavar="Git change number"
    )
    args.add_argument(
        "--change_id",
        required=True,
        type=str,
        metavar="Git change ID"
    )
    args.add_argument(
        "--git_user",
        required=True,
        type=str,
        metavar="Git CI/CD user"
    )
    args.add_argument(
        "--git_password",
        required=True,
        type=str,
        metavar="Git CI/CD user API password"
    )
    args.add_argument(
        "--repository_type",
        required=True,
        type=str,
        choices=["gerrit", "gitlab"],
        metavar="Git repository provider"
    )
    args.add_argument(
        "--repository_name",
        required=True,
        type=str,
        metavar="Git repository name"
    )
    args.add_argument(
        "--target_branch",
        required=True,
        type=str,
        metavar="Git diff merge request target branch"
    )
    args.add_argument(
        "--team",
        required=False,
        type=str,
        default="zos",
        choices=["zos", "dsb"],
        metavar="Team name to map repository address"
    )
    return args.parse_args()


class ReviewManager(Utils):
    def __init__(self):
        super().__init__()

        self.args = parser()
        self.prompts = dict()
        self.diff_context = 5
        self.responses = dict()
        self.modified_files = None
        self.files_details = dict()
        self.fetch_branch = "ai_review"
        self.diff_handling = self.generic_category()
        self.queries_source = self.generic_category()

    def main(self) -> NoReturn:
        self.modified_files = self.decide_filenames_regex(
            filenames=self.get_changed_files(
                self.args.target_branch,
                self.fetch_branch,
                self.args.repo_path
            )
        )

        self._read_files_diff()
        self._decide_diff_processing()
        self._join_queries_source()
        self._create_prompts()
        self._send_queries()

        if self.args.repository_type == "gerrit":
            self._send_gerrit_response()
        else:
            self._send_gitlab_response()

    def _read_files_diff(self) -> NoReturn:
        for file in self.modified_files:
            self.logger.info(f"Reading diffs for FILE: {file}")
            self.files_details[file] = self.extract_single_diffs(
                diff_text=self.check_diff_for_file(
                    self.diff_context,
                    self.args.target_branch,
                    self.fetch_branch,
                    file,
                    self.args.repo_path
                )
            )

    def _decide_diff_processing(self) -> NoReturn:
        for file, diffs_list in self.files_details.items():
            for diff in diffs_list:
                category = "TREE-SITTER" if self.check_chunksize(diff) > self.chunk_limit else "SINGLE"
                self.diff_handling.setdefault(category, {}).setdefault(file, []).append(diff)

        self.logger.debug(f"Categorized diffs: {self.diff_handling}")

    def _join_queries_source(self) -> NoReturn:
        for filename, diffs in self.diff_handling["SINGLE"].items():
            self.queries_source["SINGLE"][filename] = diffs

    def _create_prompts(self) -> NoReturn:
        for key, values in self.queries_source.items():
            for file, details in values.items():
                if key == "SINGLE":
                    prompt_template = self.prompt_template_mapping(file=file)
                    self.prompts[file] = [prompt_template(diff=diff) for diff in details]
                else:
                    pass

    def _send_queries(self) -> NoReturn:
        for file, prompts in self.prompts.items():
            self.responses[file] = []
            for prompt in prompts:
                git_line = self.extract_git_modified_line_from_diff(diff_prompt=prompt, context_lines=self.diff_context)
                response = self.stream_pipeline_response(prompt=prompt)
                self.logger.debug(f"LLM response for file {file} and line {git_line}-> {response}")
                if self.no_issues_indicator not in response:
                    trimmed_response = response.split("\n\n---\n***Times:***\n")[0]  # Cut off debug "Times" section
                    self.responses[file].append((git_line, trimmed_response))

    def _send_gerrit_response(self) -> NoReturn:
        self.logger.info(f"Sending message to gerrit change: {self.args.change_number}")
        host, gerrit_encoded_repo = self.gerrit_mapping(team=self.args.team)
        if not all(len(r) == 0 for r in self.responses.values()):
            r = requests.post(
                url=f"{host}/a/changes/{gerrit_encoded_repo}~{self.args.target_branch}~{self.args.change_id}/revisions/current/review",
                auth=(self.args.git_user, self.args.git_password),
                json={
                    "tag": "ai-code-review",
                    "message": "AI code review response, few issues to consider",
                    "comments": {
                        file: [
                            {
                                "line": review[0],
                                "message": f"""{review[1]}""",
                                "unresolved": True
                            } for review in responses
                        ] for file, responses in self.responses.items() if responses
                    }
                },
                timeout=30
            )
            self.logger.info(f"Sent message to gerrit with status: {r.status_code}")
        else:
            r = requests.post(
                url=f"{host}/a/changes/{gerrit_encoded_repo}~{self.args.target_branch}~{self.args.change_id}/revisions/current/review",
                auth=(self.args.git_user, self.args.git_password),
                json={
                    "tag": "ai-code-review",
                    "message": "No AI review comments for this patch. Looks good!"
                },
                timeout=30
            )
            self.logger.info(f"Sent message to gerrit with status: {r.status_code}")

    def _send_gitlab_response(self) -> NoReturn:
        self.logger.info(f"Sending message to gitlab merge request: {self.args.change_number}")
        if any(self.responses.values()):
            hash_details = self.get_merge_request_hash_details(
                project_id=self.gitlab_repository_id_mapping[self.args.repository_name],
                mr_id=self.args.change_id,
                token=self.args.git_password
            )
            for file, responses in self.responses.items():
                if not responses:
                    continue
                for line_number, message in responses:
                    r = requests.post(
                        url=f"{self.gitlab_url}/api/v4/projects/{self.gitlab_repository_id_mapping[self.args.repository_name]}/merge_requests/{self.args.change_id}/discussions",
                        headers={
                            "PRIVATE-TOKEN": self.args.git_password,
                            "Content-Type": "application/json"
                        },
                        json={
                            "body": message,
                            "position": {
                                "position_type": "text",
                                "base_sha": hash_details["diff_refs"]["base_sha"],
                                "start_sha": hash_details["diff_refs"]["start_sha"],
                                "head_sha": hash_details["diff_refs"]["head_sha"],
                                "new_path": file,
                                "new_line": line_number
                            }
                        },
                        timeout=30
                    )
                    self.logger.info(f"Sent message file: {file}, line: {line_number} to gitlab with status: {r.status_code}")

        else:
            r = requests.post(
                url=f"{self.gitlab_url}/api/v4/projects/{self.gitlab_repository_id_mapping[self.args.repository_name]}/merge_requests/{self.args.change_id}/notes",
                headers={
                    "PRIVATE-TOKEN": self.args.git_password,
                    "Content-Type": "application/json"
                },
                json={
                    "body": "No issues found by AI review.",
                },
                timeout=30
            )
            self.logger.info(f"Sent generic message to gitlab with status: {r.status_code}")

    # FIXME -> TO NOT USE UNTIL GITLAB VERSION UPGRADE TO 16.X
    def _send_gitlab_response_bulk_drafts(self) -> NoReturn:
        self.logger.info(f"[GitLab] Preparing draft comments for MR !{self.args.change_id}")
        if any(self.responses.values()):
            hash_details = self.get_merge_request_hash_details(
                project_id=self.gitlab_repository_id_mapping[self.args.repository_name],
                mr_id=self.args.change_id,
                token=self.args.git_password
            )
            session = requests.Session()
            session.headers.update({
                "PRIVATE-TOKEN": self.args.git_password,
                "Content-Type": "application/json",
            })
            for file_path, resp_list in self.responses.items():
                if not resp_list:
                    continue
                for line_number, message in resp_list:
                    r = session.post(
                        url=f"{self.gitlab_url}/api/v4/projects/{self.gitlab_repository_id_mapping[self.args.repository_name]}/merge_requests/{self.args.change_id}/draft_notes",
                        json={
                            "note": message,
                            "position": {
                                "position_type": "text",
                                "base_sha": hash_details["diff_refs"]["base_sha"],
                                "start_sha": hash_details["diff_refs"]["start_sha"],
                                "head_sha": hash_details["diff_refs"]["head_sha"],
                                "new_path": file_path,
                                "new_line": line_number,
                            }
                        },
                        timeout=30
                    )
                    time.sleep(1)  # rate limits handling
                    self.logger.debug(f"[GitLab] Draft created with status code: {r.status_code}")

            pr = session.post(
                url=f"{self.gitlab_url}/api/v4/projects/{self.gitlab_repository_id_mapping[self.args.repository_name]}/merge_requests/{self.args.change_id}/draft_notes/bulk_publish",
                timeout=30
            )
            if pr.ok:
                self.logger.info(f"[GitLab] Bulk publish success: {pr.status_code}")
            else:
                self.logger.error(f"[GitLab] Bulk publish failed: {pr.status_code} body={pr.text}")
                sys.exit(1)

        else:
            r = requests.post(
                url=f"{self.gitlab_url}/api/v4/projects/{self.gitlab_repository_id_mapping[self.args.repository_name]}/merge_requests/{self.args.change_id}/notes",
                headers={
                    "PRIVATE-TOKEN": self.args.git_password,
                    "Content-Type": "application/json"
                },
                json={
                    "body": "No issues found by AI review.",
                },
                timeout=30
            )
            self.logger.info(f"Sent generic message to gitlab with status: {r.status_code}")


if __name__ == "__main__":
    ReviewManager().main()
