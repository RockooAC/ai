#!/usr/bin/python3


import re
import os
import sys
import requests
import argparse
from utils import Utils
from pathlib import Path
from typing import NoReturn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.readers.cpp import CppReader, CppParser
from rag.embedding.prod.ingestionPipeline import ZosIngestion
from diffAnalyzer import DiffAnalyzer, FileSpecification, Processing


def parser():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--repo_path",
        type=str,
        required=True,
        help="Provide path to Git repository"
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
        self.diff_context = 25
        self.responses = dict()
        self.documents = list()
        self.files: dict[str, FileSpecification] = {}
        self.fetch_branch = "ai_review"

        self.ingestion = ZosIngestion(
            collection_name=self.decide_qdrant_tmp_collection_name(self.args.commit_message),
            cache_collection_name=self.decide_qdrant_tmp_collection_name(self.args.commit_message)
        )

        self.diff_analyzer = DiffAnalyzer(
            branch=self.fetch_branch,
            parent_branch=self.args.target_branch,
            repository_path=self.args.repo_path,
            cpp_parser=CppParser()

        )

    def main(self) -> NoReturn:
        self._get_modified_files()
        self._read_files_diff()

        self._decide_diff_processing()
        self._process_cpp_code()

        self._create_cpp_code_documents_for_embedding()
        self._trigger_code_embedding()

        self._retrieve_includes_chunk_for_file()
        self._create_prompts()
        self._send_queries()

        if self.args.repository_type == "gerrit":
            self._send_gerrit_response()
        else:
            self._send_gitlab_response()

        self.remove_temporary_qdrant_collection(
            collection_name=self.decide_qdrant_tmp_collection_name(self.args.commit_message)
        )

    def _get_modified_files(self) -> NoReturn:
        self.logger.debug(f"running method: _get_modified_files()")
        for file in self.get_changed_files(self.args.target_branch, self.fetch_branch, self.args.repo_path):
            if self.decide_filename_regex(filename=file):
                self.files[file] = FileSpecification(
                    filename=file.split("/")[-1],
                    path=file,
                    source="",
                    chunks=[],
                    diffs=[],
                    headers=[],
                    prompts=[],
                    responses=[],
                )

    def _read_files_diff(self) -> NoReturn:
        self.logger.debug(f"running method: _read_files_diff()")
        for file in self.files.keys():
            self.logger.info(f"Reading diffs for FILE: {file}")
            self.files[file].diffs = [
                self.diff_analyzer.extract_diff_facts(
                    diff=diff,
                    filename=file
                ) for diff in self.extract_single_diffs(
                    diff_text=self.check_diff_for_file(
                        self.diff_context,
                        self.args.target_branch,
                        self.fetch_branch,
                        file,
                        self.args.repo_path
                    )
                )
            ]

    def _decide_diff_processing(self) -> NoReturn:
        self.logger.debug(f"running method: _decide_diff_processing()")
        for file in self.files.keys():
            for diff in self.files[file].diffs:
                diff.processing = Processing.TREE_SITTER if self.check_chunksize(diff.content) > self.chunk_limit else Processing.SINGLE

    def _process_cpp_code(self) -> NoReturn:
        self.logger.debug(f"running method: _process_cpp_code()")
        for file in self.files.keys():
            file_path = Path(f"{self.args.repo_path}/{file}")
            self.files[file].chunks = CppReader(
                chunk_size=1024,
                chunk_overlap=256,
                debug=True,
                language="cpp"
            ).load_file(
                file=file_path
            )

            with file_path.open("r", encoding="utf-8") as f:
                self.files[file].source = f.read()

    def _retrieve_includes_chunk_for_file(self):
        self.logger.debug(f"running method: _retrieve_includes_chunk_for_file()")
        for file in self.files.keys():
            chunks = [c for c in self.files[file].chunks if c.start == 0]
            if chunks:
                include_chunk = [c for c in self.files[file].chunks if c.start == 0][0]
                diff = self.files[file].source[include_chunk.start: include_chunk.end]
                self.files[file].headers = self.diff_analyzer.check_diff_for_includes(diff=diff)

    def _create_cpp_code_documents_for_embedding(self) -> NoReturn:
        self.logger.debug(f"running method: _create_cpp_code_documents_for_embedding()")
        for file in self.files.keys():
            for chunk in self.files[file].chunks:
                doc = chunk.to_document(self.files[file].source, extra_info=None)
                if doc:
                    doc.metadata["file_path"] = re.sub(r'^([^/]+/){3}', '', file)  # match with digest based created embeddings
                    self.documents.append(doc)

    def _trigger_code_embedding(self) -> NoReturn:
        self.logger.debug(f"running method: _trigger_code_embedding()")
        nodes = self.ingestion.__ingestion_pipeline__(
            transformations=[
                self.ingestion.__doc_splitter__,
                self.ingestion.__embedder_jina__,
            ],
            vector_store=self.ingestion.__bm42_hybrid_vector_store__,
            docstore=None,
            cache=None
        ).run(
            show_progress=True,
            documents=self.documents
        )
        self.logger.info(f"Ingested {len(nodes)} nodes")

    def _create_prompts(self) -> NoReturn:
        self.logger.debug(f"running method: _create_prompts()")
        for file in self.files.keys():
            for diff in self.files[file].diffs:
                if diff.processing == "SINGLE":
                    prompt_template = self.prompt_template_mapping(file=file)
                    self.files[file].prompts = [(prompt_template(diff=diff), self.extract_no_context_lines(diff=diff.content), diff.static_identifiers) for diff in
                                                self.files[file].diffs]
                else:
                    self.files[file].prompts.append((self.prompt_template_filler_chunk(code=diff.content), diff.content, diff.static_identifiers))
                    self.logger.info(f"Prompts for file: {file}")
                    for p in [(self.prompt_template_filler_chunk(code=code), code, diff.static_identifiers) for code in diff.chunks]:
                        self.logger.info(f"Prompt: {p}\n")

    def _send_queries(self) -> NoReturn:
        self.logger.debug(f"running method: _send_queries()")
        for file in self.files.keys():
            self.logger.debug(f"Filename: {file}")
            for prompt in self.files[file].prompts:
                git_line = self.extract_git_modified_line_from_diff(diff_prompt=prompt[0], context_lines=self.diff_context)
                response = self.pipeline_response(
                    prompt=prompt[0],
                    commit_collection_name=self.decide_qdrant_tmp_collection_name(self.args.commit_message),
                    raw_diff=prompt[1],
                    file_name=file,
                    static_identifiers=prompt[2],
                    headers=self.files[file].headers
                )
                self.logger.debug(f"LLM response for file {file} and line {git_line}-> {response}")
                if not any(indicator in response for indicator in self.no_issues_indicators):
                    trimmed_response = response.split("\n\n---\n***Times:***\n")[0]  # Cut off debug "Times" section
                    self.files[file].responses.append((git_line, trimmed_response))

    def _send_gerrit_response(self) -> NoReturn:
        self.logger.debug(f"running method: _send_gerrit_response()")
        self.logger.info(f"Sending message to gerrit change: {self.args.change_number}")
        host, gerrit_encoded_repo = self.gerrit_mapping(team=self.args.team)
        if any(det.responses for det in self.files.values()):
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
                            } for review in details.responses
                        ] for file, details in self.files.items() if details.responses
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
        self.logger.debug(f"running method: _send_gitlab_response()")
        self.logger.info(f"Sending message to gitlab merge request: {self.args.change_number}")
        if any(det.responses for det in self.files.values()):
            hash_details = self.get_merge_request_hash_details(
                project_id=self.gitlab_repository_id_mapping[self.args.repository_name],
                mr_id=self.args.change_id,
                token=self.args.git_password
            )
            for file, details in self.files.items():
                if not details.responses:
                    continue
                for review in details.responses:
                    r = requests.post(
                        url=f"{self.gitlab_url}/api/v4/projects/{self.gitlab_repository_id_mapping[self.args.repository_name]}/merge_requests/{self.args.change_id}/discussions",
                        headers={
                            "PRIVATE-TOKEN": self.args.git_password,
                            "Content-Type": "application/json"
                        },
                        json={
                            "body": f"""{review[1]}""",
                            "position": {
                                "position_type": "text",
                                "base_sha": hash_details["diff_refs"]["base_sha"],
                                "start_sha": hash_details["diff_refs"]["start_sha"],
                                "head_sha": hash_details["diff_refs"]["head_sha"],
                                "new_path": details.path,
                                "new_line": review[0]
                            }
                        },
                        timeout=30
                    )
                    self.logger.info(f"Sent message file: {file}, line: {review[0]} to gitlab with status: {r.status_code}")

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

    # FIXME -> TO NOT USE UNTIL GITLAB VERSION UPGRADE TO 16.X, if needed refactor for dataclass based flow
    # def _send_gitlab_response_bulk_drafts(self) -> NoReturn:
    #     self.logger.info(f"[GitLab] Preparing draft comments for MR !{self.args.change_id}")
    #     if any(det.responses for det in self.files.values()):
    #         hash_details = self.get_merge_request_hash_details(
    #             project_id=self.gitlab_repository_id_mapping[self.args.repository_name],
    #             mr_id=self.args.change_id,
    #             token=self.args.git_password
    #         )
    #         session = requests.Session()
    #         session.headers.update({
    #             "PRIVATE-TOKEN": self.args.git_password,
    #             "Content-Type": "application/json",
    #         })
    #         for file_path, resp_list in self.files.items():
    #             if not resp_list:
    #                 continue
    #             for line_number, message in resp_list:
    #                 r = session.post(
    #                     url=f"{self.gitlab_url}/api/v4/projects/{self.gitlab_repository_id_mapping[self.args.repository_name]}/merge_requests/{self.args.change_id}/draft_notes",
    #                     json={
    #                         "note": message,
    #                         "position": {
    #                             "position_type": "text",
    #                             "base_sha": hash_details["diff_refs"]["base_sha"],
    #                             "start_sha": hash_details["diff_refs"]["start_sha"],
    #                             "head_sha": hash_details["diff_refs"]["head_sha"],
    #                             "new_path": file_path,
    #                             "new_line": line_number,
    #                         }
    #                     },
    #                     timeout=30
    #                 )
    #                 time.sleep(1)  # rate limits handling
    #                 self.logger.debug(f"[GitLab] Draft created with status code: {r.status_code}")
    #
    #         pr = session.post(
    #             url=f"{self.gitlab_url}/api/v4/projects/{self.gitlab_repository_id_mapping[self.args.repository_name]}/merge_requests/{self.args.change_id}/draft_notes/bulk_publish",
    #             timeout=30
    #         )
    #         if pr.ok:
    #             self.logger.info(f"[GitLab] Bulk publish success: {pr.status_code}")
    #         else:
    #             self.logger.error(f"[GitLab] Bulk publish failed: {pr.status_code} body={pr.text}")
    #             sys.exit(1)
    #
    #     else:
    #         r = requests.post(
    #             url=f"{self.gitlab_url}/api/v4/projects/{self.gitlab_repository_id_mapping[self.args.repository_name]}/merge_requests/{self.args.change_id}/notes",
    #             headers={
    #                 "PRIVATE-TOKEN": self.args.git_password,
    #                 "Content-Type": "application/json"
    #             },
    #             json={
    #                 "body": "No issues found by AI review.",
    #             },
    #             timeout=30
    #         )
    #         self.logger.info(f"Sent generic message to gitlab with status: {r.status_code}")


if __name__ == "__main__":
    ReviewManager().main()
