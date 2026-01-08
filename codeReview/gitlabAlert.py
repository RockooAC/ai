#!/usr/bin/python3


import requests
import argparse
from typing import NoReturn

from utils import Utils


def parser():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--change_id",
        required=True,
        type=str,
        metavar="Git change ID"
    )
    args.add_argument(
        "--git_password",
        required=True,
        type=str,
        metavar="Git CI/CD user API password"
    )
    args.add_argument(
        "--repository_name",
        required=True,
        type=str,
        metavar="Git repository name"
    )
    args.add_argument(
        "--message",
        required=True,
        type=str,
        metavar="Git comment message"
    )
    return args.parse_args()


class GitlabAlert(Utils):
    def __init__(self):
        super().__init__()
        self.args = parser()

    def main(self) -> NoReturn:
        r = requests.post(
            url=f"{self.gitlab_url}/api/v4/projects/{self.gitlab_repository_id_mapping[self.args.repository_name]}/merge_requests/{self.args.change_id}/notes",
            headers={
                "PRIVATE-TOKEN": self.args.git_password,
                "Content-Type": "application/json"
            },
            json={
                "body": self.args.message,
            },
            timeout=30
        )
        self.logger.info(f"Sent alert message to gitlab with status: {r.status_code}")


if __name__ == "__main__":
    GitlabAlert().main()
