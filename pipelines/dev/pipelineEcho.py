"""
title: Echo Pipeline
description: A pipeline for return the source message
"""

from typing import List, Union, Generator, Iterator
from libs.tools import setup_logger

logger = setup_logger(name="Echo pipeline", debug=False)

class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):
        # This function is called when the server is started.
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        echo_body = f"Echo body: {body}"
        logger.info(echo_body)
        for i, m in enumerate(messages):
            echo_message = f"Echo message #{i}: {m}"
            logger.info(echo_message)
        echo_user_message = f"Echo user message: {user_message}"
        logger.info(echo_user_message)
        return user_message
