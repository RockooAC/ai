import uuid
from pydantic import BaseModel
from typing import Optional

class Filter:
    class Valves(BaseModel):
        pass 
    
    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Extract metadata and chat_id
        metadata = body.get("metadata", {})
        chat_id = metadata.get("chat_id", "UNKNOWN")
        # Add the chat_id as a custom header
        body["x-chat-id"] = chat_id
        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        return body