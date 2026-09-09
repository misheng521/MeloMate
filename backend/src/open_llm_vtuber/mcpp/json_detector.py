"""Bounded incremental JSON detection for the tool compatibility protocol."""
import json


class StreamJSONDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buffer = ""
        self.completed_jsons = []
        self.depth = 0
        self.quoted = False
        self.escaped = False
        self.oversized = False

    def process_chunk(self, chunk: str) -> list[dict]:
        found = []
        for char in chunk:
            if not self.depth:
                if char != "{":
                    continue
                self.depth = 1
                self.buffer = "{"
                continue
            if len(self.buffer) < 256000:
                self.buffer += char
            else:
                self.oversized = True
            if self.quoted:
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.quoted = False
            elif char == '"':
                self.quoted = True
            elif char == "{":
                self.depth += 1
            elif char == "}":
                self.depth -= 1
                if self.depth == 0:
                    if not self.oversized:
                        try:
                            value = json.loads(self.buffer)
                            found.append(value)
                            self.completed_jsons.append(value)
                            del self.completed_jsons[:-32]
                        except ValueError:
                            pass
                    self.buffer = ""
                    self.oversized = False
        return found

    def get_all_jsons(self):
        return list(self.completed_jsons)
