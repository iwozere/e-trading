from dataclasses import dataclass
from typing import Dict, Optional, Type
import shlex

@dataclass
class CommandSpec:
    parameters: Dict[str, Type]  # param_name: type
    defaults: Dict[str, any]     # default values

class CommandParser:
    def __init__(self, command_spec: CommandSpec):
        self.spec = command_spec

    def parse(self, text: str) -> Dict[str, any]:
        tokens = shlex.split(text)[1:]  # Remove command token
        args = {}

        for token in tokens:
            if token.startswith("--"):
                key, value = token[2:].split("=", 1)
                args[key] = self._cast_value(key, value)

        return {**self.spec.defaults, **args}

    def _cast_value(self, key: str, value: str):
        target_type = self.spec.parameters.get(key)
        return target_type(value) if target_type else value

class CommandHandler:
    def __init__(self, parser: CommandParser):
        self.parser = parser

    def execute(self, command_text: str) -> any:
        parsed_args = self.parser.parse(command_text)
        # Business logic here
        return parsed_args