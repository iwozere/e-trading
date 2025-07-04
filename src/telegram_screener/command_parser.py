import re
import shlex
from typing import Dict
from src.model.model import CommandSpec, ParsedCommand


class EnterpriseCommandParser:
    def __init__(self, command_specs: Dict[str, CommandSpec]):
        self.command_specs = command_specs

    def parse(self, message_text: str) -> ParsedCommand:
        tokens = shlex.split(message_text)
        if not tokens:
            return ParsedCommand(command="", raw_args=[])
        command = tokens[0].lstrip("/").lower()
        spec = self.command_specs.get(command, CommandSpec(parameters={}, defaults={}, positional=[]))
        args = dict((k.lower(), v) for k, v in spec.defaults.items())
        positionals = []
        extra_flags = {}
        i = 1
        while i < len(tokens):
            token = tokens[i]
            # Support -flag value, --flag value, -flag=value, --flag=value
            m = re.match(r"^(-{1,2})([\w-]+)(=(.*))?$", token)
            if m:
                flag = m.group(2).replace("-", "_").lower()
                value = m.group(4)
                if value is None and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    value = tokens[i + 1]
                    i += 1
                # Always lower case parameter values except for tickers
                if value is not None:
                    value = value.lower()
                if flag in spec.parameters:
                    try:
                        args[flag] = spec.parameters[flag](value) if value is not None else True
                    except Exception:
                        args[flag] = value
                else:
                    extra_flags[flag] = value if value is not None else True
            elif not token.startswith("-"):
                # Only tickers should be upper case
                positionals.append(token.upper())
            i += 1
        # Assign positionals to named fields if defined
        for idx, pname in enumerate(spec.positional):
            if idx < len(positionals):
                # Only tickers should be upper case
                args[pname] = positionals[idx]
        return ParsedCommand(
            command=command,
            args=args,
            positionals=positionals,
            raw_args=tokens,
            extra_flags=extra_flags
        )

# Example command specs for extensibility
COMMAND_SPECS = {
    "report": CommandSpec(
        parameters={
            "email": bool,
            "indicators": str,
            "period": str,
            "interval": str,
            "provider": str,
        },
        defaults={
            "email": False,
            "indicators": None,
            "period": None,
            "interval": None,
            "provider": None,
        },
        positional=["tickers"]
    ),
    # Add more command specs as needed
}

# For backward compatibility
parse_command = EnterpriseCommandParser(COMMAND_SPECS).parse