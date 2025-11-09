import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import re
import shlex
from typing import Dict
from src.model.telegram_bot import CommandSpec, ParsedCommand


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
                # Keep positionals in original case for now, will handle case conversion in business logic
                positionals.append(token)
            i += 1
        # Assign positionals to named fields if defined
        for idx, pname in enumerate(spec.positional):
            if idx < len(positionals):
                # Only tickers should be upper case
                args[pname] = positionals[idx]
        # Special case: if the only positional is 'tickers', assign all positionals as a list
        if spec.positional == ["tickers"]:
            args["tickers"] = [p.upper() for p in positionals]  # Convert tickers to uppercase
        # Special case: for alerts, schedules, and admin commands, convert action to lowercase
        elif command in ["alerts", "schedules", "admin"] and len(spec.positional) == 2:
            args["action"] = positionals[0].lower() if len(positionals) > 0 else None  # Convert action to lowercase
            args["params"] = positionals[1:] if len(positionals) > 1 else []
        return ParsedCommand(
            command=command,
            args=args,
            positionals=positionals,
            raw_args=tokens,
            extra_flags=extra_flags
        )

# Example command specs for extensibility
COMMAND_SPECS = {
    "start": CommandSpec(
        parameters={
            "email": bool,
        },
        defaults={
            "email": False,
        },
        positional=[]
    ),
    "help": CommandSpec(
        parameters={
            "email": bool,
        },
        defaults={
            "email": False,
        },
        positional=[]
    ),
    "info": CommandSpec(
        parameters={
            "email": bool,
        },
        defaults={
            "email": False,
        },
        positional=[]
    ),
    "report": CommandSpec(
        parameters={
            "email": bool,
            "indicators": str,
            "period": str,
            "interval": str,
            "provider": str,
            "config": str,
        },
        defaults={
            "email": False,
            "indicators": None,
            "period": None,
            "interval": None,
            "provider": None,
            "config": None,
        },
        positional=["tickers"]
    ),
    "admin": CommandSpec(
        parameters={
            "email": bool,
        },
        defaults={
            "email": False,
        },
        positional=["action", "params"]
    ),
    "alerts": CommandSpec(
        parameters={
            "email": bool,
            "timeframe": str,
            "action_type": str,
            "config": str,
        },
        defaults={
            "email": False,
            "timeframe": "15m",
            "action_type": "notify",
            "config": None,
        },
        positional=["action", "params"]
    ),
    "schedules": CommandSpec(
        parameters={
            "email": bool,
            "indicators": str,
            "period": str,
            "interval": str,
            "provider": str,
            "config": str,
        },
        defaults={
            "email": False,
            "indicators": None,
            "period": None,
            "interval": None,
            "provider": None,
            "config": None,
        },
        positional=["action", "params"]
    ),
    "screener": CommandSpec(
        parameters={
            "email": bool,
            "immediate": bool,
        },
        defaults={
            "email": False,
            "immediate": True,
        },
        positional=["screener_name_or_config"]
    ),
    "register": CommandSpec(
        parameters={
            "email": bool,
        },
        defaults={
            "email": False,
        },
        positional=["email_address", "language"]
    ),
    "verify": CommandSpec(
        parameters={
            "email": bool,
        },
        defaults={
            "email": False,
        },
        positional=["verification_code"]
    ),
    "language": CommandSpec(
        parameters={
            "email": bool,
        },
        defaults={
            "email": False,
        },
        positional=["language_code"]
    ),
    "feedback": CommandSpec(
        parameters={
            "email": bool,
        },
        defaults={
            "email": False,
        },
        positional=["message"]
    ),
    "feature": CommandSpec(
        parameters={
            "email": bool,
        },
        defaults={
            "email": False,
        },
        positional=["message"]
    ),
    "request_approval": CommandSpec(
        parameters={
            "email": bool,
        },
        defaults={
            "email": False,
        },
        positional=[]
    ),
}

# For backward compatibility
parse_command = EnterpriseCommandParser(COMMAND_SPECS).parse
