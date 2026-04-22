"""
config_handler.py — load/save JSON configs (local and remote).
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict

import requests


def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_config(config: Dict[str, Any], path: str = "config_out.json"):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=4, default=str)
    return config, path


def save_debug_info(debug_info: Dict[str, Any], path: str = "debug_out.json") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(debug_info, fh, indent=4, default=str)


def remote_load_config(url: str, username: str | None = None, password: str | None = None):
    try:
        auth = (username, password) if username and password else None
        response = requests.get(url, auth=auth, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        print(f"Failed to load remote configuration: {exc}", file=sys.stderr)
        return None


def remote_save_config(config: Dict[str, Any], url: str, username: str, password: str) -> bool:
    try:
        response = requests.post(
            url,
            auth=(username, password),
            data={"json_config": json.dumps(config, default=str)},
            timeout=30,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        print(f"Failed to save remote configuration: {exc}", file=sys.stderr)
        return False


def remote_log(config, debug_info, url, username, password) -> bool:
    try:
        response = requests.post(
            url,
            auth=(username, password),
            data={
                "json_config": json.dumps(config, default=str),
                "json_result": json.dumps(debug_info, default=str),
            },
            timeout=30,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        print(f"Failed to send remote log: {exc}", file=sys.stderr)
        return False
