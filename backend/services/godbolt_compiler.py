#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
封裝呼叫 https://godbolt.org 的 nvcc126u2 編譯服務
--------------------------------------------------
回傳 dict(ret, ptx, stdout, stderr)
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx

from backend.exceptions import (
    GodboltAPIError,
    PtxExtractionError,
    RunError,
)

# --------------------------------------------------------------------
GODBOLT_URL = "https://godbolt.org/api/compiler/nvcc126u2/compile"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "x-requested-with": "XMLHttpRequest",
    "referer": "https://godbolt.org/",
}

# --------------------------------------------------------------------
def _extract_ptx(obj: Dict[str, Any]) -> str:
    """
    從 Godbolt JSON 擷取 PTX：
    - 新格式：obj["devices"]["PTX"]["asm"]
    - 舊格式：obj["asm"]
    """
    if "devices" in obj:
        devs = obj["devices"]
        if "PTX" in devs and "asm" in devs["PTX"]:
            return "\n".join(line["text"] for line in devs["PTX"]["asm"])
        for dev in devs.values():
            if "asm" in dev:
                return "\n".join(line["text"] for line in dev["asm"])

    if "asm" in obj:
        return "\n".join(line["text"] for line in obj["asm"])

    raise PtxExtractionError("No PTX asm found in Godbolt response")

# --------------------------------------------------------------------
async def compile_cuda(
    source_code: str,
    user_arguments: str = "",
    filters: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """呼叫 Godbolt nvcc，回傳包含 ret, ptx, stdout, stderr 的 dict"""

    payload = {
        "source": source_code,
        "compiler": "nvcc126u2",
        "options": {
            "userArguments": user_arguments or "-Xptxas -v",
            "filters": filters
            or {
                "binary": False,
                "binaryObject": False,
                "execute": True,
                "demangle": True,
                "directives": True,
                "intel": True,
                "labels": True,
                "commentOnly": True,
            },
        },
        "lang": "cuda",
        "bypassCache": 0,
        "allowStoreCodeDebug": True,
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(GODBOLT_URL, headers=HEADERS, json=payload)
            if resp.status_code >= 400:
                raise GodboltAPIError(f"HTTP {resp.status_code} from Godbolt")
            data = resp.json()
    except httpx.RequestError as e:  # pragma: no cover
        raise GodboltAPIError(f"Network error: {e}") from e
    except json.JSONDecodeError as e:  # pragma: no cover
        raise GodboltAPIError(f"Invalid JSON: {e}") from e

    # 擷取 stdout / stderr
    exec_result = data.get("execResult", {})
    stdout_txt = "\n".join(msg["text"] for msg in exec_result.get("stdout", []))
    stderr_txt = "\n".join(
        msg["text"] for msg in exec_result.get("buildResult", {}).get("stderr", [])
    )

    # 編譯錯誤檢查（中斷流程）
    errors = [
        msg["text"] for msg in data.get("stderr", []) if msg.get("severity") == "error"
    ]
    if errors:
        raise RunError("\n".join(errors))

    try:
        ptx = _extract_ptx(data)
        ret = ptx != "<Compilation failed>"
    except PtxExtractionError:
        ptx = ""
        ret = False

    return {
        "ret": ret,
        "ptx": ptx,
        "stdout": stdout_txt,
        "stderr": stderr_txt,
    }
