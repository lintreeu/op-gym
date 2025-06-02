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
    compiler: str = "nvcc126u2",
) -> Dict[str, Any]:
    """呼叫 Godbolt nvcc，回傳包含 ret, ptx, stdout, stderr 的 dict"""

    payload = {
        "source": source_code,
        "compiler": compiler,
        "options": {
            "userArguments": user_arguments or "-Xptxas -v",
            "filters": filters.model_dump() if filters else {
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

    stdout_txt = ""
    # 優先從 execResult.stdout 抓取 (list of {text: ...})
    if "stdout" in exec_result and isinstance(exec_result.get("stdout"), list):
        stdout_txt = "\n".join(
            msg.get("text", "") for msg in exec_result["stdout"]
        )
    # 若 execResult 不存在 stdout，則嘗試從 data["stdout"]
    elif isinstance(data.get("stdout"), list):
        stdout_txt = "\n".join(
            msg.get("text", "") for msg in data["stdout"]
        )
    # fallback：若 data["stdout"] 是純字串（非 list）
    else:
        stdout_txt = data.get("stdout", "")


    # 預設為空字串
    stderr_txt = ""
    # 優先從 execResult.buildResult.stderr 取得
    if "buildResult" in exec_result and isinstance(exec_result["buildResult"].get("stderr"), list):
        stderr_txt = "\n".join(
            msg.get("text", "") for msg in exec_result["buildResult"]["stderr"]
        )
    # 若沒有 buildResult 或其 stderr，改從 data["stderr"] 嘗試
    elif isinstance(data.get("stderr"), list):
        stderr_txt = "\n".join(
            msg.get("text", "") for msg in data["stderr"]
        )
    # 最後 fallback：若 data["stderr"] 是純字串
    else:
        stderr_txt = data.get("stderr", "")


    print(stderr_txt)

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
