from pydantic import BaseModel
from typing import Any, Dict, List, Optional

# ───── 1. /run 請求 / 回傳 ─────────────────────────
class RunRequest(BaseModel):
    source_code: str
    user_arguments: str = ""
    filters: Optional[Dict[str, bool]] = None
    mode: Optional[str] = "ptx"  # 可為: "ptx", "parsed"

class RunResponse(BaseModel):
    ptx: str = ""
    parsed: Dict[str, Any] = {}
    stdout: str = ""
    stderr: str = ""
    error: str = ""

# ───── 2. /ptx/parse 請求 / 回傳 ─────────────────────
class PtxParseRequest(BaseModel):
    ptx: str

class PtxParseResponse(BaseModel):
    kernel: str
    tensors: List[Dict[str, Any]]
    parameter_alias: Dict[str, str]
