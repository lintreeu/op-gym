from typing import List, Dict, Literal
from pydantic import BaseModel


# ===== 供 /run 端點沿用 --- (原) ===========================================
class CompileFilters(BaseModel):
    demangle: bool = True
    ir: bool = False
    opt: bool = False


class RunRequest(BaseModel):
    source_code: str
    user_arguments: str | None = None
    filters: CompileFilters = CompileFilters()
    mode: Literal["ptx", "parsed", "mem"] = "ptx"


class RunResponse(BaseModel):
    ret: bool
    ptx: str
    stdout: str
    stderr: str
    error: str | None = None
    parsed: dict | None = None


# ======== 新增：PTX 記憶體走訪分析 ========================================
class MemAccess(BaseModel):
    kind: Literal["load", "store"]
    param: List[str]
    base: str
    offset: str
    eltype: str
    raw: str


class PtxMemAnalyzeRequest(BaseModel):
    ptx: str


class PtxMemAnalyzeResponse(BaseModel):
    kernel: str
    accesses: List[MemAccess]
    element_size_table: Dict[str, int]
