from typing import List, Dict, Literal, Optional
from pydantic import BaseModel


# ===== 供 /run 端點沿用 --- (原) ===========================================
class CompileFilters(BaseModel):
    binary: bool = False
    binaryObject: bool = False
    execute: bool = True
    demangle: bool = True
    directives: bool = True
    intel: bool = True
    labels: bool = True
    commentOnly: bool = True


class RunRequest(BaseModel):
    kernel_code: str
    main_code: Optional[str] = None
    user_arguments: str | None = None
    filters: CompileFilters = CompileFilters()
    mode: Literal["ptx", "mem"] = "ptx"
    compiler: Optional[str] = None



class RunResponse(BaseModel):
    ptx: str
    stdout: str
    stderr: str
    error: str  = ""
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
