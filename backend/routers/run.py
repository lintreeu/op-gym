from fastapi import APIRouter, HTTPException, status

from backend.models import RunRequest, RunResponse
from backend.exceptions import GodboltAPIError, RunError
from backend.services.godbolt_compiler import compile_cuda
from backend.services.ptx_mem_analyzer import analyze_ptx

router = APIRouter()


@router.post("", response_model=RunResponse)
async def run_compile(req: RunRequest) -> RunResponse:
    """
    將使用者 CUDA 程式碼送往 Godbolt 編譯；
    req.mode:
      • "ptx"  → 只回 PTX
      • "mem"  → 同時回 memory-access 分析
      • 其他值仍保留未來擴充
    """
    try:
        data = await compile_cuda(req.source_code, req.user_arguments, req.filters)
    except (GodboltAPIError, RunError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if not data["ret"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=data["stderr"]
        )

    # ---- "ptx" ---------------------------------------------------------
    if req.mode == "ptx":
        return RunResponse(
            ret=True,
            ptx=data["ptx"],
            stdout=data["stdout"],
            stderr=data["stderr"],
        )

    # ---- "mem" (= 先編譯再分析) ---------------------------------------
    if req.mode == "mem":
        parsed = analyze_ptx(data["ptx"])
        return RunResponse(
            ret=True,
            ptx=data["ptx"],
            stdout=data["stdout"],
            stderr=data["stderr"],
            parsed=parsed.dict(),
        )

    # ---- fallback ------------------------------------------------------
    return RunResponse(
        ret=True,
        ptx=data["ptx"],
        stdout=data["stdout"],
        stderr=data["stderr"],
    )
