from fastapi import APIRouter, HTTPException, status

from backend.models import (
    PtxMemAnalyzeRequest,
    PtxMemAnalyzeResponse,
)
from backend.services.ptx_mem_analyzer import analyze_ptx

router = APIRouter()


@router.post("/mem", response_model=PtxMemAnalyzeResponse)
async def ptx_mem(req: PtxMemAnalyzeRequest) -> PtxMemAnalyzeResponse:
    try:
        return analyze_ptx(req.ptx)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PTX parse failed: {exc}",
        ) from exc
