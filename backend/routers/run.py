from fastapi import APIRouter, HTTPException, status

from backend.models import RunRequest, RunResponse
from backend.exceptions import GodboltAPIError, RunError
from backend.services.godbolt_compiler import compile_cuda
from backend.services.ptx_mem_analyzer import analyze_ptx

router = APIRouter()

@router.post("", response_model=RunResponse)
async def run_compile(req: RunRequest) -> RunResponse:
    """
    編譯 kernel_code 與（可選）main_code：
    - ptx：永遠來自 kernel_code 編譯
    - stdout/stderr：依 filters.execute 決定是否來自含 main_code 的版本
    """
    try:
        # Step 1: 編譯 kernel_code（用來取得 ptx 與分析）
        error = ''
        kernel_result = await compile_cuda(
            source_code=req.kernel_code,
            user_arguments=req.user_arguments or "",
            filters=req.filters,
            compiler=req.compiler,
        )
        if not kernel_result["ret"]:
            error = kernel_result['stderr']
       
        # print(kernel_result)
        # Step 2: 如果需要執行，送出 kernel + main 編譯
        
        if req.filters.execute and req.main_code:
            full_source = req.kernel_code + "\n\n" + req.main_code
            full_result = await compile_cuda(
                source_code=full_source,
                user_arguments=req.user_arguments or "",
                filters=req.filters,
                compiler=req.compiler,
            )
            stdout = full_result["stdout"]
            stderr = full_result["stderr"]
            if not full_result["ret"]:
                error = stderr
        else:
            stdout = kernel_result["stdout"]
            stderr = kernel_result["stderr"]

        # Step 3: 若 mode 是 mem 且 kernel 編譯成功，才進行 PTX 分析
        
        if req.mode == "mem" and kernel_result["ret"]:
            parsed = analyze_ptx(kernel_result["ptx"])
            
           
            return RunResponse(
                error = error,
                ptx=kernel_result["ptx"],
                stdout=stdout,
                stderr=stderr,
                parsed=parsed.dict(),
            )

        return RunResponse(
            error= error,
            ptx=kernel_result["ptx"],
            stdout=stdout,
            stderr=stderr,
        )

    except (GodboltAPIError, RunError) as exc:
        # 回傳標準格式錯誤回應
        return RunResponse(
            ptx="",
            stdout="",
            stderr="",
            error=str(exc),
            parsed=None,
        )

