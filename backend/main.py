from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    PtxParseRequest,
    PtxParseResponse,
    RunRequest,
    RunResponse,
)
from services.godbolt import compile_cuda
from services.ptx_parser import parse_ptx
from services.exceptions import (
    GodboltAPIError,
    PtxExtractionError,
    RunError,
)

import traceback

# ──── FastAPI init ─────────────────────────────────
app = FastAPI(title="CUDA-PTX Backend", docs_url="/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ──── API: /ptx/parse ─────────────────────────────
@app.post("/ptx/parse", response_model=PtxParseResponse)
async def parse_ptx_endpoint(req: PtxParseRequest):
    try:
        return parse_ptx(req.ptx)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ──── API: /run（舊前端相容）───────────────────────
from fastapi import HTTPException
import traceback

@app.post("/run", response_model=RunResponse)
async def run_compile(req: RunRequest):
    try:
        data = await compile_cuda(req.source_code, req.user_arguments, req.filters)

        ptx = data.get("ptx", "")
        stdout = data.get("stdout", "")
        stderr = data.get("stderr", "")
        ret = data.get("ret", False)

        # ── mode: ptx ──
        if req.mode == "ptx":
            return RunResponse(
                ptx=ptx,
                parsed={},
                stdout=stdout,
                stderr=stderr,
                error=""
            )

        # ── mode: parsed ──
        elif req.mode == "parsed":
            if not ret:
                return RunResponse(
                    ptx=ptx,
                    parsed={},
                    stdout=stdout,
                    stderr=stderr,
                    error="PTX extraction failed, cannot parse"
                )
            try:
                parsed = parse_ptx(ptx)
                return RunResponse(
                    ptx=ptx,
                    parsed=parsed,
                    stdout=stdout,
                    stderr=stderr,
                    error=""
                )
            except Exception as e:
                traceback.print_exc()
                return RunResponse(
                    ptx=ptx,
                    parsed={},
                    stdout=stdout,
                    stderr=stderr,
                    error=f"parse_ptx failed: {str(e)}"
                )

        # ── 不支援的 mode ──
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported mode: {req.mode}")

    # ── 編譯或 Godbolt 層錯誤 ──
    except RunError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except PtxExtractionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except GodboltAPIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        return RunResponse(
            ptx="",
            parsed={},
            stdout="",
            stderr="",
            error=f"Internal server error: {str(e)}"
        )