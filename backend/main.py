#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import run, ptx_mem

app = FastAPI(
    title="Op-Gym",
    version="0.1.0",
    description="Compile / Analyse / Visualise Neural Network Operator kernels in the browser",
)

# -- CORS ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Routers ---------------------------------------------------------------
app.include_router(run.router, prefix="/run", tags=["run"])
app.include_router(ptx_mem.router, prefix="/ptx", tags=["ptx-mem"])

# ----------------------------- health-check -------------------------------
@app.get("/ping")
async def ping() -> dict[str, str]:
    return {"msg": "pong"}
