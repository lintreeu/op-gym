#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyzer.py – 整合所有分析
"""
import json, sys, logging, textwrap
from pathlib import Path
from ptx_analyzer.ptx_parser   import parse_ptx
from ptx_analyzer.cfg_builder  import build_cfg
from ptx_analyzer.mem_analysis import (collect_param_alias,
                                       collect_base_ptr,
                                       analyse_ldst)
log = logging.getLogger(__name__)

def analyse_ptx(ptx_src: str):
    insns, labels = parse_ptx(ptx_src)
    blocks, cfg   = build_cfg(insns, labels)
    param_alias   = collect_param_alias(insns)
    base_ptr      = collect_base_ptr(insns)
    mem_access    = analyse_ldst(insns, param_alias, base_ptr)
    return {
        "param_alias"   : param_alias,
        "global_base_ptr": base_ptr,
        "cfg"           : cfg,
        "basic_blocks"  : {lbl: idxs for lbl, idxs in blocks},
        "memory_access" : mem_access
    }

def _self_check():
    sample = textwrap.dedent("""
    .visible .entry void_kernel(float* A)(
        .param .u64 void_kernel_param_0
    )
    {
        ld.param.u64 %rd0, [void_kernel_param_0];
        cvta.to.global.u64 %rd1, %rd0;
        st.global.f32 [%rd1+0], 0f3f800000;
        ret;
    }
    """)
    rpt = analyse_ptx(sample)
    assert rpt["memory_access"][0]["index_expr"] == "0"
    print("Self-check OK", file=sys.stderr)

if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _self_check()
        sys.exit(0)
    if len(sys.argv) != 2:
        print("Usage: python -m ptx_analyzer.analyzer <file.ptx>", file=sys.stderr)
        sys.exit(1)
    rpt = analyse_ptx(Path(sys.argv[1]).read_text())
    print(json.dumps(rpt, indent=2, ensure_ascii=False))
