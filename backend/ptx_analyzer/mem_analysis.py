#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mem_analysis.py – 參數別名、base pointer、記憶體訪問符號式
"""
import re, logging
from typing import Dict, List
from ptx_analyzer.ptx_parser import Instruction
from ptx_analyzer.sym_eval  import build_symtab
log = logging.getLogger(__name__)

_param_pat = re.compile(r"\[(.+?)\]")

def collect_param_alias(insns: List[Instruction]) -> Dict[str, str]:
    mapping = {}
    for ins in insns:
        if ins.opcode.startswith("ld.param"):
            dst = ins.operands[0]
            m = _param_pat.search(ins.operands[1])
            if m:
                mapping[dst] = m.group(1)
    log.debug("Param alias: %s", mapping)
    return mapping

def collect_base_ptr(insns: List[Instruction]) -> Dict[str, str]:
    base_map = {}
    for ins in insns:
        if ins.opcode.startswith("cvta.to.global"):
            dst, src = ins.operands[:2]
            base_map[dst] = src
    log.debug("Base ptr map: %s", base_map)
    return base_map

def analyse_ldst(insns: List[Instruction],
                 param_alias: Dict[str,str],
                 base_ptr: Dict[str,str]) -> List[dict]:
    symtab = build_symtab(insns)
    results: List[dict] = []
    for ins in insns:
        if not ins.opcode.startswith(("ld.global","st.global")):
            continue
        mem = ins.operands[1]                     # "[%rdX+16]"
        base_reg = mem.split('+')[0][1:]          # 去掉 '['
        param = "<unknown>"
        if base_reg in base_ptr and base_ptr[base_reg] in param_alias:
            param = param_alias[base_ptr[base_reg]]
        elif base_reg in param_alias:
            param = param_alias[base_reg]

        idx_expr = "0"
        if '+' in mem:
            tail = mem.split('+')[1].rstrip(']')
            if tail.isdigit():
                idx_expr = str(int(tail)//4)      # 假設 float32
            else:
                idx_expr = f"({symtab.get(tail, tail)})"

        results.append({
            "instr": ins.text,
            "param": param,
            "index_expr": idx_expr,
            "access": "load" if ins.opcode.startswith("ld.") else "store"
        })
    log.debug("Mem access: %s", results)
    return results
