#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sym_eval.py – 將 register 運算轉為符號字串
"""
from typing import Dict
from ptx_analyzer.ptx_parser import Instruction
import logging
log = logging.getLogger(__name__)

def build_symtab(insns: list[Instruction]) -> Dict[str, str]:
    sym: Dict[str, str] = {
        "%ctaid.x":"blockIdx.x", "%ctaid.y":"blockIdx.y", "%ctaid.z":"blockIdx.z",
        "%tid.x":"threadIdx.x",  "%tid.y":"threadIdx.y",  "%tid.z":"threadIdx.z",
        "%ntid.x":"blockDim.x",  "%ntid.y":"blockDim.y",  "%ntid.z":"blockDim.z"
    }
    for ins in insns:
        op = ins.opcode
        if op.startswith("mov"):
            dst, src = ins.operands[:2]
            sym[dst] = sym.get(src, src)
        elif op.startswith(("add.","sub.","mul.lo.","mad.lo.")):
            dst = ins.operands[0]
            if op.startswith("add"):
                a,b = ins.operands[1:3]
                sym[dst] = f"({sym.get(a,a)} + {sym.get(b,b)})"
            elif op.startswith("sub"):
                a,b = ins.operands[1:3]
                sym[dst] = f"({sym.get(a,a)} - {sym.get(b,b)})"
            elif op.startswith("mul.lo"):
                a,b = ins.operands[1:3]
                sym[dst] = f"({sym.get(a,a)} * {sym.get(b,b)})"
            elif op.startswith("mad.lo"):
                a,b,c = ins.operands[1:4]
                sym[dst] = f"({sym.get(a,a)}*{sym.get(b,b)} + {sym.get(c,c)})"
        elif op.startswith("mul.wide.s32"):
            dst, r, imm = ins.operands[:3]
            sym[dst] = f"({sym.get(r,r)} * {imm})"
    log.debug("Symtab: %s", sym)
    return sym
