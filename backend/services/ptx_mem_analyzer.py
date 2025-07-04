#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure-logic version of ptx_mem_analyzer.

Convert PTX string -> structured memory-access report
so that FastAPI can call it without touching the file-system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

from lark import Lark, Tree, Token

from backend.models import (
    PtxMemAnalyzeResponse,
    MemAccess,
)

# --------------------------------------------------------------------------
# -- Grammar & constants (完全沿用舊版) -------------------------------------
GRAMMAR = r"""
%import common.WS
%import common.NEWLINE
%ignore WS
%ignore NEWLINE
COMMENT: "//" /[^\n]*/

REG: /%[a-zA-Z_][a-zA-Z0-9_]*/
COMPOSITE_REG: REG "|" REG
FIELD: /\.(?:[xyz]|lo|hi)/
IMM_HEX: /0[fFdD][0-9a-fA-F]{8}/
IMM: /-?\d+(?:\.\d+)?(?![A-Za-z])/
MEM: /\[[^\]]+\]/
LABEL: /\$[a-zA-Z0-9_]+/
IDENT: /[a-zA-Z_][a-zA-Z0-9_.]*/
RBRACE: "}"
SEMICOLON: ";"
COMMA: ","

?start: block_body
block_body: line+

line: instruction SEMICOLON         -> instr_line
    | label_def                     -> label_line
    | COMMENT                       -> comment_line

label_def: LABEL ":"                -> label_def
instruction: pred? opcode operand_list? -> instruction
pred: "@" REG
opcode: IDENT
operand_list: operand (COMMA operand)*
operand: COMPOSITE_REG
       | REG FIELD?
       | reg_group
       | IMM_HEX
       | IMM
       | MEM
       | LABEL
       | IDENT

reg_group: "{" REG (COMMA REG)* "}"
"""

ENTRY_RE = re.compile(
    r"\.visible\s+\.entry\s+([a-zA-Z0-9_]+)\s*\((.*?)\)\s*", re.S
)
PARAM_RE = re.compile(r"\.param\s+\.\w+\s+([^\n]+?_param_\d+)")

ELEMENT_SIZE: Dict[str, int] = {
    "f32": 4,
    "f64": 8,
    "u32": 4,
    "u64": 8,
    "s32": 4,
    "s64": 8,
}


# --------------------------------------------------------------------------
# -- Small dataclasses ------------------------------------------------------
@dataclass
class KernelSignature:
    name: str
    proto_types: List[str]
    slot_names: List[str]
    slot_to_human: Dict[str, str]


@dataclass
class Instruction:
    idx: int
    pred: str | None
    opcode: str
    operands: List[str]
    raw: str


@dataclass
class LabelNode:
    name: str


@dataclass
class BasicBlock:
    label: str | None
    instrs: List[Instruction]


def _parse_header_and_strip(ptx: str) -> tuple[str, KernelSignature]:
    m = ENTRY_RE.search(ptx)
    if not m:
        raise RuntimeError("No .entry block found")
    kernel, proto_raw = m.groups()
    proto_types = [p.strip() for p in proto_raw.split(",") if p.strip()]
    slot_names = PARAM_RE.findall(ptx)

    default_names = [f"arg{i}" for i in range(len(proto_types))]
    slot_to_human = {
        s: (default_names[int(re.search(r"_param_(\d+)$", s).group(1))] if re.search(r"_param_(\d+)$", s) else s)
        for s in slot_names
    }

    # 🔧 改為更保守的範圍提取：只擷取 .entry block 的最內層 {}
    entry_start = ptx.find("{", m.end())
    entry_end = ptx.find("}", entry_start)
    if entry_start == -1 or entry_end == -1:
        raise RuntimeError("Malformed PTX: cannot find .entry block body")
    body = ptx[entry_start + 1:entry_end].strip()

    return body, KernelSignature(kernel, proto_types, slot_names, slot_to_human)

def _build_parser() -> Lark:
    return Lark(
        GRAMMAR,
        parser="lalr",
        lexer="basic",
        start="start",
        propagate_positions=True,
    )


def _ast_to_instr(tree: Tree, lines: List[str]) -> List[Instruction | LabelNode]:
    out, idx = [], 0
    for ch in tree.children:
        if ch.data == "label_line":
            out.append(LabelNode(ch.children[0].children[0].value))
            continue
        node = ch.children[0]
        kids = node.children
        pred: str | None = None
        opcode = ""
        ops_node = None
        if len(kids) == 3:
            pred = kids[0].children[0].value
            opcode = kids[1].children[0].value
            ops_node = kids[2]
        elif len(kids) == 2:
            if isinstance(kids[0], Tree) and kids[0].data == "pred":
                pred = kids[0].children[0].value
                opcode = kids[1].children[0].value
            else:
                opcode = kids[0].children[0].value
                ops_node = kids[1]
        else:
            opcode = kids[0].children[0].value

        ops: List[str] = []
        if isinstance(ops_node, Tree):
            for t in ops_node.children:
                if isinstance(t, Token) and t.type == "COMMA":
                    continue
                if isinstance(t, Tree):
                    if t.data == "reg_group":
                        # 展開 register group
                        regs = [child.value for child in t.children if isinstance(child, Token) and child.type == "REG"]
                        ops.append(regs)  # 以 list 形式保留
                    else:
                        token_str = ''.join(
                            child.value if isinstance(child, Token) else str(child)
                            for child in t.children
                        )
                        ops.append(token_str)
                elif isinstance(t, Token):
                    ops.append(t.value)
                else:
                    ops.append(str(t))
        out.append(
            Instruction(
                idx=idx,
                pred=pred,
                opcode=opcode,
                operands=ops,
                raw=lines[node.meta.line - 1].rstrip(),
            )
        )
        idx += 1
    return out


def _split_basic_blocks(
    ins: List[Instruction | LabelNode],
) -> List[BasicBlock]:
    blocks: List[BasicBlock] = []
    cur: List[Instruction] = []
    label: str | None = None

    def flush():
        nonlocal cur, label
        if cur:
            blocks.append(BasicBlock(label, cur))
        cur = []
        label = None

    for n in ins:
        if isinstance(n, LabelNode):
            flush()
            label = n.name
            continue
        cur.append(n)
        if n.opcode in {"bra", "ret"} or (
            n.pred and "bra" in n.opcode
        ):
            flush()
    flush()
    return blocks


def _build_cfg(bbs: List[BasicBlock]):
    idx = {b.label: i for i, b in enumerate(bbs) if b.label}
    cfg = defaultdict(list)
    for i, b in enumerate(bbs):
        if not b.instrs:
            continue
        last = b.instrs[-1]
        if last.opcode == "bra":
            tgt = last.operands[0]
            cfg[i].append(idx.get(tgt, -1))
        elif last.opcode == "ret":
            pass
        elif last.pred and "bra" in last.opcode:
            tgt = last.operands[0]
            cfg[i].append(idx.get(tgt, -1))
            if i + 1 < len(bbs):
                cfg[i].append(i + 1)
        else:
            if i + 1 < len(bbs):
                cfg[i].append(i + 1)
    return cfg


SPECIAL_MAP = {
    "%tid.x": "threadIdx.x",
    "%tid.y": "threadIdx.y",
    "%tid.z": "threadIdx.z",
    "%ntid.x": "blockDim.x",
    "%ntid.y": "blockDim.y",
    "%ntid.z": "blockDim.z",
    "%ctaid.x": "blockIdx.x",
    "%ctaid.y": "blockIdx.y",
    "%ctaid.z": "blockIdx.z",
}


class _ExpressionEngine:
    def __init__(self, alias):
        self.env = dict(alias)

    def _v(self, t):
        return (
            self.env.get(t, SPECIAL_MAP.get(t, t))
            if not t.lstrip("-").isdigit()
            else t
        )

    def _mov(self, d, s):
        self.env[d] = self._v(s)

    def _bin(self, d, a, b, op):
        self.env[d] = f"({self._v(a)}{op}{self._v(b)})"

    def _mad(self, d, a, b, c):
        self.env[d] = f"(({self._v(a)}*{self._v(b)})+{self._v(c)})"

    def handle(self, ins: Instruction):
        o, p = ins.opcode, ins.operands
        try:
            if o.startswith("mov"):
                self._mov(p[0], p[1])
            elif o.startswith("cvta"):
                self._mov(p[0], p[1])
            elif o.startswith("add"):
                self._bin(p[0], p[1], p[2], "+")
            elif o.startswith("sub"):
                self._bin(p[0], p[1], p[2], "-")
            elif o.startswith(("mul.lo", "mul.wide")):
                self._bin(p[0], p[1], p[2], "*")
            elif o.startswith("mad.lo"):
                self._mad(p[0], p[1], p[2], p[3])
            elif o.startswith("ld.param"):
                dst = p[0]
                slot = re.search(r"\[(.+)\]", p[1]).group(1)
                self.env[dst] = self._v(slot)
        except IndexError:
            pass


def _extract_type(opcode: str) -> str:
    m = re.search(r"\.(f32|f64|u32|u64|s32|s64)", opcode)
    return m.group(1) if m else "unknown"


def _analyze_mem(blocks, alias):
    eng = _ExpressionEngine(alias)
    result = []

    for bb in blocks:
        for ins in bb.instrs:
            eng.handle(ins)

            if not ins.opcode.startswith(("ld.global", "st.global")):
                continue

            kind = "load" if ins.opcode.startswith("ld.") else "store"
            eltype = _extract_type(ins.opcode)
            elsize = ELEMENT_SIZE.get(eltype, 4)

            # 向量 register group
            is_vector = isinstance(ins.operands[0], list)
            reg_group = ins.operands[0] if is_vector else None
            addr_operand_idx = 1 if is_vector else (1 if kind == "load" else 0)

            if addr_operand_idx >= len(ins.operands):
                continue
            addr = ins.operands[addr_operand_idx]
            if isinstance(addr, list):  # 意外情況
                continue
            m = re.search(r"\[(.+)\]", addr)
            if not m:
                continue
            base_expr = m.group(1).strip()
            base_resolved = eng._v(base_expr)

            param_used = sorted(name for slot, name in alias.items() if name in base_resolved)
            base_param = next((name for name in param_used if name in base_resolved), None)

            def compute_offset(index: int) -> str:
                expr = f"({base_resolved}+{index * elsize})" if index > 0 else base_resolved
                if base_param and expr != base_param:
                    offset = expr.replace(base_param, "").lstrip("+").strip()
                else:
                    offset = "0"
                return re.sub(rf"\b{elsize}\b", f"element_size_{eltype}", offset)

            if is_vector:
                for i, reg in enumerate(reg_group):
                    result.append(dict(
                        kind=kind,
                        param=param_used,
                        base=base_param or base_resolved,
                        offset=compute_offset(i),
                        eltype=eltype,
                        raw=ins.raw + f"  // vector[{i}]"
                    ))
            else:
                result.append(dict(
                    kind=kind,
                    param=param_used,
                    base=base_param or base_resolved,
                    offset=compute_offset(0),
                    eltype=eltype,
                    raw=ins.raw
                ))
    print(result)
    return result


# --------------------------------------------------------------------------
# -- public API -------------------------------------------------------------
def analyze_ptx(ptx_src: str) -> PtxMemAnalyzeResponse:
    """
    Core entry – used by FastAPI router & unit-test
    """
    body, sig = _parse_header_and_strip(ptx_src)
    parser = _build_parser()
    tree = parser.parse(body)

    lines = body.splitlines()
    instrs = _ast_to_instr(tree, lines)
    bbs = _split_basic_blocks(instrs)

    accesses_raw = _analyze_mem(bbs, sig.slot_to_human)
    accesses = [MemAccess(**acc) for acc in accesses_raw]

    return PtxMemAnalyzeResponse(
        kernel=sig.name,
        accesses=accesses,
        element_size_table=ELEMENT_SIZE,
    )
