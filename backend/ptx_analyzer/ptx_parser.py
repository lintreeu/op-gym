#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ptx_parser.py – 解析 PTX → Instruction 列表與 label 對應
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from lark import Lark, Transformer, Token, Tree
import logging

log = logging.getLogger(__name__)

GRAMMAR = r"""
%import common.WS            -> WS
%import common.NEWLINE       -> NL
%ignore WS
%ignore NL
COMMENT      : "//" /[^\n]*/
REG          : /%[a-zA-Z_][a-zA-Z0-9_]*/
FIELD        : /\.(?:[xyz]|lo|hi)/
IMM_HEX      : /0[fFdD][0-9a-fA-F]{8}/
IMM          : /-?\d+(?:\.\d+)?(?![A-Za-z])/
MEM          : /\[[^\]]+\]/
LABEL        : /\$[a-zA-Z0-9_]+/
IDENT        : /[a-zA-Z_][a-zA-Z0-9_.]*/
RBRACE       : "}"
SEMICOLON    : ";"
COMMA        : ","
%ignore RBRACE
skip_entry   : ENTRY_LINE
ENTRY_LINE   : /.visible\s+\.entry(.|\n)*?\{/
?start       : skip_entry? block_body
block_body   : line+
line         : instruction SEMICOLON         -> instr
             | label_def                     -> label
             | COMMENT                       -> _comment
label_def    : LABEL ":"
instruction  : pred? opcode operand_list?
pred         : "@" REG
opcode       : IDENT
operand_list : operand (COMMA operand)*
operand      : REG FIELD?
             | IMM_HEX
             | IMM
             | MEM
             | LABEL
             | IDENT
"""

@dataclass
class Instruction:
    idx: int
    text: str
    opcode: str
    operands: List[str]
    pred: Optional[str] = None

class _AST2IR(Transformer):
    def __init__(self):
        super().__init__()
        self.insns: List[Instruction] = []
        self.labels: Dict[str, int] = {}

    def instr(self, items):
        node: Tree = items[0]
        pred = None
        if node.children and isinstance(node.children[0], Tree) and node.children[0].data == "pred":
            pred = node.children[0].children[0].value
            node.children = node.children[1:]

        opcode = node.children[0].children[0].value

        operands: List[str] = []
        for op_tree in node.find_data("operand"):
            operands.append("".join(tok.value for tok in op_tree.children))

        self.insns.append(
            Instruction(
                idx=len(self.insns),
                text=node.meta.orig_str.strip(),
                opcode=opcode,
                operands=operands,
                pred=pred,
            )
        )
        log.debug("Add instr %s %s", opcode, operands)
        return None

    def label(self, items):
        tok: Token = items[0]
        self.labels[tok.value] = len(self.insns)
        log.debug("Label %s -> %d", tok.value, len(self.insns))
        return None

def parse_ptx(src: str):
    parser = Lark(GRAMMAR, parser="lalr", lexer="basic", propagate_positions=True)
    tree = parser.parse(src)
    conv = _AST2IR()
    conv.transform(tree)
    return conv.insns, conv.labels
