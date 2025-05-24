#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cfg_builder.py – 將指令切成 Basic Blocks 並建立 CFG
"""
from typing import Dict, List, Tuple
from ptx_analyzer.ptx_parser import Instruction
import logging
log = logging.getLogger(__name__)

BRANCH_OPS = {"bra", "call"}
RET_OPS    = {"ret"}

def build_cfg(insns: List[Instruction],
              labels: Dict[str, int]) -> Tuple[List[Tuple[str, List[int]]], Dict[str, List[str]]]:
    boundaries = {0}
    for i, ins in enumerate(insns):
        if ins.opcode in BRANCH_OPS | RET_OPS and i + 1 < len(insns):
            boundaries.add(i + 1)
        if ins.opcode in BRANCH_OPS and ins.operands:
            tgt = ins.operands[-1]
            if tgt in labels:
                boundaries.add(labels[tgt])
    boundaries = sorted(boundaries)

    blocks: List[Tuple[str, List[int]]] = []
    for beg, end in zip(boundaries, boundaries[1:] + [len(insns)]):
        blocks.append((f"BB_{beg}", list(range(beg, end))))

    bb_map: Dict[str, List[str]] = {lbl: [] for lbl, _ in blocks}
    idx2bb = {idx: lbl for lbl, idxs in blocks for idx in idxs}

    for lbl, idxs in blocks:
        last = insns[idxs[-1]]
        if last.opcode in BRANCH_OPS:
            tgt_lbl = idx2bb[labels[last.operands[-1]]]
            bb_map[lbl].append(tgt_lbl)
            if last.pred and idxs[-1] + 1 < len(insns):
                bb_map[lbl].append(idx2bb[idxs[-1] + 1])
        elif last.opcode not in RET_OPS and idxs[-1] + 1 < len(insns):
            bb_map[lbl].append(idx2bb[idxs[-1] + 1])

    log.debug("CFG: %s", bb_map)
    return blocks, bb_map
