#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tests.py  ── 迷你自動測試，不依賴 pytest
放在專案根或 tests/ 皆可。執行:
    python run_tests.py
程式退出碼 0 表全部通過，否則失敗。
"""
import sys, traceback
from pathlib import Path

# --- 專案模組匯入 ---
from ptx_analyzer.ptx_parser   import parse_ptx
from ptx_analyzer.cfg_builder  import build_cfg
from ptx_analyzer.analyzer     import analyse_ptx

# ---------- 工具 ----------
GREEN = "\033[92m"
RED   = "\033[91m"
END   = "\033[0m"

def ok(msg):  print(f"{GREEN}✓{END} {msg}")
def fail(msg):print(f"{RED}✗{END} {msg}")

def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)

# ---------- 測試樣本 ----------
ROOT = Path(__file__).resolve().parents[1]   # 專案根
PTX_SAMPLES = list((ROOT / "sample_ptx").glob("*.ptx"))
if not PTX_SAMPLES:
    print("找不到 sample_ptx/*.ptx，請先準備範例檔", file=sys.stderr)
    sys.exit(1)

# ---------- 個別測試 ----------
def test_parser(ptx_path: Path):
    src = ptx_path.read_text()
    insns, labels = parse_ptx(src)
    expect(len(insns) > 5,          "指令數過少")
    expect(any(op.opcode.startswith("ld.param") for op in insns),
           "開頭未見 ld.param")
    ok(f"Parser on {ptx_path.name}")

def test_cfg(ptx_path: Path):
    insns, labels = parse_ptx(ptx_path.read_text())
    blocks, cfg = build_cfg(insns, labels)
    expect(blocks, "Basic blocks 為空")
    first_lbl = blocks[0][0]
    expect(cfg[first_lbl], "首 block 無後繼")
    ok(f"CFG on {ptx_path.name}")

def test_end2end(ptx_path: Path):
    rpt = analyse_ptx(ptx_path.read_text())
    expect(rpt["param_alias"],      "param_alias 為空")
    expect(rpt["memory_access"],    "memory_access 為空")
    idx_exprs = [m["index_expr"] for m in rpt["memory_access"]]
    expect(all(idx_exprs),          "index_expr 有空字串")
    ok(f"End-to-End on {ptx_path.name}")

# ---------- 執行 ----------
TEST_FUNCS = [test_parser, test_cfg, test_end2end]
FAILED = 0

for ptx in PTX_SAMPLES:
    for func in TEST_FUNCS:
        try:
            func(ptx)
        except AssertionError as e:
            FAILED += 1
            fail(f"{func.__name__}({ptx.name}): {e}")
        except Exception as e:
            FAILED += 1
            fail(f"{func.__name__}({ptx.name}) raised {e.__class__.__name__}")
            traceback.print_exc()

if FAILED == 0:
    print(f"{GREEN}ALL TESTS PASSED{END}")
    sys.exit(0)
else:
    print(f"{RED}{FAILED} TEST(S) FAILED{END}")
    sys.exit(1)
