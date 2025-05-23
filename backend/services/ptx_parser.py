import re
from typing import Any, Dict, List, Optional

# ----------- 私有小工具 -------------------------------------------------
def _parse_params(ptx: str) -> List[Dict[str, Any]]:
    params, idx = [], 0
    for line in re.findall(r"\.param\s+\.[^\n]+", ptx):
        m = re.match(r"\.param\s+\.(\w+)\s+([^\s,]+)", line.strip())
        if m:
            dtype, name = m.groups()
            params.append({"index": idx, "name": name, "ptype": dtype})
            idx += 1
    return params

def _detect_element_size(ptx: str) -> Optional[int]:
    m = re.search(r"shl\.b64\s+%\w+,\s+%\w+,\s+(\d+);", ptx)
    return 1 << int(m.group(1)) if m else None

def _build_tensor_traversal(params: List[Dict[str, Any]], elem_size: int):
    tpl = {
        "index_per_thread": "blockIdx.x * num_features + threadIdx.x",
        "block_traversal": {
            "start_index": "blockIdx.x * num_features",
            "end_index": "blockIdx.x * num_features + num_features - 1",
            "stride": 1,
        },
    }
    tensors = []
    for p in params:
        if p["ptype"].startswith("u64"):
            role = "input" if p["index"] == 0 else "output"
            tensors.append(
                {
                    "index": p["index"],
                    "name": p["name"],
                    "role": role,
                    "element_size_bytes": elem_size,
                    **tpl,
                }
            )
    return tensors

# ----------- 對外 API ---------------------------------------------------
def parse_ptx(ptx: str) -> Dict[str, Any]:
    elem_size = _detect_element_size(ptx) or 4
    params = _parse_params(ptx)
    return {
        "kernel": re.search(r"\.entry\s+([^(]+)", ptx).group(1),
        "tensors": _build_tensor_traversal(params, elem_size),
        "parameter_alias": {"num_rows": params[-2]["name"], "num_features": params[-1]["name"]}  # 推測
    }
