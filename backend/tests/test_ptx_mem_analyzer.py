from pathlib import Path
from backend.services.ptx_mem_analyzer import analyze_ptx


def test_simple_softmax_kernel():
    sample_ptx = (Path(__file__).parent / "sample_softmax.ptx").read_text(
        "utf-8"
    )
    result = analyze_ptx(sample_ptx)

    # smoke assertions
    assert result.kernel == "softmax_kernel"
    assert len(result.accesses) > 0
    kinds = {acc.kind for acc in result.accesses}
    assert kinds.issubset({"load", "store"})
