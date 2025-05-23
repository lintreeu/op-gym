# ★ 新增檔案，集中所有自定錯誤

class GodboltAPIError(Exception):
    """Godbolt 服務呼叫失敗（網路或非 2xx 狀態碼）"""

class PtxExtractionError(Exception):
    """成功取得 JSON，但在內文找不到 PTX"""

class RunError(Exception):
    """Godbolt 回傳編譯錯誤訊息"""
