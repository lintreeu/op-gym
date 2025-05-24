"""PTX 靜態分析器 – 專案初始化（含 logging）"""
import logging, os, sys

_lvl = logging.DEBUG if os.getenv("PTX_ANALYZER_DEBUG") else logging.INFO
logging.basicConfig(
    level=_lvl,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
