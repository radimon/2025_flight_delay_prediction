import re
from difflib import SequenceMatcher

def norm_place(s: str) -> str:
    """把地點文字正規化，讓相似度比對更穩。"""
    if not s:
        return ""
    s = s.strip().lower()
    # 去掉括號內容（常見：站名後面一串行政區）
    s = re.sub(r"\(.*?\)", " ", s)
    # 把逗號換空白、壓縮多餘空白
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def best_label(it: dict) -> str:
    """從 geocoder 結果挑一個最像 '名稱' 的欄位拿來比對。"""
    if it.get("name"):
        return it["name"]
    nd = it.get("namedetails") or {}
    return nd.get("name:ja") or nd.get("name:en") or it.get("display_name") or it.get("title") or ""

def similarity(a: str, b: str) -> float:
    """0~1，相似度越高越像。"""
    a = norm_place(a)
    b = norm_place(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def importance(it: dict) -> float:
    try:
        return float(it.get("importance") or 0.0)
    except Exception:
        return 0.0


