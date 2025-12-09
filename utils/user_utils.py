"""集中管理各 API 端點共用的使用者名稱處理工具。"""
from __future__ import annotations
import os
from typing import Optional
from fastapi import HTTPException

_INVALID_USER_CHARS = {'\\', '/', ':', '*', '?', '"', '<', '>', '|'}
_RESERVED_USER_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def normalize_user_name(user_name: Optional[str]) -> Optional[str]:
    """將使用者名稱移除頭尾空白並轉為小寫。"""
    try:
        return user_name.strip().lower() if isinstance(user_name, str) else user_name
    except Exception:
        return user_name


def validate_user_name(user_name: str) -> None:
    """驗證 API 傳入的使用者名稱。
    Raises:
        HTTPException: 若缺少 user_name 或包含禁用字元則拋出例外。
    """
    if not isinstance(user_name, str) or not user_name.strip():
        raise HTTPException(status_code=400, detail="缺少或不合法的 user_name")

    if any(ch in user_name for ch in _INVALID_USER_CHARS):
        raise HTTPException(status_code=400, detail='user_name 不可包含路徑或特殊字元(\\/:*?"<>|)')

    lowered = user_name.strip().lower()
    if ".." in user_name:
        raise HTTPException(status_code=400, detail="不合法的 user_name")
    if lowered in _RESERVED_USER_NAMES:
        raise HTTPException(status_code=400, detail="不合法的 user_name")


def ensure_user_name(user_name: Optional[str]) -> str:
    """強制要求並回傳合法的使用者名稱（小寫）。"""
    if not isinstance(user_name, str) or not user_name.strip():
        raise HTTPException(status_code=400, detail="缺少或不合法的 user_name")
    validate_user_name(user_name)
    normalized = normalize_user_name(user_name)
    if not normalized:
        raise HTTPException(status_code=400, detail="缺少或不合法的 user_name")
    return normalized


def runs_root(user_name: Optional[str]) -> str:
    """回傳指定使用者命名空間下的 runs 根目錄。"""
    user = ensure_user_name(user_name)
    return os.path.join('runs', user)


def extract_user_from_dataset_name(dataset_name: str) -> Optional[str]:
    """從類似 "user/dataset" 的資料集名稱中推得使用者名稱。"""
    try:
        if not dataset_name:
            return None
        parts = dataset_name.replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0]:
            return parts[0].strip().lower()
        return None
    except Exception:
        return None


__all__ = [
    "normalize_user_name",
    "validate_user_name",
    "ensure_user_name",
    "runs_root",
    "extract_user_from_dataset_name",
]
