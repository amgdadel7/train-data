"""إدارة مسارات الصور داخل المشروع.

الهدف: ملف splits.csv يجب أن يبقى صالحًا حتى لو نُقل المشروع أو تغيّر الجهاز
(مثلًا من D:\\...\\train-data على ويندوز إلى /home/user/train-data على لينكس).
لذلك:
  * تُكتب المسارات في splits.csv بشكل **نسبي بالنسبة لجذر المشروع**.
  * عند القراءة (resolve_path) تُجرّب عدة مراجع حتى يُعثر على الملف فعليًا.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root(start: str | Path = ".") -> Path:
    """أعلى مجلد يحوي حزمة src/ — المرجع الأساسي للمسارات النسبية."""
    start = Path(start).resolve()
    if start.is_file():
        start = start.parent
    for cand in [start, *start.parents]:
        if (cand / "src").is_dir():
            return cand
    return start


def rel_to_root(path: str | Path, root: str | Path | None = None) -> str:
    """مسار نسبي عن جذر المشروع، بفواصل / ليعمل على كل الأنظمة."""
    root = Path(root) if root is not None else project_root(Path.cwd())
    return os.path.relpath(Path(path).resolve(), Path(root).resolve()).replace(os.sep, "/")


def resolve_path(raw: str, base: str | Path) -> Path:
    """يجد مسار صورة من سطر في splits.csv مهما كان الجهاز أو مكان المشروع.

    يجرّب بالترتيب: المسار كما هو (إن كان مطلقًا وموجودًا)، ثم بالنسبة لمجلد
    ملف splits.csv، ثم بالنسبة لجذر المشروع، ثم بالنسبة لمجلد العمل الحالي.
    """
    p = Path(raw)
    bases = [Path(base).resolve(), project_root(base), Path.cwd()]
    if p.is_absolute():
        if p.exists():
            return p
        # مسار مطلق من جهاز/نظام آخر: نجرّب نفس الذيل بالنسبة للمراجع المعروفة.
        rel = Path(*p.parts[1:]) if len(p.parts) > 1 else p
        candidates = [b / rel for b in bases]
    else:
        candidates = [b / p for b in bases]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]
