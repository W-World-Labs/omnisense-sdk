"""omnisense 包的公共 API。

惰性导出(PEP 562 ``__getattr__``):``import omnisense`` 不再为
vision(PIL)/registry(yaml)/hardware 付 ~28ms 冷启动 —— CLI 的
``--help``/``status``/``serve`` 与子模块导入路径都不需要它们,只有真正
取用属性时才加载对应子模块。公开符号与兼容行为逐字不变:
``omnisense.process_image`` / ``from omnisense import vision`` /
``omnisense.AlgorithmRegistry`` 全部照常工作。
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AlgorithmRegistry",
    "default_model_dir",
    "detect_hardware",
    "detection",
    "process_image",
    "process_images",
    "select_device",
    "vision",
]

# 属性名 -> (子模块, 属性名);vision 是子模块本身。
_LAZY_ATTRS = {
    "vision": (".vision", None),
    "process_image": (".vision", "process_image"),
    "process_images": (".vision", "process_images"),
    "detection": (".vision", "detection"),
    "AlgorithmRegistry": (".registry", "AlgorithmRegistry"),
    "default_model_dir": (".registry", "default_model_dir"),
    "detect_hardware": (".hardware", "detect_hardware"),
    "select_device": (".hardware", "select_device"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(spec[0], __name__)
    return module if spec[1] is None else getattr(module, spec[1])


def __dir__() -> list[str]:
    return sorted(__all__)
