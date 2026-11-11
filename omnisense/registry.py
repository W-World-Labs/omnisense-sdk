"""Algorithm metadata only; model and engine acquisition is disabled."""
from __future__ import annotations
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
BUNDLED_ALGORITHMS = PACKAGE_ROOT / "bundled_algorithms"
ALGORITHM_DIR_ENV = "OMNISENSE_ALGORITHM_DIR"
MODEL_DIR_ENV = "OMNISENSE_MODEL_DIR"
_CACHED_DEFAULTS_GUARD = threading.Lock()
_CACHED_DEFAULTS: dict[tuple[str, str], "AlgorithmRegistry"] = {}

@dataclass
class LoadedAlgorithm:
    manifest: dict[str, Any]
    manifest_path: Path
    model_base: Path
    available_models: list[str]
    missing_models: list[str]

    @property
    def id(self) -> str:
        return str(self.manifest["id"])

    def model_availability(self) -> tuple[list[str], list[str]]:
        available = []
        missing = []
        for model in self.manifest.get("models", []):
            if model["path"].startswith("builtin/") or resolve_model_path(
                self.model_base, model["path"]
            ).exists():
                available.append(model["id"])
            else:
                missing.append(model["id"])
        return available, missing

    def privacy(self) -> list[str]:
        """响应信封的 privacy 标记:按 manifest 声明顺序去重,零 fs IO。

        ``descriptor()`` 把同一列表折进完整 descriptor;只要 flag 集合的
        每请求路径(响应信封)必须用它 —— descriptor() 会先跑
        model_availability(),那是逐模型 ``Path.exists()`` 的 stat 扫描。
        """
        privacy: list[str] = []
        for capability in self.manifest.get("capabilities", []):
            for flag in capability.get("privacy", []):
                if flag not in privacy:
                    privacy.append(flag)
        return privacy

    def descriptor(self) -> dict[str, Any]:
        from .backends import external_supplier

        available_models, missing_models = self.model_availability()
        privacy = self.privacy()

        # status 必须保持 Rust RunStatus 集合;开箱即用/托管语义放在 availability。
        supplier = external_supplier(str(self.manifest["id"]))
        if self.id == "mock-detector":
            status = "ok"
            availability = "ready"
        elif supplier is not None:
            status = "missing_model"
            availability = "hosted"
        else:
            status = "missing_model"
            availability = "needs_model"

        return {
            "id": self.manifest["id"],
            "version": self.manifest["version"],
            "name_zh": self.manifest["name_zh"],
            "name_en": self.manifest["name_en"],
            "runtime_kind": self.manifest["runtime"]["kind"],
            "status": status,
            "availability": availability,
            "capabilities": [
                capability["id"] for capability in self.manifest.get("capabilities", [])
            ],
            "available_models": available_models,
            "missing_models": missing_models,
            "external_engine": supplier,
            "privacy": privacy,
        }


class AlgorithmRegistry:
    def __init__(self, algorithms: dict[str, LoadedAlgorithm]):
        self._algorithms = algorithms

    @classmethod
    def load_default(
        cls, *, auto_download: bool = False, model_base: str | Path | None = None
    ) -> "AlgorithmRegistry":
        return cls.load_from_dir(
            _default_algorithms_root(),
            auto_download=auto_download,
            model_base=model_base,
        )

    @classmethod
    def load_default_with_model_downloads(
        cls, *, model_base: str | Path | None = None
    ) -> "AlgorithmRegistry":
        return cls.load_default(auto_download=True, model_base=model_base)

    @classmethod
    def load_from_dir(
        cls,
        root: str | Path,
        *,
        auto_download: bool = False,
        model_base: str | Path | None = None,
    ) -> "AlgorithmRegistry":
        root = Path(root)
        base = Path(model_base) if model_base is not None else default_model_dir()
        if auto_download:
            raise ValueError("this SDK does not acquire models; configure an independent engine")

        algorithms = {}
        for manifest_path in find_manifest_paths(root):
            manifest = load_manifest(manifest_path)
            loaded = describe_manifest(manifest, manifest_path, base)
            algorithms[loaded.id] = loaded
        return cls(algorithms)

    def list_algorithms(self) -> list[dict[str, Any]]:
        return [algorithm.descriptor() for algorithm in self._algorithms.values()]

    def manifest(self, algorithm_id: str) -> dict[str, Any]:
        return dict(self._algorithms[algorithm_id].manifest)

    def get(self, algorithm_id: str) -> LoadedAlgorithm:
        return self._algorithms[algorithm_id]

    def __contains__(self, algorithm_id: object) -> bool:
        return algorithm_id in self._algorithms

    def available_ids(self) -> list[str]:
        """已注册的算法 id(源码树读 algorithms/,wheel 读打包期生成的 bundled 全量镜像)。"""
        return sorted(self._algorithms)


def find_manifest_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths = []
    for child in root.iterdir():
        if child.is_dir() and (child / "manifest.yaml").exists():
            paths.append(child / "manifest.yaml")
        elif child.suffix == ".yaml":
            paths.append(child)
    return sorted(paths)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    validate_manifest(manifest, path)
    return manifest


def validate_manifest(manifest: dict[str, Any], path: Path) -> None:
    required = ["schema_version", "id", "version", "runtime", "capabilities"]
    for key in required:
        if key not in manifest:
            raise ValueError(f"{path}: missing required key {key}")
    if manifest["schema_version"] != 1:
        raise ValueError(f"{path}: unsupported schema_version {manifest['schema_version']}")
    model_ids = {model["id"] for model in manifest.get("models", [])}
    capability_ids = {capability["id"] for capability in manifest.get("capabilities", [])}
    for capability in manifest.get("capabilities", []):
        for model_id in capability.get("model_ids", []):
            if model_id not in model_ids:
                raise ValueError(f"{path}: unknown model id {model_id}")
        for dependency in capability.get("depends_on", []):
            if dependency not in capability_ids:
                raise ValueError(f"{path}: unknown dependency {dependency}")


def describe_manifest(
    manifest: dict[str, Any], manifest_path: Path, model_base: Path
) -> LoadedAlgorithm:
    available = []
    missing = []
    for model in manifest.get("models", []):
        if model["path"].startswith("builtin/") or resolve_model_path(
            model_base, model["path"]
        ).exists():
            available.append(model["id"])
        else:
            missing.append(model["id"])
    return LoadedAlgorithm(
        manifest=manifest,
        manifest_path=manifest_path,
        model_base=model_base,
        available_models=available,
        missing_models=missing,
    )


def default_model_dir() -> Path:
    configured = os.environ.get(MODEL_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / ".omnisense/models"
    return Path.home() / ".cache/.omnisense/models"


def resolve_model_path(model_base: Path, relative_path: str) -> Path:
    direct = model_base / relative_path
    if direct.exists():
        return direct

    relative = Path(relative_path)
    if relative.parts and relative.parts[0] == "ax650n" and relative.name:
        flat = model_base / relative.name
        if flat.exists():
            return flat

    return direct


def _default_algorithms_root() -> Path:
    configured = os.environ.get(ALGORITHM_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    root = PROJECT_ROOT / "algorithms"
    return root if root.exists() else BUNDLED_ALGORITHMS


def cached_default_registry() -> "AlgorithmRegistry":
    """Cache manifest metadata within this process.

    ``load_default()`` 每次调用都重新扫描 manifest 目录、逐文件
    ``yaml.safe_load``、并对每个模型路径做 ``Path.exists()`` stat(~18 个
    manifest、~78 个模型路径);SDK 逐图调用 ``process_image`` 时这是纯重复
    工作。manifest 集在进程生命周期内视为不变,与 Rust 侧 boot 装载一次的
    语义一致。

    缓存键 = 解析后的 ``(algorithms root, model base)``,所以
    ``OMNISENSE_ALGORITHM_DIR`` / ``OMNISENSE_MODEL_DIR`` / ``XDG_CACHE_HOME``
    变化会得到独立条目,不会串错目录。

    刻意不覆盖 ``auto_download=True``:下载有副作用,重试语义要求每次都
    真实执行 —— 需要下载的调用方请直接调 ``load_default(auto_download=True)``。
    测试/热重载可用 ``cached_default_registry_clear()`` 复位。
    """
    root = _default_algorithms_root()
    key = (str(root), str(default_model_dir()))
    with _CACHED_DEFAULTS_GUARD:
        registry = _CACHED_DEFAULTS.get(key)
        if registry is None:
            registry = AlgorithmRegistry.load_from_dir(root)
            _CACHED_DEFAULTS[key] = registry
        return registry


def cached_default_registry_clear() -> None:
    """清空默认 registry 缓存(测试与 manifest 热更新场景用)。"""
    with _CACHED_DEFAULTS_GUARD:
        _CACHED_DEFAULTS.clear()

