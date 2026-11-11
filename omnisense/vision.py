"""Image task client for an explicitly configured independent HTTP engine."""
from __future__ import annotations
import io
import json
import time
import urllib.request
from pathlib import Path
from typing import Any
from PIL import Image
from .backends import algorithm_suffix, engine_url, engine_setting
from .registry import AlgorithmRegistry, LoadedAlgorithm, cached_default_registry
ImageInput = str | Path | bytes | bytearray | memoryview

def process_image(
    image: ImageInput,
    *,
    algorithm_id: str = "object-detection",
    task: str | None = None,
    registry: AlgorithmRegistry | None = None,
    model_variant: str | None = None,
    detector_variant: str = "face11n",
    weight: str | None = None,
    auto_download: bool = False,
    query: str | None = None,
    categories: list[str] | None = None,
    prompt: str | None = None,
    dimensions: int = 4096,
    normalize: bool = True,
    mime_type: str | None = None,
) -> dict[str, Any]:
    """Forward a task without installing engines, opening models or downloading weights."""
    started = time.perf_counter()
    registry = registry if registry is not None else cached_default_registry()
    if algorithm_id not in registry:
        return _unknown_algorithm_response(algorithm_id, registry, started)
    algorithm = registry.get(algorithm_id)
    if auto_download:
        return _response(algorithm, "invalid_request", [], started, warnings=["automatic downloads are disabled; configure an independent engine"] )
    if algorithm_id == "mock-detector" and not engine_setting("OMNISENSE_ENGINE_URL", algorithm_id):
        return _response(algorithm, "ok", [_mock_bbox(image)], started, engine="mock")
    return _run_external_engine(algorithm, image, started, task=task, model_variant=model_variant, model_id=weight, mime_type=mime_type, engine_options={"query": query, "categories": categories, "prompt": prompt, "dimensions": dimensions, "normalize": normalize})


def process_images(images: list[str | Path], **options: Any) -> dict[str, Any]:
    """Forward 1..64 image inputs as one independent-engine request."""
    if len(images) == 1:
        return process_image(images[0], **options)
    started = time.perf_counter()
    registry = options.pop("registry", None) or cached_default_registry()
    algorithm_id = options.pop("algorithm_id", "object-detection")
    if algorithm_id not in registry:
        return _unknown_algorithm_response(algorithm_id, registry, started)
    algorithm = registry.get(algorithm_id)
    if options.pop("auto_download", False):
        return _response(algorithm, "invalid_request", [], started, warnings=["automatic downloads are disabled; configure an independent engine"])
    task = options.pop("task", None)
    model_variant = options.pop("model_variant", None)
    model_id = options.pop("weight", None)
    options.pop("detector_variant", None)
    return _run_external_engine(algorithm, [Path(image) for image in images], started, task=task, model_variant=model_variant, model_id=model_id, mime_type=None, engine_options=options)

class _DetectionFacade:
    def run(
        self,
        image: str | Path,
        *,
        algorithm_id: str = "object-detection",
        task: str | None = None,
        registry: AlgorithmRegistry | None = None,
        model_variant: str | None = None,
        detector_variant: str = "face11n",
        weight: str | None = None,
        auto_download: bool = False,
        query: str | None = None,
        categories: list[str] | None = None,
        prompt: str | None = None,
        dimensions: int = 4096,
        normalize: bool = True,
    ) -> dict[str, Any]:
        return process_image(
            image,
            algorithm_id=algorithm_id,
            task=task,
            registry=registry,
            model_variant=model_variant,
            detector_variant=detector_variant,
            weight=weight,
            auto_download=auto_download,
            query=query,
            categories=categories,
            prompt=prompt,
            dimensions=dimensions,
            normalize=normalize,
        )


def _run_external_engine(
    algorithm: LoadedAlgorithm,
    image: ImageInput | list[ImageInput],
    started: float,
    *,
    task: str | None,
    model_variant: str | None,
    model_id: str | None,
    mime_type: str | None,
    engine_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import base64
    import mimetypes
    import urllib.request
    import urllib.error
    from urllib.parse import quote

    try:
        base = engine_url(algorithm.id)
        if base is None:
            return _response(algorithm, "missing_model", [], started,
                             missing_models=algorithm.missing_models,
                             engine="external-http",
                             warnings=["independent engine is not configured; set OMNISENSE_ENGINE_URL__" +
                                       algorithm_suffix(algorithm.id)])
        cap = 32 * 1024 * 1024
        images = image if isinstance(image, list) else [image]
        if not images or len(images) > 64:
            raise ValueError("engine image list must contain 1..64 items")
        total = 0
        items = []
        for item in images:
            item_mime = mime_type
            if isinstance(item, (bytes, bytearray, memoryview)):
                if len(item) > cap - total:
                    raise ValueError("engine input exceeds 32 MiB")
                raw = bytes(item)
            else:
                with Path(item).open("rb") as stream:
                    raw = stream.read(cap + 1)
                item_mime = item_mime or mimetypes.guess_type(str(item))[0]
            total += len(raw)
            if total > cap:
                raise ValueError("engine input exceeds 32 MiB")
            items.append({"mime_type": item_mime or "image/jpeg",
                          "data_base64": base64.b64encode(raw).decode("ascii")})
        payload = ({"kind": "image_bytes_list", "items": items} if isinstance(image, list)
                   else {"kind": "image_bytes", **items[0]})
        options: dict[str, Any] = {k: v for k, v in (engine_options or {}).items() if v is not None}
        if task:
            options["task"] = task
        configured_model = engine_setting("OMNISENSE_ENGINE_MODEL", algorithm.id)
        selected = configured_model or model_id or model_variant
        if selected:
            options["model_id"] = selected
        body = json.dumps({"algorithm_id": algorithm.id,
                           "input": payload, "options": options}, allow_nan=False).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        token = engine_setting("OMNISENSE_ENGINE_TOKEN", algorithm.id)
        if token:
            if any(c.isspace() for c in token):
                raise ValueError("engine token contains whitespace")
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(base + "/api/algorithms/" + quote(algorithm.id, safe="") + "/run",
                                         data=body, headers=headers, method="POST")
        timeout = max(1, min(3600, int(engine_setting("OMNISENSE_ENGINE_TIMEOUT_SECS", algorithm.id) or "120")))
        # No implicit proxy, redirects, package installation, pickle or model loading.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoEngineRedirect())
        deadline = time.monotonic() + timeout
        with opener.open(request, timeout=timeout) as response:
            if int(response.headers.get("Content-Length", "0")) > cap:
                raise ValueError("engine response exceeds 32 MiB")
            chunks = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("engine deadline exceeded")
                # HTTPResponse.read1 returns after one socket read, so a trickling
                # peer cannot hide the total deadline inside a large read().
                sock = getattr(getattr(response.fp, "raw", None), "_sock", None)
                if sock is not None:
                    sock.settimeout(remaining)
                chunk = response.read1(min(64 * 1024, cap + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
                if len(chunks) > cap:
                    raise ValueError("engine response exceeds 32 MiB")
            data = bytes(chunks)
        result = json.loads(data)
        if not isinstance(result, dict) or result.get("algorithm_id") != algorithm.id:
            raise ValueError("engine response algorithm_id mismatch")
        if result.get("status") not in {"ok", "missing_model", "runtime_error", "unsupported_input", "invalid_request", "timeout", "unknown_algorithm"}:
            raise ValueError("engine response has an unknown status")
        if not isinstance(result.get("outputs", []), list):
            raise ValueError("engine outputs must be a JSON array")
        result.setdefault("engine", "external-http")
        result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        result["privacy"] = algorithm.privacy()
        return result
    except Exception as exc:
        # Do not echo configured URLs, bearer tokens or upstream response bodies.
        warning = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        return _response(algorithm, "runtime_error", [], started, engine="external-http",
                         warnings=["independent engine unavailable: " + warning])


class _NoEngineRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise ValueError("engine redirects are disabled")


def _mock_bbox(image: ImageInput) -> dict[str, Any]:
    src: Any = io.BytesIO(bytes(image)) if isinstance(image, (bytes, bytearray, memoryview)) else image
    with Image.open(src) as opened:
        width, height = opened.size
    return {
        "kind": "bbox",
        "bbox": {
            "x1": width * 0.1,
            "y1": height * 0.1,
            "x2": width * 0.9,
            "y2": height * 0.9,
        },
        "label": "mock_object",
        "confidence": 0.99,
    }


def _unknown_algorithm_response(
    algorithm_id: str, registry: AlgorithmRegistry, started: float
) -> dict[str, Any]:
    """未注册算法的标准信封。

    没有 ``LoadedAlgorithm`` 可供 ``_response`` 取 descriptor/privacy,因此直接构造
    同一套信封字段(与 ``_response`` / Rust ``AlgorithmResponse`` 保持一致),并在
    warning 中列出当前实际可用的算法 id,让调用方自行纠正。
    """
    available = ", ".join(registry.available_ids()) or "(none)"
    return {
        "algorithm_id": algorithm_id,
        "status": "unknown_algorithm",
        "outputs": [],
        "missing_models": [],
        "warnings": [
            f"unknown algorithm_id '{algorithm_id}'; available algorithms: {available}"
        ],
        "privacy": [],
        "engine": None,
        "provider": None,
        "model_id": None,
        "fallback_chain": [],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def _response(
    algorithm: LoadedAlgorithm,
    status: str,
    outputs: list[dict[str, Any]],
    started: float,
    *,
    missing_models: list[str] | None = None,
    warnings: list[str] | None = None,
    engine: str | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    fallback_chain: list[str] | None = None,
) -> dict[str, Any]:
    """构造与 Rust ``AlgorithmResponse``(src/types.rs)一致的响应信封。

    Python 面、HTTP 面、Rust 进程内调用三者返回同一套 JSON 形状,包括 ``engine`` /
    ``provider`` / ``model_id`` / ``fallback_chain`` 这些数据血缘字段。
    """
    # 每条响应只消费 privacy —— descriptor() 会先 stat 每个模型文件
    # (model_availability),那是对信封毫无贡献的每请求 fs IO。
    return {
        "algorithm_id": algorithm.id,
        "status": status,
        "outputs": outputs,
        "missing_models": missing_models or [],
        "warnings": warnings or [],
        "privacy": algorithm.privacy(),
        "engine": engine,
        "provider": provider,
        "model_id": model_id,
        "fallback_chain": fallback_chain or [],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


detection = _DetectionFacade()
