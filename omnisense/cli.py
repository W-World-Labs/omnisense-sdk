"""OmniSense SDK command line client."""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from typing import Any, Sequence

DEFAULT_ALGORITHM = "object-detection"

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="omnisense", description="OmniSense HTTP task SDK", epilog="omnisense detect image.jpg; omnisense status")
    sub = parser.add_subparsers(dest="command")
    detect = sub.add_parser("detect", help="send an image to your configured engine")
    detect.add_argument("image")
    detect.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    detect.add_argument("--task", default=None)
    detect.add_argument("--weight", default=None, help="registered engine model identifier")
    detect.add_argument("--model-variant", default=None)
    detect.add_argument("--auto-download", action="store_true", help="deprecated; always rejected")
    sub.add_parser("status", help="list SDK task configuration, without probing engine health")
    args = parser.parse_args(argv)
    if args.command == "detect":
        return _cmd_detect(args)
    if args.command == "status":
        return _cmd_status(args)
    parser.print_help()
    return 0

def _cmd_detect(args: argparse.Namespace) -> int:
    from pathlib import Path

    if not Path(args.image).exists():
        print(f"❌ 找不到图片: {args.image}", file=sys.stderr)
        return 2

    # No package or model acquisition is performed during inference.
    print("🔧 连接用户已配置的引擎；不下载或安装模型…", file=sys.stderr)

    from .vision import process_image

    result = process_image(
        args.image,
        algorithm_id=args.algorithm,
        task=args.task,
        weight=args.weight,
        model_variant=args.model_variant,
        auto_download=args.auto_download,
    )
    print(render_detection(args.image, result))
    # 出框(ok)才算成功;模型缺失 / 运行错误等以非零退出,方便脚本判断。
    return 0 if result.get("status") == "ok" else 1


def _cmd_status(args: argparse.Namespace) -> int:
    from .registry import AlgorithmRegistry

    registry = AlgorithmRegistry.load_default()
    print(render_status(registry.list_algorithms()))
    return 0


def render_detection(image: str, result: dict[str, Any]) -> str:
    status = result.get("status")
    if status == "missing_model":
        return (
            f"⚠️  {result.get('algorithm_id')} 的模型还没就位（需由用户安装或配置独立引擎）。\n"
            "    设置 OMNISENSE_ENGINE_URL__OBJECT_DETECTION；模型由独立引擎管理。"
        )
    if status == "unknown_algorithm":
        # warnings 里已带「available algorithms: ...」,直接透传,引导改正。
        warnings = result.get("warnings") or ["unknown algorithm"]
        return "⚠️  " + "\n    ".join(warnings)
    if status != "ok":
        lines = [f"⚠️  运行状态 {status}"]
        for warning in result.get("warnings") or []:
            lines.append(f"    - {warning}")
        return "\n".join(lines)

    counts = _count_labels(result.get("outputs") or [])
    total = sum(counts.values())
    meta = "模型 {}, {}, {}ms".format(
        result.get("model_id") or "?",
        result.get("provider") or "?",
        result.get("latency_ms", "?"),
    )
    if total == 0:
        return f"✅ 在 {image} 中没有检测到目标。（{meta}）"

    lines = [f"✅ 在 {image} 中检测到 {total} 个目标（{meta}）:"]
    # 按数量降序、同数量按标签字典序,与 Rust 侧一致。
    for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"   • {label} × {count}")
    return "\n".join(lines)


def _count_labels(outputs: Sequence[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for output in outputs:
        label = output.get("label")
        if label:
            counts[label] += 1
    return counts


def render_status(algorithms: Sequence[dict[str, Any]]) -> str:
    if not algorithms:
        return "（当前没有已注册的算法）"

    by_status: dict[str, list[dict[str, Any]]] = {}
    for algorithm in algorithms:
        by_status.setdefault(_availability_key(algorithm), []).append(algorithm)

    lines: list[str] = ["OmniSense —— 当前这台机能用的算法:\n"]
    rendered_any = False
    for status_key, heading in _STATUS_GROUPS:
        group = by_status.get(status_key)
        if not group:
            continue
        rendered_any = True
        lines.append(heading)
        for algorithm in sorted(group, key=lambda a: a.get("id", "")):
            lines.extend(_render_algorithm_lines(algorithm))
        lines.append("")

    # 兜底:出现未预期的 status 也别吞掉。
    known = {key for key, _ in _STATUS_GROUPS}
    leftovers = [a for a in algorithms if _availability_key(a) not in known]
    if leftovers:
        rendered_any = True
        lines.append("• 其它:")
        for algorithm in sorted(leftovers, key=lambda a: a.get("id", "")):
            lines.extend(_render_algorithm_lines(algorithm))
        lines.append("")

    if not rendered_any:
        return "（当前没有已注册的算法）"

    lines.append("跑一张图试试:  omnisense detect <图片路径>")
    return "\n".join(lines).rstrip() + "\n"


def _availability_key(algorithm: dict[str, Any]) -> str:
    availability = algorithm.get("availability")
    if availability:
        return str(availability)
    status = algorithm.get("status", "missing_model")
    if status == "ok":
        return "ready"
    if status == "portable_ready":
        return "hosted"
    return "needs_model"


def _render_algorithm_lines(algorithm: dict[str, Any]) -> list[str]:
    name = algorithm.get("name_zh") or algorithm.get("name_en") or algorithm.get("id")
    capabilities = algorithm.get("capabilities") or []
    cap_text = "、".join(capabilities) if capabilities else "—"
    lines = [f"   • {name}（{algorithm.get('id')}）: {cap_text}"]
    if algorithm.get("id") == DEFAULT_ALGORITHM:
        lines.append("       调用:  omnisense detect <图片>")
    return lines


_STATUS_GROUPS = [("ready", "✅ 可用（仅合成 mock）"), ("hosted", "📦 独立引擎已配置（可用性以实际调用为准）"), ("needs_model", "⚠️  需要模型或独立引擎（用户自行下载安装）")]
