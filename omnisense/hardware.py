from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
from typing import Any


def detect_hardware() -> dict[str, Any]:
    devices = _hardware_devices()
    provider_probes = _provider_probes(devices)
    compute = ["cpu"]
    torch_info = _torch_info()
    cuda_available = bool(torch_info["cuda"]) or _command_ok(
        "nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"]
    )

    if cuda_available:
        compute.append("cuda")

    if _probe_available(provider_probes, "amd_rocm"):
        compute.append("amd_rocm")
    if _probe_available(provider_probes, "amd_migraphx"):
        compute.append("amd_migraphx")
    if _probe_available(provider_probes, "amd_xdna_vitisai"):
        compute.append("amd_xdna_vitisai")

    if platform.system() == "Darwin":
        compute.append("metal")
        if _module_available("mlx"):
            compute.append("mlx")
    if _axcl_available():
        compute.append("ax650n_axcl")

    model_runtimes = ["pt", "onnx", "mock"]
    if _module_available("tensorrt") or _command_ok("trtexec", ["--version"]):
        model_runtimes.append("tensorrt")
    if _probe_available(provider_probes, "amd_migraphx"):
        model_runtimes.append("migraphx")
    if _probe_available(provider_probes, "amd_xdna_vitisai"):
        model_runtimes.append("vitisai")
    if _module_available("mlx"):
        model_runtimes.append("mlx")
    if _axcl_available():
        model_runtimes.append("axmodel")

    video_codecs = ["software"]
    for accel in _ffmpeg_hwaccels():
        if _video_codec_available(accel, cuda_available):
            video_codecs.append(accel)

    image_codecs = ["software"]
    if "cuda" in compute:
        image_codecs.append("nvjpeg")
    if "metal" in compute:
        image_codecs.append("coreimage")

    return {
        "compute_providers": _dedup(compute),
        "model_runtimes": _dedup(model_runtimes),
        "image_codecs": _dedup(image_codecs),
        "video_codecs": _dedup(video_codecs),
        "devices": devices,
        "provider_probes": provider_probes,
        "provider_smokes": _provider_smokes(),
        "fallback": {
            "compute": "cpu",
            "image_codec": "software",
            "video_codec": "software",
        },
    }


def select_device(preferred_providers: list[str] | None = None) -> str:
    preferred = preferred_providers or ["cuda", "amd_rocm", "metal", "mlx", "cpu"]
    available = set(detect_hardware()["compute_providers"])
    torch_info = _torch_info()

    for provider in preferred:
        if provider == "cuda" and "cuda" in available:
            return "cuda"
        if provider == "amd_rocm" and "amd_rocm" in available:
            return "cuda"
        if provider in {"metal", "mlx"} and {"metal", "mlx"} & available and torch_info["mps"]:
            return "mps"
        if provider == "cpu":
            return "cpu"

    return "cpu"


def _torch_info() -> dict[str, bool]:
    try:
        import torch
    except Exception:
        return {"cuda": False, "hip": False, "mps": False}

    cuda = bool(getattr(torch.cuda, "is_available", lambda: False)())
    version = getattr(torch, "version", None)
    hip = bool(getattr(version, "hip", None))
    mps = bool(
        getattr(getattr(torch, "backends", None), "mps", None)
        and torch.backends.mps.is_available()
    )
    return {"cuda": cuda, "hip": hip, "mps": mps}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _probe_available(probes: list[dict[str, Any]], provider: str) -> bool:
    return any(probe["provider"] == provider and probe["available"] for probe in probes)


def _hardware_devices() -> list[dict[str, Any]]:
    devices = [
        {
            "kind": "cpu",
            "vendor": _cpu_vendor(),
            "name": _cpu_model_name() or platform.machine(),
            "present": True,
        }
    ]
    devices.extend(_devices_from_lspci_output(_lspci_output()))
    if _axcl_available():
        devices.append(
            {
                "kind": "npu",
                "vendor": "axera",
                "name": "AX650N AXCL",
                "present": True,
                "path": os.environ.get("OMNISENSE_NATIVE_AXCL_LIB"),
            }
        )
    return devices


def _cpu_model_name() -> str | None:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name") and ":" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _cpu_vendor() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            vendor = _cpu_vendor_from_cpuinfo(handle.read())
            if vendor:
                return vendor
    except OSError:
        pass
    return platform.machine()


def _cpu_vendor_from_cpuinfo(cpuinfo: str) -> str | None:
    for line in cpuinfo.splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key == "vendor_id":
            if value == "AuthenticAMD":
                return "amd"
            if value == "GenuineIntel":
                return "intel"
            return value.lower()
        if key == "model name":
            lower = value.lower()
            if "amd" in lower or "ryzen" in lower:
                return "amd"
            if "intel" in lower:
                return "intel"
    return None


def _lspci_output() -> str:
    if shutil.which("lspci") is None:
        return ""
    try:
        result = subprocess.run(
            ["lspci", "-nn"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except Exception:
        return ""
    return result.stdout if result.returncode == 0 else ""


def _devices_from_lspci_output(output: str) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if not line:
            continue
        if _display_controller_line(lower):
            kind = "gpu"
            path = _first_dri_render_device_path()
        elif _amd_xdna_line(lower):
            kind = "npu"
            path = _first_accel_device_path()
        else:
            continue
        vendor = "unknown"
        if "amd" in lower or "advanced micro devices" in lower:
            vendor = "amd"
        elif "intel" in lower:
            vendor = "intel"
        elif "nvidia" in lower:
            vendor = "nvidia"
        device = {
            "kind": kind,
            "vendor": vendor,
            "name": line,
            "present": True,
            "bus_id": line.split()[0],
            "device_id": _pci_device_id(line),
        }
        if path:
            device["path"] = path
        devices.append(device)
    return devices


def _display_controller_line(line: str) -> bool:
    return (
        "vga compatible controller" in line
        or "display controller" in line
        or "3d controller" in line
    ) and "ethernet controller" not in line


def _amd_xdna_line(line: str) -> bool:
    return "1022:1502" in line or "1022:17f0" in line or (
        "advanced micro devices" in line and "amd ipu" in line
    )


def _pci_device_id(line: str) -> str | None:
    matches = []
    for part in line.split("["):
        token = part.split("]", 1)[0]
        halves = token.split(":")
        if (
            len(halves) == 2
            and len(halves[0]) == 4
            and len(halves[1]) == 4
            and all(ch in "0123456789abcdefABCDEF" for ch in halves[0] + halves[1])
        ):
            matches.append(token.lower())
    return matches[-1] if matches else None


def _first_dri_render_device_path() -> str | None:
    for index in range(256):
        path = f"/dev/dri/renderD{index}"
        if os.path.exists(path):
            return path
    return None


def _first_accel_device_path() -> str | None:
    for index in range(16):
        path = f"/dev/accel/accel{index}"
        if os.path.exists(path):
            return path
    return None


def _kfd_device_present() -> bool:
    return os.path.exists("/dev/kfd")


def _provider_probes(devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    amd_gpu_present = any(device["kind"] == "gpu" and device["vendor"] == "amd" for device in devices)
    amd_gpu_compute_present = amd_gpu_present and _kfd_device_present()
    xdna_present = _amd_xdna_device_present()
    xdna_supported = _amd_xdna_platform_supported()
    xrt_present = _xrt_available()
    vitisai_present = _vitisai_provider_available()
    return [
        {
            "provider": "cpu",
            "available": True,
            "supported": True,
            "device_present": True,
            "runtime_present": True,
            "reason": "available",
            "evidence": ["host CPU present"],
            "missing": [],
        },
        _provider_probe(
            "amd_rocm",
            amd_gpu_compute_present,
            True,
            _rocm_available(),
            ["AMD GPU device", "/dev/kfd ROCm compute interface"],
            [],
            ["ROCm HIP runtime"],
        ),
        _provider_probe(
            "amd_migraphx",
            amd_gpu_compute_present,
            True,
            _migraphx_available(),
            ["AMD GPU device", "/dev/kfd ROCm compute interface"],
            [],
            ["MIGraphX runtime or ONNX Runtime MIGraphX provider"],
        ),
        _provider_probe(
            "amd_xdna_vitisai",
            xdna_present,
            xdna_supported,
            xrt_present and vitisai_present,
            ["AMD XDNA device"],
            ["Linux PHX/HPT/STX/KRK XDNA PCI ID 1022:1502 or 1022:17f0"],
            ["XRT runtime", "VitisAI/VAIP ONNX Runtime provider"],
        ),
        _provider_probe(
            "ax650n_axcl",
            _axcl_available(),
            True,
            _axcl_available(),
            ["AXCL runtime visible"],
            [],
            ["AXCL runtime"],
        ),
    ]


def _provider_probe(
    provider: str,
    device_present: bool,
    supported: bool,
    runtime_present: bool,
    device_evidence: list[str],
    support_requirements: list[str],
    runtime_requirements: list[str],
) -> dict[str, Any]:
    available = device_present and supported and runtime_present
    evidence: list[str] = []
    missing: list[str] = []
    if device_present:
        evidence.extend(device_evidence)
    else:
        missing.extend(device_evidence)
    if runtime_present:
        evidence.append("provider runtime visible")
    else:
        missing.extend(runtime_requirements)
    if not supported:
        missing.extend(support_requirements or ["supported platform"])
    else:
        evidence.extend(support_requirements)
    return {
        "provider": provider,
        "available": available,
        "supported": supported,
        "device_present": device_present,
        "runtime_present": runtime_present,
        "reason": "available" if available else f"missing {', '.join(missing)}",
        "evidence": evidence,
        "missing": missing,
    }


def _provider_smokes() -> list[dict[str, Any]]:
    smoke_path = os.environ.get(
        "OMNISENSE_AMD_ROCM_SMOKE_JSON", "/tmp/omnisense-amd-rocminfo.json"
    )
    try:
        with open(smoke_path, "r", encoding="utf-8") as handle:
            smoke = json.load(handle)
    except Exception:
        return []
    required = {"provider", "runtime", "status", "reason", "exit_code", "image", "log_path"}
    if not isinstance(smoke, dict) or not required.issubset(smoke):
        return []
    return [
        {
            "provider": str(smoke["provider"]),
            "runtime": str(smoke["runtime"]),
            "status": str(smoke["status"]),
            "reason": str(smoke["reason"]),
            "exit_code": int(smoke["exit_code"]),
            "image": str(smoke["image"]),
            "log_path": str(smoke["log_path"]),
        }
    ]


def _rocm_available() -> bool:
    if os.environ.get("OMNISENSE_NATIVE_ENABLE_ROCM", "").lower() in {"1", "true", "yes"}:
        return True
    return _command_ok("rocminfo", []) or _command_ok("rocm-smi", []) or _lib_dir_has(
        "OMNISENSE_NATIVE_ROCM_LIB",
        ["libamdhip64.so", "libhiprtc.so", "librocm_smi64.so", "amdhip64.dll"],
    )


def _migraphx_available() -> bool:
    if os.environ.get("OMNISENSE_NATIVE_ENABLE_MIGRAPHX", "").lower() in {"1", "true", "yes"}:
        return True
    if _command_ok("migraphx-driver", ["--version"]):
        return True
    return _lib_dir_has(
        "OMNISENSE_NATIVE_MIGRAPHX_LIB",
        [
            "libmigraphx.so",
            "libmigraphx_onnx.so",
            "libonnxruntime_providers_migraphx.so",
            "migraphx.dll",
        ],
    )


def _amd_xdna_vitisai_available() -> bool:
    if os.environ.get("OMNISENSE_NATIVE_ENABLE_AMD_XDNA", "").lower() in {"1", "true", "yes"}:
        return True
    return _amd_xdna_device_present() and _amd_xdna_platform_supported() and _xrt_available() and _vitisai_provider_available()


def _amd_xdna_device_present() -> bool:
    device = os.environ.get("OMNISENSE_NATIVE_AMD_XDNA_DEVICE")
    return bool(device and os.path.exists(device)) or any(
        os.path.exists(f"/dev/accel/accel{index}") for index in range(16)
    )


def _amd_xdna_platform_supported() -> bool:
    if os.environ.get("OMNISENSE_NATIVE_AMD_XDNA_PLATFORM_OK", "").lower() in {"1", "true", "yes"}:
        return True
    if platform.system() == "Linux":
        return _amd_xdna_platform_supported_from_lspci_output(_lspci_output())
    return True


def _amd_xdna_platform_supported_from_lspci_output(output: str) -> bool:
    lower = output.lower()
    return "1022:1502" in lower or "1022:17f0" in lower


def _xrt_available() -> bool:
    return _command_ok("xrt-smi", ["--version"]) or _lib_dir_has(
        "OMNISENSE_NATIVE_XRT_LIB",
        ["libxrt_core.so", "libxrt_coreutil.so", "libxrt_core.so.2", "xrt_core.dll"],
    )


def _vitisai_provider_available() -> bool:
    return _lib_dir_has(
        "OMNISENSE_NATIVE_VITISAI_LIB",
        [
            "libonnxruntime_providers_vitisai.so",
            "libvaip.so",
            "onnxruntime_providers_vitisai.dll",
        ],
    )


def _axcl_available() -> bool:
    if os.environ.get("OMNISENSE_NATIVE_ENABLE_AXCL", "").lower() in {"1", "true", "yes"}:
        return True
    return _lib_dir_has(
        "OMNISENSE_NATIVE_AXCL_LIB",
        ["libaxcl_rt.so", "libaxcl.so", "libaxcl_pcie_host.so", "axclrt.dll"],
    )


def _lib_dir_has(env_name: str, names: list[str]) -> bool:
    lib_dir = os.environ.get(env_name)
    if lib_dir:
        for name in names:
            if os.path.exists(os.path.join(lib_dir, name)):
                return True
    return False


def _command_ok(command: str, args: list[str]) -> bool:
    if shutil.which(command) is None:
        return False
    try:
        result = subprocess.run(
            [command, *args],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except Exception:
        return False
    return result.returncode == 0


def _ffmpeg_hwaccels() -> list[str]:
    if shutil.which("ffmpeg") is None:
        return []
    try:
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("Hardware")
    ]


def _video_codec_available(codec: str, cuda_available: bool) -> bool:
    if codec == "cuda":
        return cuda_available
    if codec == "nvenc":
        return cuda_available
    if codec == "vaapi":
        return _dri_render_device_present()
    if codec == "vulkan":
        return _dri_render_device_present()
    if codec == "qsv":
        return _intel_gpu_present() and _dri_render_device_present()
    if codec == "videotoolbox":
        return platform.system() == "Darwin"
    if codec == "amf":
        return _lib_dir_has("OMNISENSE_NATIVE_ROCM_LIB", ["libamfrt64.so", "libamfrt64.so.1", "amfrt64.dll"])
    return False


def _dri_render_device_present() -> bool:
    return any(os.path.exists(f"/dev/dri/renderD{index}") for index in range(256))


def _intel_gpu_present() -> bool:
    if shutil.which("lspci") is None:
        return False
    try:
        result = subprocess.run(
            ["lspci"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    text = result.stdout.lower()
    return any(
        "intel" in line and any(token in line for token in ["vga", "display", "3d"])
        for line in text.splitlines()
    )


def _dedup(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
