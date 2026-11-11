# 打包期把 algorithms/*/manifest.yaml 生成到 omnisense/bundled_algorithms/,
# 让 wheel/sdist 自带算法清单而 git 只维护 algorithms/ 一份真相源。
# 从 sdist 再构建 wheel 时 algorithms/ 不存在(MANIFEST.in prune),
# 此时 sdist 里已含生成好的 bundled_algorithms/,跳过重新生成即可。
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "algorithms"
BUNDLED = ROOT / "omnisense" / "bundled_algorithms"


def sync_bundled_manifests() -> None:
    if not SOURCE.is_dir():
        if not BUNDLED.is_dir():
            raise RuntimeError(
                "neither algorithms/ nor omnisense/bundled_algorithms/ exists; "
                "cannot bundle algorithm manifests"
            )
        return
    if BUNDLED.exists():
        shutil.rmtree(BUNDLED)
    for manifest in sorted(SOURCE.glob("*/manifest.yaml")):
        destination = BUNDLED / manifest.parent.name / "manifest.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest, destination)


class BuildPyWithBundledManifests(build_py):
    def run(self):
        sync_bundled_manifests()
        super().run()


class SdistWithBundledManifests(sdist):
    def run(self):
        sync_bundled_manifests()
        super().run()


setup(
    cmdclass={
        "build_py": BuildPyWithBundledManifests,
        "sdist": SdistWithBundledManifests,
    }
)
