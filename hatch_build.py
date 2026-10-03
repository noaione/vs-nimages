from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import NamedTuple

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags

# VapourSynth reads this directory, then loads the first variant in the manifest
# that the host CPU supports, looking for `<name><platform extension>` with the
# variant suffix between them.
MANIFEST_HEADER = "[VapourSynth Manifest V1]\n"

# x86-64 microarchitecture levels. The baseline build targets v2 (SSE4.2), so it
# runs on anything from Nehalem onward and emits no AVX2 outside the guarded
# functions. The avx2 build lets the compiler use AVX2 everywhere; the runtime
# checks in the kernels stay, so loading it on a CPU without AVX2 is still safe,
# it is just not the build to install there.
X86_64_V2 = "x86-64-v2"
X86_64_V3 = "x86-64-v3"


class Variant(NamedTuple):
    """One CPU optimization level of the plugin."""

    #: Suffix between the name and the extension, empty for the baseline build.
    suffix: str
    #: Value for cargo's `-C target-cpu`.
    target_cpu: str


def target_triple(environment: dict[str, str]) -> str:
    return environment.get("CARGO_BUILD_TARGET", "").lower()


def is_x86_64(environment: dict[str, str]) -> bool:
    triple = target_triple(environment)
    if triple:
        return "x86_64" in triple or "amd64" in triple
    return sysconfig.get_platform().startswith(("win-amd64", "linux-x86_64"))


def plugin_extension(environment: dict[str, str]) -> str:
    triple = target_triple(environment)
    if "windows" in triple or "mingw" in triple:
        return ".dll"
    if "darwin" in triple or "apple" in triple:
        return ".dylib"
    if triple:
        return ".so"
    if sys.platform == "win32":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def plugin_stem(environment: dict[str, str]) -> str:
    """The stem cargo writes for the library.

    Every unix target prefixes a `cdylib` with `lib`, so the file is
    `libvs_nimages.so` or `libvs_nimages.dylib` there, while Windows writes
    `vs_nimages.dll`. That stem is also what the manifest lists, so there is no
    separate name to keep in step with the file.
    """
    if plugin_extension(environment) == ".dll":
        return "vs_nimages"
    return "libvs_nimages"


def base_filename(environment: dict[str, str]) -> str:
    """The name cargo writes for every variant of the library."""
    return f"{plugin_stem(environment)}{plugin_extension(environment)}"


def plugin_filename(environment: dict[str, str], variant: Variant) -> str:
    return f"{plugin_stem(environment)}{variant.suffix}{plugin_extension(environment)}"


def variants(environment: dict[str, str]) -> list[Variant]:
    """The builds this host can produce.

    Only x86-64 has more than one optimization level worth shipping. A macOS
    build is arm64 only, so it gets the single baseline artifact and a manifest
    that names it.
    """
    if not is_x86_64(environment):
        return [Variant("", X86_64_V2)]
    return [Variant("", X86_64_V2), Variant(".avx2", X86_64_V3)]


def manifest(environment: dict[str, str]) -> str:
    """The `manifest.vs` for this plugin.

    The manifest lists the plugin's stem, which carries the platform's library
    prefix: `vs_nimages` on Windows and `libvs_nimages` on macOS and Linux.
    VapourSynth appends the `.<variant>` suffix itself when the host CPU
    supports one, so a variant is never named here.
    """
    return MANIFEST_HEADER + f"{plugin_stem(environment)}\n"


def release_directory(root: Path, environment: dict[str, str]) -> Path:
    target_dir = Path(environment.get("CARGO_TARGET_DIR", root / "target"))
    if not target_dir.is_absolute():
        target_dir = root / target_dir

    cargo_target = environment.get("CARGO_BUILD_TARGET")
    if cargo_target:
        target_dir /= cargo_target
    return target_dir / "release"


def build_plugin(root: Path, environment: dict[str, str], variant: Variant) -> Path:
    cargo = environment.get("CARGO", "cargo")
    flags = f"-C target-cpu={variant.target_cpu}"
    build_environment = dict(environment)
    existing = build_environment.get("RUSTFLAGS", "").strip()
    build_environment["RUSTFLAGS"] = f"{existing} {flags}".strip()

    try:
        subprocess.run(
            [cargo, "build", "--release", "--locked"],
            cwd=root,
            env=build_environment,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Cargo is required to build the VapourSynth plugin") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Cargo failed while building the {variant.target_cpu} plugin"
        ) from error

    # Cargo names every variant of the library the same, because the variant is a
    # compiler flag and not a cargo feature, so the suffix is added here. A
    # variant with no suffix is the artifact cargo just wrote, used as it is.
    built = release_directory(root, environment) / base_filename(environment)
    if not built.is_file():
        raise RuntimeError(f"Cargo completed but did not produce {built}")

    if not variant.suffix:
        return built

    artifact = built.with_name(plugin_filename(environment, variant))
    shutil.copy2(built, artifact)
    return artifact


def wheel_platform_tag() -> str:
    if sys.platform == "linux":
        # A build host's supported tags do not certify the plugin's ABI or
        # external libraries. Only auditwheel may give Linux release wheels
        # their manylinux tag, after checking and bundling those dependencies.
        return sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return next(tags.platform_tags())


# Do not subscript `BuildHookInterface`: hatchling 1.27-1.32.2 declare it
# with one type parameter and 1.32.3 added a second one, so any fixed
# subscript makes the hook unloadable for the other releases. The plain class
# is accepted by every version and the hook never needs the specialization.
class NativePluginHook(BuildHookInterface):  # type: ignore[type-arg]
    """Build the Cargo plugin and place it in VapourSynth's plugin tree."""

    #: VapourSynth looks for a directory of this name under `plugins`.
    plugin_directory = Path("vapoursynth") / "plugins" / "nimages"

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        root = Path(self.root)
        environment = os.environ.copy()

        force_include = build_data.setdefault("force_include", {})
        if not isinstance(force_include, dict):
            raise TypeError("Hatch build data force_include must be a mapping")

        destination_directory = root / self.plugin_directory
        destination_directory.mkdir(parents=True, exist_ok=True)

        for variant in variants(environment):
            artifact = build_plugin(root, environment, variant)
            filename = plugin_filename(environment, variant)
            staged_plugin = destination_directory / filename
            shutil.copy2(artifact, staged_plugin)
            force_include[str(staged_plugin)] = str(self.plugin_directory / filename)

        manifest_path = destination_directory / "manifest.vs"
        manifest_path.write_text(manifest(environment), encoding="utf-8", newline="\n")
        force_include[str(manifest_path)] = str(self.plugin_directory / "manifest.vs")

        # Keep the license and attribution files beside the native artifact in
        # every wheel. Hatch's normal package selection does not include
        # repository-level files or arbitrary license directories.
        force_include[str(root / "LICENSE")] = "LICENSE"

        # The wheel contains native plugins, so it must not be tagged as a
        # universal pure-Python wheel.
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{wheel_platform_tag()}"

    def finalize(
        self,
        version: str,
        build_data: dict[str, object],
        artifact_path: str,
    ) -> None:
        del version, build_data, artifact_path
        shutil.rmtree(
            Path(self.root) / "vapoursynth",
            ignore_errors=True,
        )
