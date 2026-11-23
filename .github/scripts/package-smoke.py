"""Package all release crates and run the egui example using only the resulting archives."""

import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
import tomllib

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "target"
EXAMPLE = ROOT / "examples" / "egui"
CRATES = (
    "iletki-core",
    "iletki-wgpu",
    "iletki-2d",
    "iletki-3d",
    "iletki-egui",
    "iletki-slint",
    "iletki",
)
LICENSES = ("LICENSE-MIT", "LICENSE-APACHE")


def cargo(*args, cwd=ROOT):
    subprocess.run(("cargo", *args), cwd=cwd, check=True)


def main():
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    version = workspace["workspace"]["package"]["version"]
    arguments = [arg for name in CRATES for arg in ("-p", name)]
    # Cargo's temporary registry is used for packaging. An independent consumer validates the
    # archives after extraction and exercises the same version requirements as crates.io users.
    cargo("package", "--locked", "--no-verify", "--allow-dirty", "--target-dir", TARGET, *arguments)

    with TemporaryDirectory(prefix="package-smoke-", dir=TARGET) as work_dir:
        work = Path(work_dir)
        archives = work / "archives"
        consumer = work / "consumer"
        shutil.copytree(EXAMPLE, consumer, ignore=shutil.ignore_patterns("target", "Cargo.lock"))
        patches = []
        for name in CRATES:
            basename = f"{name}-{version}"
            with tarfile.open(TARGET / "package" / f"{basename}.crate", "r:gz") as archive:
                archive.extractall(archives, filter="data")
            crate = archives / basename
            for filename in LICENSES:
                if (crate / filename).read_bytes() != (ROOT / filename).read_bytes():
                    raise ValueError(f"{basename} has a stale {filename}")
            if not (crate / "README.md").read_text(encoding="utf-8").strip():
                raise ValueError(f"{basename} has an empty README.md")
            patches.append(f"{name} = {{ path = {json.dumps('../archives/' + basename)} }}")

        with (consumer / "Cargo.toml").open("a", encoding="utf-8") as manifest:
            # Compile the Slint adapter from its archive as part of this consumer too.
            manifest.write(f'\niletki-slint = "={version}"\n')
            manifest.write("\n[patch.crates-io]\n" + "\n".join(patches) + "\n")
        with (consumer / "src" / "main.rs").open("a", encoding="utf-8") as source:
            source.write("\nuse iletki_slint as _;\n")
        cargo("generate-lockfile", cwd=consumer)
        lock = tomllib.loads((consumer / "Cargo.lock").read_text(encoding="utf-8"))
        for name in CRATES:
            packages = [entry for entry in lock["package"] if entry["name"] == name]
            if len(packages) != 1 or packages[0]["version"] != version or "source" in packages[0]:
                raise ValueError(f"consumer resolved {name} outside the release archives: {packages}")

        cargo("run", "--locked", "--target-dir", TARGET / "package-example-build", cwd=consumer)
        print(f"Verified the archive consumer and license files for {len(CRATES)} crates.", flush=True)


if __name__ == "__main__":
    main()
