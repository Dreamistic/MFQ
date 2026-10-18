"""Package the existing Windows release as an install-free ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def payload(project: Path, executable: Path) -> dict[str, Path]:
    tauri = project / "MFQStudio/src-tauri"
    bundle = json.loads((tauri / "tauri.release-windows.conf.json").read_text())["bundle"]
    files: dict[str, Path] = {}
    names: set[str] = set()

    def add(source: Path, destination: str) -> None:
        path = PurePosixPath(destination.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in str(path):
            raise ValueError(f"Unsafe archive path: {destination}")
        name = str(path)
        if name.casefold() in names:
            raise ValueError(f"Duplicate archive path: {name}")
        if not source.is_file():
            raise FileNotFoundError(source)
        files[name] = source
        names.add(name.casefold())

    add(executable, "mfq-studio.exe")
    for source, destination in bundle["resources"].items():
        root = (tauri / source).resolve()
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    add(path, str(PurePosixPath(destination) / path.relative_to(root).as_posix()))
        else:
            add(root, destination)
    for source in bundle["externalBin"]:
        path = (tauri / f"{source}-x86_64-pc-windows-msvc.exe").resolve()
        add(path, Path(source).name + ".exe")
    for required in ("mfq-runtime.exe", "mfq-diagnostics.exe", "mfq-cli/mfq-cli.exe", "web-ui/index.html"):
        if required not in files:
            raise FileNotFoundError(f"Missing portable resource: {required}")
    return files


def build_archive(project: Path, executable: Path, output: Path) -> Path:
    files = payload(project, executable)
    config = json.loads((project / "MFQStudio/src-tauri/tauri.conf.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"MFQ Studio_{config['version']}_x64-portable.zip"
    partial = archive.with_suffix(".zip.partial")
    total = sum(path.stat().st_size for path in files.values())
    completed = 0
    next_progress = 256 * 1024 * 1024
    print(f"Packing {len(files)} files ({total / 1024**3:.2f} GiB)", flush=True)
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as package:
            for name, path in sorted(files.items()):
                package.write(path, "MFQ Studio/" + name)
                completed += path.stat().st_size
                if completed >= next_progress:
                    print(f"Packed {completed / 1024**3:.2f} / {total / 1024**3:.2f} GiB", flush=True)
                    next_progress = completed + 256 * 1024 * 1024
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"Built {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio-executable", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    build_archive(Path(__file__).resolve().parents[1], args.studio_executable, args.output_directory)


if __name__ == "__main__":
    main()
