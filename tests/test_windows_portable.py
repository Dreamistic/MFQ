from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "windows_portable", Path(__file__).parents[1] / "packaging/build_portable_windows.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def project(tmp_path):
    root = tmp_path / "source tree"
    tauri = root / "MFQStudio/src-tauri"
    tauri.mkdir(parents=True)
    (tauri / "tauri.conf.json").write_text(json.dumps({"version": "0.4.0"}))
    (tauri / "tauri.release-windows.conf.json").write_text(json.dumps({"bundle": {
        "resources": {
            "../dist": "web-ui", "../../packaging/Resources/mfq-cli": "mfq-cli",
            "../../packaging/windows-runtime": "", "../../packaging/Resources/licenses": "licenses",
        },
        "externalBin": ["../../packaging/sidecars/mfq-runtime", "../../packaging/sidecars/mfq-diagnostics"],
    }}))
    for path in (
        "build/mfq-studio.exe", "MFQStudio/dist/index.html",
        "packaging/Resources/mfq-cli/mfq-cli.exe", "packaging/Resources/mfq-cli/_internal/python.dll",
        "packaging/windows-runtime/cudart64_12.dll", "packaging/Resources/licenses/NOTICE",
        "packaging/sidecars/mfq-runtime-x86_64-pc-windows-msvc.exe",
        "packaging/sidecars/mfq-diagnostics-x86_64-pc-windows-msvc.exe",
        "build/private-server.log", "packaging/.venv/private.txt",
    ):
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(file.name.encode())
    return root


def test_zip_contains_the_installer_resources_without_build_files(tmp_path):
    root = project(tmp_path)
    archive = module.build_archive(root, root / "build/mfq-studio.exe", tmp_path / "output")
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert set(package.namelist()) == {
            "MFQ Studio/mfq-studio.exe", "MFQ Studio/mfq-runtime.exe", "MFQ Studio/mfq-diagnostics.exe",
            "MFQ Studio/mfq-cli/mfq-cli.exe", "MFQ Studio/mfq-cli/_internal/python.dll",
            "MFQ Studio/cudart64_12.dll", "MFQ Studio/web-ui/index.html", "MFQ Studio/licenses/NOTICE",
        }
        assert package.read("MFQ Studio/mfq-runtime.exe") == b"mfq-runtime-x86_64-pc-windows-msvc.exe"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive.with_suffix(".zip.sha256").read_text() == f"{digest}  {archive.name}\n"


def test_incomplete_payload_does_not_replace_a_previous_archive(tmp_path):
    root = project(tmp_path)
    (root / "MFQStudio/dist/index.html").unlink()
    output = tmp_path / "output"
    output.mkdir()
    archive = output / "MFQ Studio_0.4.0_x64-portable.zip"
    archive.write_bytes(b"previous package")
    with pytest.raises(FileNotFoundError, match="web-ui/index.html"):
        module.build_archive(root, root / "build/mfq-studio.exe", output)
    assert archive.read_bytes() == b"previous package"
    assert not archive.with_suffix(".zip.partial").exists()


@pytest.mark.parametrize("destination", ["../outside", "/outside", "C:/outside", "MFQ-STUDIO.EXE"])
def test_unsafe_or_conflicting_resource_destination_is_rejected(tmp_path, destination):
    root = project(tmp_path)
    config = root / "MFQStudio/src-tauri/tauri.release-windows.conf.json"
    value = json.loads(config.read_text())
    value["bundle"]["resources"] = {"../../build/mfq-studio.exe": destination}
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.payload(root, root / "build/mfq-studio.exe")
