"""Release scanning must find paths even inside compressed Python payloads."""
import importlib.util
import marshal
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "release_paths", Path(__file__).parents[1] / "packaging" / "check_release_paths.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("path", [
    r"C:\Users\Build User\source\crate.rs",
    "C:/Users/Build User/source/crate.rs",
    r"C:\\Users\\Build User\\source\\crate.rs",
])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16le"])
def test_private_windows_paths_are_detected(path, encoding):
    checker = module.PathChecker([r"C:\Users\Build User"])
    checker.check_bytes(path.encode(encoding), "program.exe")
    assert checker.failures == {"program.exe"}


def test_normalized_diagnostic_paths_are_allowed():
    checker = module.PathChecker([r"C:\Users\Build User"])
    checker.check_bytes(b"/mfq-build/home/.cargo/src/crate.rs", "program.exe")
    assert not checker.failures


def test_zip_python_filenames_are_inspected(tmp_path):
    code = compile("pass", r"C:\Users\Build User\stdlib\module.py", "exec")
    path = tmp_path / "base_library.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("module.pyc", b"\0" * 16 + marshal.dumps(code))
    checker = module.PathChecker([r"C:\Users\Build User"])
    checker.check_file(path)
    assert any("!module.pyc" in item for item in checker.failures)


def test_python_constant_tuples_are_inspected():
    code = compile("paths = (r'C:\\Users\\Build User\\source', 'other')", "module.py", "exec")
    checker = module.PathChecker([r"C:\Users\Build User"])
    checker.check_code(code, "module")
    assert checker.failures == {"module"}


def test_matches_across_scan_blocks_are_detected(tmp_path):
    path = tmp_path / "program.exe"
    path.write_bytes(b"\0" * (4 * 1024 * 1024 - 5) + br"C:\Users\Build User\source")
    checker = module.PathChecker([r"C:\Users\Build User"])
    checker.check_file(path)
    assert checker.failures == {str(path)}
