from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "tests")
FORBIDDEN_SUPPRESSIONS = ("# type: ignore", "# noqa")


def _python_modules() -> list[Path]:
    modules = [*REPOSITORY_ROOT.glob("*.py")]
    modules.extend(path for root in PYTHON_ROOTS for path in root.rglob("*.py"))
    return sorted(modules)


def test_python_modules_stay_below_four_hundred_lines() -> None:
    oversized = []
    for path in _python_modules():
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count >= 400:
            oversized.append(f"{path.relative_to(REPOSITORY_ROOT)} ({line_count})")

    assert not oversized, "Python modules must remain below 400 lines: " + ", ".join(oversized)


def test_exception_handlers_name_specific_failures() -> None:
    broad_handlers = []
    for path in _python_modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ExceptHandler) and _is_broad_handler(node):
                broad_handlers.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")

    assert not broad_handlers, "Broad exception handlers are forbidden: " + ", ".join(broad_handlers)


def _is_broad_handler(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    return isinstance(handler.type, ast.Name) and handler.type.id in {"BaseException", "Exception"}


def test_type_checker_suppressions_are_not_committed() -> None:
    suppressions = []
    for path in _python_modules():
        comments = _comments(path)
        if any(marker in comment for comment in comments for marker in FORBIDDEN_SUPPRESSIONS):
            suppressions.append(str(path.relative_to(REPOSITORY_ROOT)))

    assert not suppressions, "Fix typing and lint failures instead of suppressing them: " + ", ".join(suppressions)


def _comments(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [token.string for token in tokens if token.type == tokenize.COMMENT]
