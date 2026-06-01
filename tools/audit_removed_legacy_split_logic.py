import argparse
import os
import re
import sys


DEFAULT_EXCLUDES = {
    "__pycache__",
    ".git",
    "submodules",
}


DANGEROUS_PATTERNS = [
    ("first-K train pool slice", re.compile(r"train_pool\s*\[\s*:\s*[A-Za-z0-9_]+\s*\]")),
    ("first-K sorted image slice", re.compile(r"sorted_images\s*\[\s*:\s*[A-Za-z0-9_]+\s*\]")),
    ("first-K selected image slice", re.compile(r"selected_images\s*\[\s*:\s*[A-Za-z0-9_]+\s*\]")),
    ("legacy train_only_colmap default", re.compile(r"split_init_policy\s*=\s*[\"']train_only_colmap[\"']")),
    ("preset uses train_only_colmap", re.compile(r"args\.split_init_policy\s*=\s*[\"']train_only_colmap[\"']")),
]


def _iter_python_files(repo):
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDES]
        for filename in files:
            if filename.endswith(".py"):
                path = os.path.join(root, filename)
                if os.path.basename(path) == os.path.basename(__file__):
                    continue
                yield path


def main():
    parser = argparse.ArgumentParser(description="Audit that legacy incorrect split logic is disabled.")
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    failures = []
    for path in _iter_python_files(repo):
        rel = os.path.relpath(path, repo)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        for label, pattern in DANGEROUS_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append((label, rel, line, match.group(0)))

    if failures:
        for label, rel, line, snippet in failures:
            print(f"[LEGACY-SPLIT-AUDIT][FAIL] {label}: {rel}:{line}: {snippet}")
        print("[LEGACY-SPLIT-AUDIT] status=FAIL")
        return 1

    print("[LEGACY-SPLIT-AUDIT] status=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

