from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_repository.py"
SPEC = importlib.util.spec_from_file_location("verify_repository", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


@unittest.skipUnless(shutil.which("git"), "Git is needed to check ignored build files")
class RepositoryHygieneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git("init", "--quiet")
        (self.root / ".gitignore").write_text("*.pyd\n__pycache__/\n", encoding="utf-8")
        (self.root / "native.pyd").write_bytes(b"generated test fixture")
        cache = self.root / "__pycache__"
        cache.mkdir()
        (cache / "module.pyc").write_bytes(b"generated test fixture")

    def git(self, *arguments: str) -> None:
        subprocess.run(
            ["git", *arguments], cwd=self.root, check=True, capture_output=True
        )

    def test_ignored_build_products_do_not_fail_publication_check(self) -> None:
        with patch.object(VERIFY, "ROOT", self.root):
            VERIFY.verify_hygiene()

    def test_tracked_build_products_still_fail_publication_check(self) -> None:
        self.git("add", "--force", "native.pyd")
        with patch.object(VERIFY, "ROOT", self.root):
            with self.assertRaisesRegex(AssertionError, "stale runtime/build files"):
                VERIFY.verify_hygiene()


if __name__ == "__main__":
    unittest.main()
