"""Импорты пакета и пути не требуют скачанного корпуса или GPU."""

import importlib
import json
import os
from pathlib import Path
import pkgutil
import subprocess
import sys
import tempfile
import unittest

import armenian_llm
from armenian_llm.paths import PROJECT_DIR


class PackageTests(unittest.TestCase):
    def test_all_modules_import_without_data(self):
        for module in pkgutil.walk_packages(armenian_llm.__path__, "armenian_llm."):
            if not module.name.endswith(".__main__"):
                with self.subTest(module=module.name):
                    importlib.import_module(module.name)

    def test_paths_are_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory) / "external"
            environment = dict(os.environ, LLM_STORAGE_ROOT=str(storage))
            environment.pop("LLM_PROJECT_ROOT", None)
            result = subprocess.run(
                [sys.executable, "-c",
                 "import json; from armenian_llm.paths import PROJECT_DIR, STORAGE_ROOT, FACTS_CONFIG; "
                 "print(json.dumps([str(PROJECT_DIR),str(STORAGE_ROOT),FACTS_CONFIG.is_file()]))"],
                cwd=directory, env=environment, check=True, capture_output=True, text=True,
            )
        project, actual_storage, config_exists = json.loads(result.stdout)
        self.assertEqual(Path(project), PROJECT_DIR)
        self.assertEqual(Path(actual_storage), storage.resolve())
        self.assertTrue(config_exists)

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "armenian_llm", "--help"],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("export-model", result.stdout)
        self.assertIn("continue-pretrain", result.stdout)
