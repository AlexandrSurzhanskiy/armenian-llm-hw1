"""Сохранённый пакет работает отдельно; старые snapshots тоже экспортируются."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from armenian_llm.provenance import snapshot_code
from armenian_llm.reporting.export_model import checkpoint_sources, loading_instructions


class SnapshotTests(unittest.TestCase):
    def test_snapshot_imports_as_standalone_package(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "code"
            hashes = snapshot_code(destination)
            self.assertIn("src/armenian_llm/modeling/gpt.py", hashes)
            self.assertTrue((destination / "README.md").is_file())
            environment = dict(os.environ)
            environment.pop("LLM_PROJECT_ROOT", None)
            result = subprocess.run(
                [sys.executable, "-c",
                 "import json,sys; sys.path.insert(0,sys.argv[1]); "
                 "from armenian_llm.paths import PROJECT_DIR,FACTS_CONFIG; "
                 "print(json.dumps([str(PROJECT_DIR),FACTS_CONFIG.is_file()]))",
                 str(destination / "src")],
                cwd=directory, env=environment, check=True, capture_output=True, text=True,
            )
            project, config_exists = json.loads(result.stdout)
            self.assertEqual(Path(project), destination.resolve())
            self.assertTrue(config_exists)
            sources, layout = checkpoint_sources(Path(directory))
            self.assertEqual(layout, "src_package")
            self.assertIn("code/src/armenian_llm/modeling/gpt.py", sources)
            self.assertIn("code/README.md", sources)

    def test_legacy_flat_snapshot_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "code").mkdir()
            for name in ("model.py", "generate.py"):
                (run / "code" / name).write_text("# old source\n")
            sources, layout = checkpoint_sources(run)
            self.assertEqual(layout, "legacy_flat")
            self.assertEqual(sources["code/model.py"], run / "code/model.py")
            self.assertIn("from model import GPT", loading_instructions(layout))
