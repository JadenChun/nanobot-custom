"""Offline stdlib tests: no application, provider, or pytest plugin activation."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.check_type_regressions import compare, main


class TypeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old = self.root / "old"
        self.new = self.root / "new"
        for root in (self.old, self.new):
            root.mkdir()
            (root / "pyproject.toml").write_text(
                '[tool.basedpyright]\ntypeCheckingMode = "strict"\ninclude = ["nanobot"]\n'
            )

    def report(self, root, messages, *, version="1.40.1", files=2, line=1):
        diagnostics = [
            {"file": str(root / "nanobot/a.py"), "severity": "error",
             "rule": "reportArgumentType", "message": message,
             "range": {"start": {"line": line, "character": 0}}}
            for message in messages
        ]
        value = {"version": version, "generalDiagnostics": diagnostics,
                 "summary": {"filesAnalyzed": files, "errorCount": len(diagnostics),
                             "warningCount": 0, "informationCount": 0}}
        path = root / "report.json"
        path.write_text(json.dumps(value))
        return path

    def check(self, before, after, **kwargs):
        return compare(self.report(self.old, before), self.report(self.new, after, **kwargs),
                       self.old, self.new)

    def test_line_movement_is_not_new_and_debt_is_not_clean(self):
        result = self.check(["existing"], ["existing"], line=90)
        self.assertEqual(result["verdict"], "PASS_NO_NEW_DIAGNOSTICS")
        self.assertFalse(result["strict_clean"])

    def test_lower_total_cannot_hide_new_error(self):
        result = self.check(["old", "old", "old"], ["old", "new"])
        self.assertEqual(result["introduced"], 1)
        self.assertEqual(result["resolved"], 2)
        self.assertEqual(result["verdict"], "FAIL")

    def test_duplicate_increase_fails(self):
        self.assertEqual(self.check(["old"], ["old", "old"])["introduced"], 1)

    def test_empty_diagnostics_can_be_clean(self):
        self.assertTrue(self.check(["old"], [])["strict_clean"])

    def test_scope_reduction_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "file count decreased"):
            self.check(["old"], [], files=1)

    def test_version_change_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "version changed"):
            self.check([], [], version="different")

    def test_policy_change_fails_closed(self):
        (self.new / "pyproject.toml").write_text('[tool.basedpyright]\ntypeCheckingMode="basic"')
        with self.assertRaisesRegex(ValueError, "policy changed"):
            self.check([], [])

    def test_inconsistent_summary_fails_closed(self):
        before = self.report(self.old, [])
        after = self.report(self.new, ["new"])
        data = json.loads(after.read_text())
        data["summary"]["errorCount"] = 0
        after.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "summary"):
            compare(before, after, self.old, self.new)

    def test_outside_root_fails_closed(self):
        before = self.report(self.old, [])
        after = self.report(self.new, ["new"])
        data = json.loads(after.read_text())
        data["generalDiagnostics"][0]["file"] = str(self.old / "a.py")
        after.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            compare(before, after, self.old, self.new)

    def test_signature_changes_are_not_hidden_by_same_count(self):
        for field, value in (("rule", "reportReturnType"), ("message", "changed"),
                             ("file", str(self.new / "nanobot/other.py"))):
            with self.subTest(field=field):
                before = self.report(self.old, ["old"])
                after = self.report(self.new, ["old"])
                data = json.loads(after.read_text())
                data["generalDiagnostics"][0][field] = value
                after.write_text(json.dumps(data))
                self.assertEqual(compare(before, after, self.old, self.new)["introduced"], 1)

    def test_cli_exit_codes_and_invalid_json(self):
        for messages, expected in (([], 0), (["new"], 1), (None, 2)):
            with self.subTest(expected=expected):
                before = self.report(self.old, [])
                after = self.report(self.new, messages or [])
                if messages is None:
                    after.write_text("{")
                argv = ["check_type_regressions", str(before), str(after),
                        "--baseline-root", str(self.old), "--current-root", str(self.new)]
                output = io.StringIO()
                with patch("sys.argv", argv), redirect_stdout(output):
                    self.assertEqual(main(), expected)
                self.assertIn("verdict", json.loads(output.getvalue()))


if __name__ == "__main__":
    unittest.main()
