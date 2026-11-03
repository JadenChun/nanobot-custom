"""Offline stdlib tests: no application, provider, or pytest plugin activation."""
import hashlib
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

    def report(self, root, messages, *, version="1.40.1", files=2, line=1,
               source="nanobot/a.py"):
        diagnostics = [
            {"file": str(root / source), "severity": "error",
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

    def inventory(self, root, entries, *, name="inventory.json"):
        value = {}
        for relative, content in entries.items():
            if content is None:
                value[relative] = "0" * 64
                continue
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            value[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        path = root / name
        path.write_text(json.dumps(value))
        return path

    def deletions(self, root, paths, *, name="deletions.json"):
        path = root / name
        path.write_text(json.dumps(paths))
        return path

    def check(self, before, after, **kwargs):
        return compare(self.report(self.old, before), self.report(self.new, after, **kwargs),
                       self.old, self.new)

    def check_evidence(self, before, after, *, before_entries, after_entries, deleted,
                       **kwargs):
        source = kwargs.get("source", "nanobot/a.py")
        return compare(
            self.report(self.old, before, source=source),
            self.report(self.new, after, **kwargs),
            self.old,
            self.new,
            before_inventory=self.inventory(self.old, before_entries, name="before.json"),
            after_inventory=self.inventory(self.new, after_entries, name="after.json"),
            deletion_manifest=self.deletions(self.new, deleted),
        )

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

    def test_scope_reduction_with_complete_deletion_evidence_is_compared(self):
        result = self.check_evidence(
            [], [],
            before_entries={"nanobot/keep.py": "keep", "nanobot/deleted.py": None},
            after_entries={"nanobot/keep.py": "keep"},
            deleted=["nanobot/deleted.py"],
            files=1,
        )
        self.assertEqual(result["verdict"], "PASS_NO_NEW_DIAGNOSTICS")
        self.assertTrue(result["strict_clean"])

    def test_undeclared_disappearance_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            self.check_evidence(
                [], [],
                before_entries={"nanobot/keep.py": "keep", "nanobot/deleted.py": None},
                after_entries={"nanobot/keep.py": "keep"},
                deleted=["nanobot/other.py"],
                files=1,
            )

    def test_inventory_count_must_match_report(self):
        with self.assertRaisesRegex(ValueError, "baseline inventory count"):
            self.check_evidence(
                [], [],
                before_entries={"nanobot/keep.py": "keep"},
                after_entries={"nanobot/keep.py": "keep"},
                deleted=["nanobot/deleted.py"],
                files=1,
            )

    def test_current_inventory_must_match_actual_sources(self):
        with self.assertRaisesRegex(ValueError, "actual remaining"):
            self.check_evidence(
                [], [],
                before_entries={"nanobot/keep.py": "keep", "nanobot/deleted.py": None},
                after_entries={"nanobot/keep.py": "keep", "nanobot/ghost.py": None},
                deleted=["nanobot/deleted.py"],
                files=2,
            )

    def test_deletion_evidence_must_be_complete_when_supplied(self):
        with self.assertRaisesRegex(ValueError, "requires both inventories"):
            compare(
                self.report(self.old, []),
                self.report(self.new, [], files=1),
                self.old,
                self.new,
                after_inventory=self.inventory(
                    self.new, {"nanobot/keep.py": "keep"}, name="after.json"
                ),
            )

    def test_inventory_paths_must_be_normalized_and_relative(self):
        path = self.new / "inventory.json"
        path.write_text(json.dumps({"/absolute.py": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "normalized and relative"):
            compare(self.report(self.old, []), self.report(self.new, []),
                    self.old, self.new,
                    before_inventory=self.inventory(self.old, {"nanobot/a.py": "a"}),
                    after_inventory=path,
                    deletion_manifest=self.deletions(self.new, ["nanobot/a.py"]))

    def test_new_signature_fails_with_complete_deletion_evidence(self):
        result = self.check_evidence(
            [], ["new"],
            before_entries={"nanobot/keep.py": "keep", "nanobot/deleted.py": None},
            after_entries={"nanobot/keep.py": "keep"},
            deleted=["nanobot/deleted.py"],
            files=1,
            source="nanobot/keep.py",
        )
        self.assertEqual(result["introduced"], 1)
        self.assertEqual(result["verdict"], "FAIL")

    def test_line_movement_with_complete_deletion_evidence_is_not_new(self):
        result = self.check_evidence(
            ["existing"], ["existing"],
            before_entries={"nanobot/keep.py": "keep", "nanobot/deleted.py": None},
            after_entries={"nanobot/keep.py": "keep"},
            deleted=["nanobot/deleted.py"],
            files=1,
            line=90,
            source="nanobot/keep.py",
        )
        self.assertEqual(result["verdict"], "PASS_NO_NEW_DIAGNOSTICS")
        self.assertFalse(result["strict_clean"])

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

    def test_cli_accepts_complete_deletion_evidence(self):
        before = self.report(self.old, [])
        after = self.report(self.new, [], files=1)
        argv = ["check_type_regressions", str(before), str(after),
                "--baseline-root", str(self.old), "--current-root", str(self.new),
                "--before-inventory", str(self.inventory(
                    self.old, {"nanobot/keep.py": "keep", "nanobot/deleted.py": None},
                    name="before.json")),
                "--after-inventory", str(self.inventory(
                    self.new, {"nanobot/keep.py": "keep"}, name="after.json")),
                "--deletion-manifest", str(self.deletions(self.new, ["nanobot/deleted.py"]))]
        output = io.StringIO()
        with patch("sys.argv", argv), redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertEqual(json.loads(output.getvalue())["verdict"], "PASS_NO_NEW_DIAGNOSTICS")


if __name__ == "__main__":
    unittest.main()
