from dataclasses import FrozenInstanceError
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_artifacts import (
    RunArtifacts, atomic_write_json, create_run_artifacts, exclusive_job,
)


def _hold_job_lock(path, connection):
    try:
        with exclusive_job(path):
            connection.send("acquired")
            connection.recv()
        connection.send("released")
    finally:
        connection.close()


class RunArtifactsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_runs_have_distinct_directories_and_fixed_artifact_names(self):
        runs = [create_run_artifacts(self.directory / "runs") for _ in range(10)]
        self.assertEqual(len({run.directory for run in runs}), 10)
        for run in runs:
            self.assertTrue(run.directory.is_dir())
            self.assertEqual(run.run_path, run.directory / "run.json")
            self.assertEqual(run.query_path, run.directory / "query.json")
            self.assertEqual(run.prompt_path, run.directory / "llm-prompt.txt")
            self.assertEqual(run.answer_path, run.directory / "llm-answer.txt")
            self.assertEqual(list(run.directory.iterdir()), [])
        with self.assertRaises(FrozenInstanceError):
            runs[0].directory = self.directory
        self.assertEqual(RunArtifacts(str(self.directory)).directory, self.directory)

    def test_unique_run_does_not_reuse_a_colliding_directory(self):
        first = create_run_artifacts(self.directory)
        marker = first.directory / "existing.txt"
        marker.write_text("preserved", encoding="utf-8")
        # Exclusive mkdir also protects existing output if a generated name collides.
        with (
            patch("run_artifacts.datetime") as clock,
            patch("run_artifacts.uuid4") as identifier,
        ):
            stamp, suffix = first.directory.name.rsplit("-", 1)
            clock.now.return_value.strftime.return_value = stamp
            identifier.return_value.hex = suffix
            with self.assertRaises(FileExistsError):
                create_run_artifacts(self.directory)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserved")

    def test_json_writes_utf8_using_unique_same_directory_temporary_files(self):
        path = self.directory / "nested" / "run.json"
        import run_artifacts

        replace = run_artifacts.os.replace
        sources = []

        def record_replace(source, destination):
            sources.append(Path(source))
            self.assertEqual(Path(source).parent, path.parent)
            return replace(source, destination)

        with patch("run_artifacts.os.replace", side_effect=record_replace):
            atomic_write_json(path, {"question": "Larox® — feed changes"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {
                "question": "Larox® — feed changes",
            })
            self.assertIn("Larox® —", path.read_text(encoding="utf-8"))
            atomic_write_json(path, {"question": "Original question preserved", "status": "done"})
        self.assertEqual(len(set(sources)), 2)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {
            "question": "Original question preserved", "status": "done",
        })
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_serialization_keeps_previous_json(self):
        path = self.directory / "run.json"
        atomic_write_json(path, {"status": "old"})
        old_bytes = path.read_bytes()
        for value in ({"unsupported": {1, 2}}, {"number": float("nan")}):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    atomic_write_json(path, value)
                self.assertEqual(path.read_bytes(), old_bytes)
                self.assertEqual(list(self.directory.iterdir()), [path])

    def test_failed_replace_keeps_previous_json_and_removes_temporary_file(self):
        path = self.directory / "run.json"
        atomic_write_json(path, {"status": "old"})
        old_bytes = path.read_bytes()
        with patch("run_artifacts.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                atomic_write_json(path, {"status": "new"})
        self.assertEqual(path.read_bytes(), old_bytes)
        self.assertEqual(list(self.directory.iterdir()), [path])

    def test_failed_flush_keeps_previous_json_and_removes_temporary_file(self):
        path = self.directory / "run.json"
        atomic_write_json(path, {"status": "old"})
        old_bytes = path.read_bytes()
        with patch("run_artifacts.os.fsync", side_effect=OSError("flush failed")):
            with self.assertRaisesRegex(OSError, "flush failed"):
                atomic_write_json(path, {"status": "new"})
        self.assertEqual(path.read_bytes(), old_bytes)
        self.assertEqual(list(self.directory.iterdir()), [path])

    def test_job_lock_releases_after_exception_without_deleting_its_file(self):
        path = self.directory / "job.lock"
        with self.assertRaisesRegex(ValueError, "job failed"):
            with exclusive_job(path):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with exclusive_job(path):
                        self.fail("A second job acquired the lock.")
                raise ValueError("job failed")
        self.assertTrue(path.is_file())
        with exclusive_job(path):
            self.assertTrue(path.is_file())
        self.assertTrue(path.is_file())

    def start_lock_owner(self, path):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_hold_job_lock, args=(str(path), child))
        process.start()
        child.close()

        def cleanup():
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            parent.close()
            process.close()

        self.addCleanup(cleanup)
        self.assertTrue(parent.poll(5), "Child did not report lock ownership.")
        self.assertEqual(parent.recv(), "acquired")
        return process, parent

    def test_os_lock_rejects_spawned_contender_and_releases_normally(self):
        path = self.directory / "job.lock"
        process, connection = self.start_lock_owner(path)
        with self.assertRaisesRegex(RuntimeError, "already running"):
            with exclusive_job(path):
                self.fail("Concurrent process acquired the lock.")
        connection.send("release")
        self.assertTrue(connection.poll(5))
        self.assertEqual(connection.recv(), "released")
        process.join(timeout=5)
        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, 0)
        with exclusive_job(path):
            self.assertTrue(path.exists())

    def test_os_releases_job_lock_when_owner_process_is_terminated(self):
        path = self.directory / "job.lock"
        process, _ = self.start_lock_owner(path)
        process.terminate()
        process.join(timeout=5)
        self.assertFalse(process.is_alive())
        with exclusive_job(path):
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
