import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _return_records():
    return [{
        "id": "example:1", "source": "example.pdf", "pages": [1, 2],
        "review_status": "pending", "text": "Preserved evidence", "score": 0.75,
        "worker_pid": os.getpid(),
    }]


def _raise_error():
    raise ValueError("Explicit worker failure")


def _abrupt_exit():
    os._exit(7)


def _unpicklable_result():
    return lambda: None


def _sleep_before_return():
    import time

    time.sleep(30)
    return []


class RetrievalWorkerTests(unittest.TestCase):
    def test_child_exits_before_structured_records_return(self):
        from retrieval_worker import _run_in_process

        started = []
        result = _run_in_process(_return_records, (), on_start=started.append)
        self.assertEqual(len(started), 1)
        self.assertNotEqual(started[0], os.getpid())
        self.assertEqual(result[0]["worker_pid"], started[0])
        self.assertEqual(result[0]["review_status"], "pending")
        self.assertEqual(result[0]["pages"], [1, 2])
        self.assertNotIn(started[0], [child.pid for child in multiprocessing.active_children()])

    def test_remote_error_includes_exception_and_traceback(self):
        from retrieval_worker import _run_in_process

        with self.assertRaisesRegex(RuntimeError, "ValueError: Explicit worker failure") as caught:
            _run_in_process(_raise_error, ())
        self.assertIn("_raise_error", str(caught.exception))
        self.assertIn("Traceback", str(caught.exception))

    def test_abrupt_exit_is_reported_instead_of_hanging(self):
        from retrieval_worker import _run_in_process

        with self.assertRaisesRegex(RuntimeError, "7"):
            _run_in_process(_abrupt_exit, ())

    def test_result_serialization_failure_is_reported(self):
        from retrieval_worker import _run_in_process

        with self.assertRaisesRegex(RuntimeError, "pickle|Pickl"):
            _run_in_process(_unpicklable_result, ())

    def test_callback_failure_and_interrupt_clean_up_owned_child(self):
        from retrieval_worker import _run_in_process

        for exception in (RuntimeError("callback failed"), KeyboardInterrupt()):
            started = []

            def fail_callback(pid):
                started.append(pid)
                raise exception

            with self.subTest(exception=type(exception).__name__):
                with self.assertRaises(type(exception)):
                    _run_in_process(_sleep_before_return, (), on_start=fail_callback)
                self.assertNotIn(started[0], [child.pid for child in multiprocessing.active_children()])

    def test_invalid_request_does_not_start_a_process(self):
        from retrieval_worker import retrieve_isolated

        for query, top_k in ((" ", 1), (None, 1), ("question", 0), ("question", True)):
            with self.subTest(query=query, top_k=top_k):
                with self.assertRaises(ValueError):
                    retrieve_isolated(query, top_k)

    def test_import_does_not_load_backends_or_change_environment(self):
        script = """
import os
import sys
for name in ('numpy', 'torch', 'sentence_transformers', 'transformers', 'llama_cpp'):
    sys.modules[name] = None
environment = dict(os.environ)
import retrieval_worker
assert dict(os.environ) == environment
assert callable(retrieval_worker.retrieve_isolated)
"""
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-c", script],
            cwd=PROJECT_DIR, capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
