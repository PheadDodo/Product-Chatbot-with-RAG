import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch


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


def _return_before_child_exit():
    import threading

    # A real child can send its result and still hang during native/final cleanup.
    threading.Thread(target=time.sleep, args=(30,), daemon=False).start()
    return _return_records()


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

    def test_result_deadline_terminates_child_and_allows_another_request(self):
        from retrieval_worker import _run_in_process

        started = []
        began = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "result"):
            _run_in_process(
                _sleep_before_return, (), on_start=started.append, timeout_seconds=1.0,
            )
        self.assertLess(time.monotonic() - began, 8.0)
        self.assertEqual(len(started), 1)
        self.assertNotIn(started[0], [child.pid for child in multiprocessing.active_children()])
        result = _run_in_process(_return_records, (), timeout_seconds=5.0)
        self.assertEqual(result[0]["text"], "Preserved evidence")

    def test_exit_deadline_terminates_child_that_already_sent_result(self):
        from retrieval_worker import _run_in_process

        started = []
        began = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "exit"):
            _run_in_process(
                _return_before_child_exit, (), on_start=started.append, timeout_seconds=3.0,
            )
        self.assertLess(time.monotonic() - began, 10.0)
        self.assertEqual(len(started), 1)
        self.assertNotIn(started[0], [child.pid for child in multiprocessing.active_children()])
        result = _run_in_process(_return_records, (), timeout_seconds=5.0)
        self.assertEqual(result[0]["review_status"], "pending")

    def test_invalid_deadlines_are_rejected_before_spawning(self):
        from retrieval_worker import _run_in_process, retrieve_isolated

        invalid = (None, True, False, 0, -1, float("inf"), float("-inf"), float("nan"), "1")
        with patch("retrieval_worker.multiprocessing.get_context") as get_context:
            for timeout_seconds in invalid:
                with self.subTest(timeout_seconds=timeout_seconds):
                    with self.assertRaisesRegex(ValueError, "timeout_seconds"):
                        _run_in_process(_return_records, (), timeout_seconds=timeout_seconds)
                    with self.assertRaisesRegex(ValueError, "timeout_seconds"):
                        retrieve_isolated("question", timeout_seconds=timeout_seconds)
            get_context.assert_not_called()

    def test_public_timeout_is_forwarded_and_has_conservative_default(self):
        from retrieval_worker import DEFAULT_RETRIEVAL_TIMEOUT_SECONDS, retrieve_isolated

        self.assertEqual(DEFAULT_RETRIEVAL_TIMEOUT_SECONDS, 180.0)
        with patch("retrieval_worker._run_in_process", return_value=[]) as run:
            self.assertEqual(retrieve_isolated("question", timeout_seconds=12.5), [])
            self.assertEqual(run.call_args.kwargs["timeout_seconds"], 12.5)
            retrieve_isolated("question")
            self.assertEqual(run.call_args.kwargs["timeout_seconds"], 180.0)

    def test_result_and_exit_waits_share_one_deadline(self):
        from retrieval_worker import _run_in_process

        receive, send = Mock(), Mock()
        receive.poll.return_value = True
        receive.recv.return_value = {"status": "ok", "result": []}
        with (
            patch("retrieval_worker.multiprocessing.get_context") as get_context,
            patch("retrieval_worker.monotonic", side_effect=(100.0, 101.0, 103.0)),
        ):
            context = get_context.return_value
            context.Pipe.return_value = (receive, send)
            process = context.Process.return_value
            process.exitcode = 0
            process.is_alive.return_value = False
            self.assertEqual(_run_in_process(_return_records, (), timeout_seconds=5.0), [])
            receive.poll.assert_called_once_with(4.0)
            process.join.assert_any_call(timeout=2.0)

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
