import copy
from contextlib import ExitStack, contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import load_llm
import product_chat
import product_rag


PROJECT_DIR = Path(__file__).resolve().parents[1]


class ProductChatTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.answer_path = self.directory / "llm-answer.txt"
        self.prompt_path = self.directory / "llm-prompt.txt"
        self.query_path = self.directory / "query.json"
        self.run_path = self.directory / "run.json"
        self.model_path = self.directory / "backend-stub.gguf"
        self.model_path.write_bytes(b"Test placeholder; no model weights")
        self.events = []
        self.question = "How can this filter handle changing feed?"
        self.rewritten_question = "Filter options for changing feed conditions"
        self.expected_retrieval_query = self.rewritten_question
        self.records = [{
            "id": "brochure:passage:#/texts/1",
            "source": "brochure.pdf",
            "pages": [2, 3],
            "label": "passage",
            "review_status": "pending",
            "text": "Unique supplied evidence: Larox\u00ae adjusts each cycle.",
            "provenance": [{"page_no": 2}],
            "members": [{"id": "brochure:#/texts/1", "pages": [2, 3]}],
        }]
        self.original_records = copy.deepcopy(self.records)
        self.retrieval_error = None
        self.loading_error = None
        self.loading_failure_at = 1
        self.rewrite_error = None
        self.rewrite_finish_reason = "stop"
        self.generation_error = None
        self.closing_error = None
        self.closing_failure_at = 1
        self.no_results = False
        self.load_attempts = 0
        self.loaded_models = []
        self.completion_models = []
        self.dll_directory_open = False
        owner = self

        class RetrieverStub:
            def __init__(self, records):
                owner.assertEqual(records, owner.records)

            def load(self):
                owner.assertTrue(
                    all(model.closed for model in owner.loaded_models),
                    "Retrieval must not start while an LLM remains loaded.",
                )
                owner.assertFalse(owner.dll_directory_open)
                owner.events.append("retriever-loaded")

            def retrieve(self, question, top_k):
                owner.events.append("retrieved")
                owner.assertEqual(question, owner.expected_retrieval_query)
                owner.assertEqual(top_k, product_rag.RESULT_LIMIT)
                if owner.retrieval_error:
                    raise owner.retrieval_error
                if owner.no_results:
                    return []
                return [{**record, "score": 0.75} for record in owner.records]

            def close(self):
                owner.events.append("retriever-closed")

        def isolated_retrieval_stub(query, top_k, verbose, on_start):
            # Actual subprocess/JSON transfer is covered by worker tests. Here
            # retain real retrieval/logging code with fake model backends.
            if on_start is not None:
                on_start(12345)
            return product_rag.retrieve(query, top_k=top_k, verbose=verbose)

        class LlamaStub:
            def __init__(self, **kwargs):
                owner.assertTrue(owner.dll_directory_open)
                owner.assertTrue(all(model.closed for model in owner.loaded_models))
                owner.load_attempts += 1
                if owner.loading_error and owner.load_attempts == owner.loading_failure_at:
                    owner.events.append("llm-load-failed")
                    raise owner.loading_error
                self.closed = False
                self.index = owner.load_attempts
                owner.loaded_models.append(self)
                owner.events.append("llm-loaded")

            def create_completion(self, **kwargs):
                owner.assertFalse(self.closed)
                owner.assertTrue(owner.dll_directory_open)
                owner.completion_models.append(self)
                prompt = kwargs["prompt"]
                owner.assertIn(owner.question, prompt)
                owner.assertEqual(kwargs["temperature"], 0.0)
                owner.assertEqual(kwargs["seed"], 42)
                if "Brochure excerpts:" not in prompt:
                    owner.events.append("rewritten")
                    owner.assertNotIn(owner.records[0]["text"], prompt)
                    owner.assertEqual(kwargs["max_tokens"], 128)
                    if owner.rewrite_error:
                        raise owner.rewrite_error
                    text, finish_reason = owner.rewritten_question, owner.rewrite_finish_reason
                else:
                    owner.events.append("generated")
                    owner.assertIn(owner.records[0]["text"], prompt)
                    owner.assertIn("Pages: [2, 3]", prompt)
                    owner.assertIn("Review status: pending", prompt)
                    owner.assertEqual(kwargs["max_tokens"], 1024)
                    if owner.generation_error:
                        raise owner.generation_error
                    text, finish_reason = "Grounded stub answer.", "stop"
                return {
                    "choices": [{"text": text, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 123, "completion_tokens": 8, "total_tokens": 131},
                }

            def close(self):
                owner.assertFalse(self.closed, "The LLM must close exactly once.")
                owner.assertTrue(owner.dll_directory_open)
                self.closed = True
                owner.events.append("llm-closed")
                if owner.closing_error and self.index == owner.closing_failure_at:
                    raise owner.closing_error

        @contextmanager
        def dll_directory(_path):
            self.assertFalse(self.dll_directory_open)
            self.dll_directory_open = True
            try:
                yield
            finally:
                self.dll_directory_open = False

        backend = types.ModuleType("llama_cpp")
        backend.Llama = LlamaStub
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(patch.multiple(
            product_chat,
            ANSWER_PATH=self.answer_path,
            QUERY_PATH=self.query_path,
            RUN_PATH=self.run_path,
            create=True,
        ))
        stack.enter_context(patch.object(product_rag, "project_dir", self.directory))
        stack.enter_context(patch.object(product_rag, "load_processed_records", return_value=self.records))
        stack.enter_context(patch.object(product_rag, "_ProductRetriever", RetrieverStub))
        stack.enter_context(patch.object(product_chat, "retrieve", side_effect=isolated_retrieval_stub))
        stack.enter_context(patch.multiple(
            load_llm,
            MODEL_PATH=self.model_path,
            PROMPT_PATH=self.prompt_path,
            ANSWER_PATH=self.answer_path,
        ))
        stack.enter_context(patch.object(
            load_llm, "load_retrieval_context",
            side_effect=AssertionError("Live generation must not read a saved retrieval log."),
        ))
        stack.enter_context(patch.object(
            os, "add_dll_directory", side_effect=dll_directory, create=True,
        ))
        stack.enter_context(patch.dict(sys.modules, {"llama_cpp": backend}))

    def read_json(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def assert_failed_run(self, error_text=None):
        answer_log = self.answer_path.read_text(encoding="utf-8")
        self.assertIn("Run status: FAILED", answer_log)
        self.assertNotIn("Old successful answer", answer_log)
        run = self.read_json(self.run_path)
        self.assertEqual(run["status"], "FAILED")
        if error_text:
            self.assertIn(error_text, answer_log)
            self.assertIn(error_text, run["traceback"])
        self.assertTrue(all(model.closed for model in self.loaded_models))
        self.assertFalse(self.dll_directory_open)
        return run

    def test_phased_retrieval_starts_after_every_loaded_llm_is_closed(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("No diagnostic-log input")):
            answer = product_chat.answer_question(self.question)

        self.assertEqual(answer, "Grounded stub answer.")
        self.assertEqual(self.events, [
            "llm-loaded", "rewritten", "llm-closed", "retriever-loaded", "retrieved",
            "retriever-closed", "llm-loaded", "generated", "llm-closed",
        ])
        self.assertEqual(len(self.loaded_models), 2)
        self.assertIsNot(self.loaded_models[0], self.loaded_models[1])
        self.assertEqual(self.completion_models, self.loaded_models)
        self.assertTrue(all(model.closed for model in self.loaded_models))
        self.assertFalse(self.dll_directory_open)
        self.assertEqual(self.records, self.original_records)
        prompt = self.prompt_path.read_text(encoding="utf-8")
        self.assertIn(f"Customer question:\n{self.question}\n", prompt)
        self.assertNotIn(self.rewritten_question, prompt)
        self.assertIn("ID: brochure:passage:#/texts/1", prompt)
        self.assertIn("Source: brochure.pdf", prompt)
        self.assertIn("Reranker score: 0.75", prompt)
        answer_log = self.answer_path.read_text(encoding="utf-8")
        self.assertIn("Context source: Live structured retrieval records", answer_log)
        self.assertIn("Run status: COMPLETED", answer_log)
        self.assertIn("Finish reason: stop", answer_log)
        logs = self.directory / "tmp/retrieval_runs"
        latest = (logs / "latest.txt").read_bytes()
        histories = list(logs.glob("retrieval_*.txt"))
        self.assertEqual(len(histories), 1)
        self.assertEqual(latest, histories[0].read_bytes())
        self.assertIn(self.rewritten_question, latest.decode("utf-8"))

    def test_query_and_stage_logs_preserve_both_questions_and_timings(self):
        product_chat.answer_question(self.question)
        query = self.read_json(self.query_path)
        self.assertEqual(query["original_question"], self.question)
        self.assertEqual(query["retrieval_query"], self.rewritten_question)
        self.assertTrue(query["rewrite_enabled"])
        run = self.read_json(self.run_path)
        self.assertEqual(run["status"], "COMPLETED")
        self.assertEqual(run["original_question"], self.question)
        self.assertEqual(run["retrieval_query"], self.rewritten_question)
        self.assertEqual(run["selected_passages"], 1)
        self.assertEqual(run["retrieval_pid"], 12345)
        for phase in ("query_llm_load", "query_rewrite", "query_preparation", "retrieval", "generation", "total"):
            self.assertGreaterEqual(run["timings_seconds"][phase], 0)

    def test_no_rewrite_uses_only_one_answer_model_after_retrieval(self):
        self.expected_retrieval_query = self.question
        answer = product_chat.answer_question(self.question, rewrite_query=False)
        self.assertEqual(answer, "Grounded stub answer.")
        self.assertEqual(self.events, [
            "retriever-loaded", "retrieved", "retriever-closed",
            "llm-loaded", "generated", "llm-closed",
        ])
        self.assertEqual(len(self.loaded_models), 1)
        query = self.read_json(self.query_path)
        self.assertEqual(query["original_question"], self.question)
        self.assertEqual(query["retrieval_query"], self.question)
        self.assertFalse(query["rewrite_enabled"])

    def test_original_question_is_kept_exactly_even_when_search_query_changes(self):
        self.question = "  Can Larox PF handle feed changes without new equipment?  "
        product_chat.answer_question(self.question)
        self.assertEqual(self.read_json(self.query_path)["original_question"], self.question)
        self.assertIn(f"Customer question:\n{self.question}\n", self.prompt_path.read_text(encoding="utf-8"))

    def test_rewrite_failure_closes_first_model_and_skips_retrieval(self):
        self.answer_path.write_text("Old successful answer", encoding="utf-8")
        self.rewrite_error = RuntimeError("query rewrite stub failed")
        with self.assertRaisesRegex(RuntimeError, "query rewrite stub failed"):
            product_chat.answer_question(self.question)
        self.assertEqual(self.events, ["llm-loaded", "rewritten", "llm-closed"])
        self.assert_failed_run("query rewrite stub failed")

    def test_empty_or_truncated_rewrite_does_not_retrieve(self):
        for rewritten, reason in (("   ", "stop"), ("incomplete query", "length")):
            with self.subTest(reason=reason):
                self.events.clear()
                self.loaded_models.clear()
                self.rewritten_question, self.rewrite_finish_reason = rewritten, reason
                with self.assertRaises(ValueError):
                    product_chat.answer_question(self.question)
                self.assertEqual(self.events, ["llm-loaded", "rewritten", "llm-closed"])
                self.assert_failed_run()

    def test_retrieval_failure_occurs_with_first_llm_already_closed(self):
        self.answer_path.write_text("Old successful answer", encoding="utf-8")
        self.retrieval_error = RuntimeError("retrieval stub failed")
        with self.assertRaisesRegex(RuntimeError, "retrieval stub failed"):
            product_chat.answer_question(self.question)
        self.assertEqual(self.events, [
            "llm-loaded", "rewritten", "llm-closed", "retriever-loaded", "retrieved", "retriever-closed",
        ])
        self.assertEqual(len(self.loaded_models), 1)
        self.assertEqual(self.assert_failed_run("retrieval stub failed")["stage"], "retrieval")
        retrieval_log = self.directory / "tmp/retrieval_runs/latest.txt"
        self.assertIn("Run status: FAILED", retrieval_log.read_text(encoding="utf-8"))

    def test_no_evidence_skips_answer_model_loading(self):
        self.no_results = True
        with self.assertRaisesRegex(ValueError, "No passages were retrieved"):
            product_chat.answer_question(self.question)
        self.assertEqual(len(self.loaded_models), 1)
        self.assertNotIn("generated", self.events)
        self.assert_failed_run()

    def test_generation_failure_closes_answer_model_and_keeps_failed_log(self):
        self.generation_error = RuntimeError("generation stub failed")
        with self.assertRaisesRegex(RuntimeError, "generation stub failed"):
            product_chat.answer_question(self.question)
        self.assertEqual(len(self.loaded_models), 2)
        self.assertEqual(self.events[-2:], ["generated", "llm-closed"])
        self.assertEqual(self.assert_failed_run("generation stub failed")["stage"], "generation")
        self.assertIn("Context source: Live structured retrieval records", self.answer_path.read_text(encoding="utf-8"))

    def test_blank_question_does_not_start_either_stage(self):
        for question in ("   ", None):
            with self.subTest(question=question):
                with self.assertRaisesRegex(ValueError, "Question cannot be empty"):
                    product_chat.answer_question(question)
        self.assertEqual(self.events, [])

    def test_formatter_rejects_missing_page_metadata_before_loading_answer_model(self):
        bad_record = {**self.records[0], "score": 0.75, "pages": []}
        with self.assertRaisesRegex(ValueError, "page references"):
            load_llm.generate_answer(self.question, [bad_record])
        self.assertEqual(self.events, [])
        self.assertIn("Run status: FAILED", self.answer_path.read_text(encoding="utf-8"))

    def test_main_loads_before_input_then_reloads_after_retrieval(self):
        def enter_question(_prompt):
            self.events.append("question-input")
            self.assertEqual(len(self.loaded_models), 1)
            self.assertFalse(self.loaded_models[0].closed)
            return self.question

        with patch("builtins.input", side_effect=enter_question):
            product_chat.main([])
        self.assertEqual(self.events, [
            "llm-loaded", "question-input", "rewritten", "llm-closed", "retriever-loaded", "retrieved",
            "retriever-closed", "llm-loaded", "generated", "llm-closed",
        ])
        self.assertEqual(len(self.loaded_models), 2)

    def test_main_preserves_exact_original_input_for_final_answer(self):
        self.question = "  Does Larox PF handle changes in feed?  "
        with patch("builtins.input", return_value=self.question):
            product_chat.main([])
        query = self.read_json(self.query_path)
        self.assertEqual(query["original_question"], self.question)
        self.assertIn(f"Customer question:\n{self.question}\n", self.prompt_path.read_text(encoding="utf-8"))

    def test_main_no_rewrite_asks_input_without_loading_first_llm(self):
        self.expected_retrieval_query = self.question

        def enter_question(_prompt):
            self.events.append("question-input")
            self.assertEqual(self.loaded_models, [])
            return self.question

        with patch("builtins.input", side_effect=enter_question):
            product_chat.main(["--no-rewrite"])
        self.assertEqual(self.events, [
            "question-input", "retriever-loaded", "retrieved", "retriever-closed",
            "llm-loaded", "generated", "llm-closed",
        ])

    def test_first_model_loading_failure_records_failed_without_retrieval_or_input(self):
        self.loading_error = RuntimeError("model loading stub failed")
        for invoke in (
            lambda: product_chat.answer_question(self.question),
            lambda: product_chat.main([]),
        ):
            with self.subTest(entry_point=invoke):
                self.events.clear()
                self.load_attempts = 0
                self.answer_path.write_text("Old successful answer", encoding="utf-8")
                with patch("builtins.input") as enter_question:
                    with self.assertRaisesRegex(RuntimeError, "model loading stub failed"):
                        invoke()
                enter_question.assert_not_called()
                self.assertEqual(self.events, ["llm-load-failed"])
                self.assert_failed_run("model loading stub failed")

    def test_answer_model_loading_failure_marks_run_failed(self):
        self.loading_error = RuntimeError("answer model loading failed")
        self.loading_failure_at = 2
        with self.assertRaisesRegex(RuntimeError, "answer model loading failed"):
            product_chat.answer_question(self.question)
        self.assertEqual(self.events[-1], "llm-load-failed")
        self.assertEqual(len(self.loaded_models), 1)
        self.assert_failed_run("answer model loading failed")

    def test_main_input_failure_closes_first_llm_and_records_failed(self):
        for input_error in (EOFError("no input"), KeyboardInterrupt()):
            with self.subTest(exception=type(input_error).__name__):
                self.events.clear()
                self.loaded_models.clear()
                with patch("builtins.input", side_effect=input_error):
                    with self.assertRaises(type(input_error)):
                        product_chat.main([])
                self.assertEqual(self.events, ["llm-loaded", "llm-closed"])
                self.assertEqual(self.assert_failed_run()["stage"], "question input")

    def test_main_blank_input_closes_first_llm_without_retrieval(self):
        with patch("builtins.input", return_value="   "):
            with self.assertRaisesRegex(ValueError, "Question cannot be empty"):
                product_chat.main([])
        self.assertEqual(self.events, ["llm-loaded", "llm-closed"])
        self.assertEqual(self.assert_failed_run()["stage"], "question input")

    def test_generation_interruption_closes_answer_model_and_records_failed(self):
        self.generation_error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            product_chat.answer_question(self.question)
        self.assertEqual(len(self.loaded_models), 2)
        self.assertEqual(self.events[-2:], ["generated", "llm-closed"])
        self.assertEqual(self.assert_failed_run()["stage"], "generation")

    def test_borrowed_generation_never_constructs_or_closes_another_model(self):
        for generation_error in (None, RuntimeError("borrowed generation failed")):
            with self.subTest(failure=generation_error is not None):
                self.events.clear()
                self.loaded_models.clear()
                self.completion_models.clear()
                self.generation_error = generation_error
                with load_llm.load_model() as model:
                    with patch.object(load_llm, "load_model", side_effect=AssertionError("Duplicate LLM")):
                        if generation_error:
                            with self.assertRaisesRegex(RuntimeError, "borrowed generation failed"):
                                load_llm.generate_answer(self.question, [{**self.records[0], "score": 0.75}], llm=model)
                        else:
                            answer = load_llm.generate_answer(self.question, [{**self.records[0], "score": 0.75}], llm=model)
                            self.assertEqual(answer, "Grounded stub answer.")
                    self.assertFalse(model.closed)
                    self.assertEqual(self.events, ["llm-loaded", "generated"])
                    self.assertIs(self.completion_models[0], model)
                self.assertEqual(self.events[-1], "llm-closed")

    def test_standalone_generation_owns_and_closes_its_model(self):
        for generation_error in (None, RuntimeError("standalone generation failed")):
            with self.subTest(failure=generation_error is not None):
                self.events.clear()
                self.loaded_models.clear()
                self.completion_models.clear()
                self.generation_error = generation_error
                records = [{**self.records[0], "score": 0.75}]
                if generation_error:
                    with self.assertRaisesRegex(RuntimeError, "standalone generation failed"):
                        load_llm.generate_answer(self.question, records)
                else:
                    self.assertEqual(load_llm.generate_answer(self.question, records), "Grounded stub answer.")
                self.assertEqual(self.events, ["llm-loaded", "generated", "llm-closed"])
                self.assertEqual(len(self.loaded_models), 1)
                self.assertFalse(self.dll_directory_open)

    def test_query_model_cleanup_failure_skips_retrieval(self):
        self.closing_error = RuntimeError("query model cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "query model cleanup failed"):
            product_chat.answer_question(self.question)
        self.assertEqual(self.events, ["llm-loaded", "rewritten", "llm-closed"])
        self.assertFalse(self.query_path.exists(), "A query snapshot must not claim successful preparation after cleanup fails.")
        self.assert_failed_run("query model cleanup failed")

    def test_answer_model_cleanup_failure_replaces_success_with_failed_status(self):
        self.closing_error = RuntimeError("answer model cleanup failed")
        self.closing_failure_at = 2
        self.answer_path.write_text("Old successful answer", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "answer model cleanup failed"):
            product_chat.answer_question(self.question)
        self.assertEqual(self.events[-2:], ["generated", "llm-closed"])
        self.assert_failed_run("answer model cleanup failed")


class ImportSafetyTests(unittest.TestCase):
    def test_entry_point_imports_without_native_dependencies_or_side_effects(self):
        script = """
import os
import sys
for name in ('torch', 'sentence_transformers', 'transformers', 'llama_cpp'):
    sys.modules[name] = None
original_environment = dict(os.environ)
import product_chat
import load_llm
assert dict(os.environ) == original_environment
assert callable(product_chat.answer_question)
assert callable(load_llm.rewrite_query)
"""
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-c", script],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
