"""Exercise local configuration without loading models or changing settings."""

import copy
import math
import os
import runpy
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

import app_config
from app_config import LocalModelConfig, load_local_config


class LocalModelConfigTests(unittest.TestCase):
    def test_defaults_preserve_existing_local_model_and_resource_settings(self):
        config = load_local_config(environ={})
        self.assertEqual(
            config.model_path,
            Path("D:/ml_projects/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf").resolve(),
        )
        self.assertEqual(config.cuda_bin_dir, Path("D:/ml_projects/cuda/bin/x64").resolve())
        self.assertEqual(config.n_gpu_layers, 52)
        self.assertEqual(config.n_ctx, 10000)
        self.assertEqual(config.n_batch, 512)
        self.assertEqual(config.n_ubatch, 128)
        self.assertEqual(config.max_output_tokens, 1024)
        self.assertEqual(config.retrieval_timeout_seconds, 180.0)
        self.assertEqual(config.max_question_chars, 8000)

    def test_loaded_config_is_immutable(self):
        config = load_local_config(environ={})
        with self.assertRaises(FrozenInstanceError):
            config.n_ctx = 2048

    def test_overrides_are_validated_and_relative_paths_use_explicit_project_root(self):
        environment = {
            "RAG_MODEL_PATH": " models/local.gguf ",
            "RAG_CUDA_BIN_DIR": "runtime/cuda",
            "RAG_N_GPU_LAYERS": "0",
            "RAG_N_CTX": "4096",
            "RAG_N_BATCH": "64",
            "RAG_N_UBATCH": "32",
            "RAG_MAX_OUTPUT_TOKENS": "256",
            "RAG_RETRIEVAL_TIMEOUT_SECONDS": "15.5",
            "RAG_MAX_QUESTION_CHARS": " 900 ",
        }
        original = copy.deepcopy(environment)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_local_config(environment, project_dir=root)
            self.assertEqual(config.model_path, root / "models" / "local.gguf")
            self.assertEqual(config.cuda_bin_dir, root / "runtime" / "cuda")
        self.assertEqual(environment, original)
        self.assertEqual(config.n_gpu_layers, 0)
        self.assertEqual(config.n_ctx, 4096)
        self.assertEqual(config.n_batch, 64)
        self.assertEqual(config.n_ubatch, 32)
        self.assertEqual(config.max_output_tokens, 256)
        self.assertEqual(config.retrieval_timeout_seconds, 15.5)
        self.assertEqual(config.max_question_chars, 900)

    def test_default_project_root_is_module_directory_and_absolute_paths_are_preserved(self):
        config = load_local_config({
            "RAG_MODEL_PATH": "models/local.gguf",
            "RAG_CUDA_BIN_DIR": "runtime/cuda",
        })
        root = Path(app_config.__file__).resolve().parent
        self.assertEqual(config.model_path, root / "models" / "local.gguf")
        self.assertEqual(config.cuda_bin_dir, root / "runtime" / "cuda")
        with tempfile.TemporaryDirectory() as directory:
            absolute = Path(directory) / "external" / "model.gguf"
            config = load_local_config(
                {"RAG_MODEL_PATH": str(absolute)}, root / "another-root",
            )
            self.assertEqual(config.model_path, absolute)

    def test_home_paths_expand_and_model_availability_is_not_required(self):
        config = load_local_config({
            "RAG_MODEL_PATH": "~/models/example-local-config-test.gguf",
            "RAG_CUDA_BIN_DIR": "~/runtime/example-local-config-test",
        })
        self.assertEqual(
            config.model_path, (Path.home() / "models/example-local-config-test.gguf").resolve(),
        )
        self.assertEqual(
            config.cuda_bin_dir, (Path.home() / "runtime/example-local-config-test").resolve(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(Path, "is_file", side_effect=AssertionError("Unexpected availability check")):
                with patch.object(Path, "is_dir", side_effect=AssertionError("Unexpected availability check")):
                    config = load_local_config({
                        "RAG_MODEL_PATH": "missing/model.gguf",
                        "RAG_CUDA_BIN_DIR": "missing/cuda",
                    }, root)
            self.assertEqual(config.model_path, root / "missing/model.gguf")
            self.assertEqual(list(root.iterdir()), [])

    def test_cpu_all_layers_and_positive_layer_counts_are_supported(self):
        for count in (-1, 0, 1, 52):
            with self.subTest(count=count):
                self.assertEqual(
                    load_local_config({"RAG_N_GPU_LAYERS": str(count)}).n_gpu_layers,
                    count,
                )
        for value in (-2, True, False, 1.0, "52"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    replace(LocalModelConfig(), n_gpu_layers=value)

    def test_positive_integer_settings_reject_zero_negative_floats_and_booleans(self):
        settings = {
            "n_ctx": "RAG_N_CTX",
            "n_batch": "RAG_N_BATCH",
            "n_ubatch": "RAG_N_UBATCH",
            "max_output_tokens": "RAG_MAX_OUTPUT_TOKENS",
            "max_question_chars": "RAG_MAX_QUESTION_CHARS",
        }
        for field, variable in settings.items():
            for invalid in (0, -1, True, False, 2.0, "2"):
                with self.subTest(field=field, direct=invalid):
                    with self.assertRaises(ValueError):
                        replace(LocalModelConfig(), **{field: invalid})
            for invalid in ("0", "-1", "1.5", "True", "false", "1e3", "NaN", "Infinity"):
                with self.subTest(field=field, environment=invalid):
                    with self.assertRaises(ValueError):
                        load_local_config({variable: invalid})

    def test_invalid_environment_types_and_blank_settings_are_not_silently_defaulted(self):
        variables = (
            "RAG_MODEL_PATH", "RAG_CUDA_BIN_DIR", "RAG_N_GPU_LAYERS",
            "RAG_N_CTX", "RAG_N_BATCH", "RAG_N_UBATCH",
            "RAG_MAX_OUTPUT_TOKENS", "RAG_RETRIEVAL_TIMEOUT_SECONDS",
            "RAG_MAX_QUESTION_CHARS",
        )
        for variable in variables:
            for invalid in ("", "  ", None, True, False, 8):
                with self.subTest(variable=variable, value=invalid):
                    with self.assertRaisesRegex(ValueError, variable):
                        load_local_config({variable: invalid})
        for invalid in ([], "RAG_N_CTX=2048", 7):
            with self.subTest(environ=invalid):
                with self.assertRaises(ValueError):
                    load_local_config(invalid)

    def test_timeout_is_positive_finite_and_allows_fractional_seconds(self):
        for value in ("0.25", "15", "1e2"):
            with self.subTest(value=value):
                config = load_local_config({"RAG_RETRIEVAL_TIMEOUT_SECONDS": value})
                self.assertEqual(config.retrieval_timeout_seconds, float(value))
                self.assertTrue(math.isfinite(config.retrieval_timeout_seconds))
        for value in ("0", "-1", "nan", "inf", "-inf", "1e400", "no timeout"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    load_local_config({"RAG_RETRIEVAL_TIMEOUT_SECONDS": value})
        for value in (0, -1, True, False, float("nan"), float("inf"), 10**400, "180", None):
            with self.subTest(direct=value):
                with self.assertRaises(ValueError):
                    replace(LocalModelConfig(), retrieval_timeout_seconds=value)

    def test_context_and_batch_relationships_reject_conflicting_settings(self):
        for environment in (
            {"RAG_N_CTX": "1024"},
            {"RAG_N_CTX": "1024", "RAG_MAX_OUTPUT_TOKENS": "1025"},
            {"RAG_N_BATCH": "64", "RAG_N_UBATCH": "65"},
        ):
            with self.subTest(environment=environment):
                with self.assertRaises(ValueError):
                    load_local_config(environment)
        config = load_local_config({
            "RAG_N_CTX": "1024", "RAG_MAX_OUTPUT_TOKENS": "1023",
            "RAG_N_BATCH": "64", "RAG_N_UBATCH": "64",
        })
        self.assertEqual(config.max_output_tokens, config.n_ctx - 1)
        self.assertEqual(config.n_ubatch, config.n_batch)

    def test_invalid_paths_and_project_roots_raise_configuration_errors(self):
        for variable in ("RAG_MODEL_PATH", "RAG_CUDA_BIN_DIR"):
            for invalid in ("model\x00.gguf", "models/one\ntwo.gguf", "cuda\rbin"):
                with self.subTest(variable=variable, value=invalid):
                    with self.assertRaisesRegex(ValueError, variable):
                        load_local_config({variable: invalid})
        for root in ("", "  ", True, 9, "root\x00name"):
            with self.subTest(root=root):
                with self.assertRaisesRegex(ValueError, "project_dir"):
                    load_local_config({}, root)
        for field in ("model_path", "cuda_bin_dir"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    replace(LocalModelConfig(), **{field: "relative/path"})

    def test_only_documented_settings_are_read_and_caller_environment_is_unchanged(self):
        class KnownSettingsOnly(dict):
            def get(self, key, default=None):
                if not key.startswith("RAG_"):
                    raise AssertionError("Unexpected environment access")
                return super().get(key, default)

            def __iter__(self):
                raise AssertionError("Do not enumerate unrelated environment settings")

        environment = KnownSettingsOnly({"RAG_N_GPU_LAYERS": "0"})
        config = load_local_config(environment)
        self.assertEqual(config.n_gpu_layers, 0)
        self.assertEqual(dict.items(environment), {"RAG_N_GPU_LAYERS": "0"}.items())
        original = os.environ.copy()
        with patch.dict(os.environ, {"RAG_N_GPU_LAYERS": "0"}):
            patched = os.environ.copy()
            self.assertEqual(load_local_config().n_gpu_layers, 0)
            self.assertEqual(dict(os.environ), patched)
        self.assertEqual(dict(os.environ), original)

    def test_import_does_not_resolve_paths_read_data_or_import_native_backends(self):
        with patch.object(Path, "resolve", side_effect=AssertionError("Unexpected filesystem resolution")):
            with patch.object(Path, "read_text", side_effect=AssertionError("Unexpected data read")):
                with patch.object(Path, "write_text", side_effect=AssertionError("Unexpected data write")):
                    with patch.dict(os.environ, {"RAG_N_CTX": "not-an-integer"}):
                        namespace = runpy.run_path(app_config.__file__, run_name="config_import_check")
        self.assertTrue(callable(namespace["load_local_config"]))
        source = Path(app_config.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import llama_cpp", source)
        self.assertNotIn("import transformers", source)


if __name__ == "__main__":
    unittest.main()
