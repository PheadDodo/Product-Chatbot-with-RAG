"""Answer one product question with separate query, retrieval and LLM phases."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import traceback

from app_config import load_local_config
from load_llm import generate_answer, load_model
from load_llm import rewrite_query as rewrite_search_query
from product_rag import RESULT_LIMIT
from retrieval_worker import retrieve_isolated as retrieve
from run_artifacts import atomic_write_json, create_run_artifacts, exclusive_job


PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "tmp/product_runs"
LOCK_PATH = PROJECT_DIR / "tmp/product-chat.lock"


def _write_json(path, value):
    atomic_write_json(path, value)


def _save_status(path, stage, question=None, *, failed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"Stage: {stage}\n"
    if question is not None:
        header = f"Question: {question}\n" + header
    header += "Context source: Live structured retrieval records\n"
    status = "FAILED" if failed else "STARTED"
    details = "\n" + traceback.format_exc() if failed else ""
    path.write_text(
        header + f"Run status: {status}\n" + details, encoding="utf-8"
    )


def _run_question(get_question, *, rewrite_query, config=None):
    config = load_local_config() if config is None else config
    with exclusive_job(LOCK_PATH):
        artifacts = create_run_artifacts(RUNS_DIR)
        return _execute_question(
            get_question, rewrite_query=rewrite_query, config=config, artifacts=artifacts,
        )


def _execute_question(get_question, *, rewrite_query, config, artifacts):
    started = perf_counter()
    timings = {}
    report = {
        "status": "STARTED",
        "stage": "loading query LLM" if rewrite_query else "question input",
        "rewrite_enabled": rewrite_query,
        "original_question": None,
        "retrieval_query": None,
        "run_id": artifacts.directory.name,
        "output_directory": str(artifacts.directory),
        "configuration": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(config).items()
        },
        "timings_seconds": timings,
    }

    def stage(name):
        report["stage"] = name
        report["elapsed_seconds"] = round(perf_counter() - started, 3)
        _write_json(artifacts.run_path, report)

    def read_question():
        stage("question input")
        question = get_question()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")
        if len(question) > config.max_question_chars:
            raise ValueError("Question exceeds the configured character limit.")
        report["original_question"] = question
        return question

    _save_status(artifacts.answer_path, report["stage"])
    stage(report["stage"])
    try:
        preparation_started = perf_counter()
        if rewrite_query:
            with load_model(config=config) as llm:
                timings["query_llm_load"] = round(
                    perf_counter() - preparation_started, 3
                )
                question = read_question()
                stage("rewriting query")
                _save_status(artifacts.answer_path, report["stage"], question)
                rewrite_started = perf_counter()
                retrieval_query = rewrite_search_query(question, llm=llm)
                timings["query_rewrite"] = round(
                    perf_counter() - rewrite_started, 3
                )
                stage("closing query LLM")
            # The first model is closed before loading either retrieval model.
        else:
            question = read_question()
            retrieval_query = question
        timings["query_preparation"] = round(
            perf_counter() - preparation_started, 3
        )
        report["retrieval_query"] = retrieval_query
        _write_json(artifacts.query_path, {
            "original_question": question,
            "retrieval_query": retrieval_query,
            "rewrite_enabled": rewrite_query,
        })

        stage("retrieval")
        _save_status(artifacts.answer_path, report["stage"], question)
        print("Retrieving relevant brochure passages...", flush=True)
        print("Search query:", retrieval_query, flush=True)
        retrieval_started = perf_counter()

        def record_retrieval_pid(pid):
            report["retrieval_pid"] = pid
            stage("retrieval")

        passages = retrieve(
            retrieval_query, top_k=RESULT_LIMIT, verbose=False,
            on_start=record_retrieval_pid,
            timeout_seconds=config.retrieval_timeout_seconds,
        )
        timings["retrieval"] = round(perf_counter() - retrieval_started, 3)
        if not passages:
            raise ValueError("No passages were retrieved for this question.")
        report["selected_passages"] = len(passages)

        # retrieve() has released its own models before the answer LLM loads.
        stage("generation")
        print("Loading the answer LLM and generating an answer...", flush=True)
        generation_started = perf_counter()
        answer = generate_answer(
            question, passages, output_dir=artifacts.directory, config=config,
        )
        timings["generation"] = round(perf_counter() - generation_started, 3)
        timings["total"] = round(perf_counter() - started, 3)
        report["status"] = "COMPLETED"
        stage("completed")
        print("Saved query:", artifacts.query_path, flush=True)
        print("Saved stage timings:", artifacts.run_path, flush=True)
        return answer
    except BaseException:
        report["status"] = "FAILED"
        report["traceback"] = traceback.format_exc()
        timings["total"] = round(perf_counter() - started, 3)
        _write_json(artifacts.run_path, report)
        _save_status(artifacts.answer_path, report["stage"], report["original_question"], failed=True)
        raise


def answer_question(question, *, rewrite_query=True):
    """Interpret, unload, retrieve, and load a fresh model for the answer."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")
    config = load_local_config()
    if len(question) > config.max_question_chars:
        raise ValueError("Question exceeds the configured character limit.")
    return _run_question(lambda: question, rewrite_query=rewrite_query, config=config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-rewrite", action="store_true",
        help="Use the original search query and skip the first LLM load.",
    )
    args = parser.parse_args(argv)
    answer = _run_question(
        lambda: input("Enter your product question: "),
        rewrite_query=not args.no_rewrite,
    )
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()
