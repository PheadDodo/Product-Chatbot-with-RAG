"""Run retrieval in an owned process so its CUDA context exits before generation."""

import math
import multiprocessing
from numbers import Real
from time import monotonic
import traceback

from product_rag import RESULT_LIMIT


DEFAULT_RETRIEVAL_TIMEOUT_SECONDS = 180.0


def _validate_timeout(timeout_seconds):
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real):
        raise ValueError("timeout_seconds must be a positive finite number.")
    try:
        timeout_seconds = float(timeout_seconds)
    except (OverflowError, ValueError):
        raise ValueError("timeout_seconds must be a positive finite number.") from None
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite number.")
    return timeout_seconds


def _worker_entry(connection, target, arguments):
    try:
        result = target(*arguments)
        connection.send({"status": "ok", "result": result})
    except BaseException as error:
        connection.send({
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
    finally:
        connection.close()


def _run_in_process(
    target, arguments, *, on_start=None,
    timeout_seconds=DEFAULT_RETRIEVAL_TIMEOUT_SECONDS,
):
    """Bound result-availability/exit waits and own child cleanup."""
    timeout_seconds = _validate_timeout(timeout_seconds)
    deadline = monotonic() + timeout_seconds
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker_entry, args=(send, target, arguments))
    started = False
    try:
        process.start()
        started = True
        send.close()
        if on_start is not None:
            on_start(process.pid)

        def join_before_deadline():
            process.join(timeout=max(0.0, deadline - monotonic()))
            if process.is_alive():
                raise TimeoutError(
                    f"Timed out waiting for retrieval worker exit after {timeout_seconds:g} seconds."
                )

        try:
            if not receive.poll(max(0.0, deadline - monotonic())):
                raise TimeoutError(
                    f"Timed out waiting for retrieval worker result after {timeout_seconds:g} seconds."
                )
            message = receive.recv()
        except EOFError as error:
            join_before_deadline()
            raise RuntimeError(
                f"Retrieval worker exited without a result (exit code {process.exitcode})."
            ) from error
        # Receive before joining: a large result can otherwise fill the pipe and
        # deadlock the sender. Joining here guarantees native CUDA cleanup first.
        join_before_deadline()
        if process.exitcode != 0:
            raise RuntimeError(f"Retrieval worker exited with code {process.exitcode}.")
        if message.get("status") == "error":
            raise RuntimeError(
                f"Retrieval worker failed: {message['error']}\n{message['traceback']}"
            )
        if message.get("status") != "ok" or "result" not in message:
            raise RuntimeError("Retrieval worker returned an invalid response.")
        return message["result"]
    finally:
        send.close()
        receive.close()
        if started:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            else:
                process.join()
            process.close()


def _retrieve(query, top_k, verbose):
    from product_rag import retrieve

    return retrieve(query, top_k=top_k, verbose=verbose)


def retrieve_isolated(
    query, top_k=RESULT_LIMIT, *, verbose=True, on_start=None,
    timeout_seconds=DEFAULT_RETRIEVAL_TIMEOUT_SECONDS,
):
    """Retrieve structured records in a fresh child, fully exiting before return.

    ``on_start(pid)`` exposes the owned child to diagnostics. It does not transfer
    ownership: this call joins the child on success and terminates it on failure
    or interruption. ``timeout_seconds`` budgets waiting for result availability
    and worker exit. Startup, callbacks, complete message transfer/deserialization
    and cleanup are synchronous and can extend elapsed time beyond this budget.
    This is not an enforced deadline for the entire job or LLM generation.
    The 180-second default is conservative configuration, not a measured service
    guarantee. Importing the module never loads models or changes caches.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be an integer of at least 1.")
    timeout_seconds = _validate_timeout(timeout_seconds)
    return _run_in_process(
        _retrieve, (query, top_k, verbose), on_start=on_start,
        timeout_seconds=timeout_seconds,
    )
