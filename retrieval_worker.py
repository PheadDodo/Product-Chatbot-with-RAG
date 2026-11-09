"""Run retrieval in an owned process so its CUDA context exits before generation."""

import multiprocessing
import traceback

from product_rag import RESULT_LIMIT


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


def _run_in_process(target, arguments, *, on_start=None):
    """Return only after the child exits; propagate failures and own cleanup."""
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
        try:
            message = receive.recv()
        except EOFError as error:
            process.join()
            raise RuntimeError(
                f"Retrieval worker exited without a result (exit code {process.exitcode})."
            ) from error
        # Receive before joining: a large result can otherwise fill the pipe and
        # deadlock the sender. Joining here guarantees native CUDA cleanup first.
        process.join()
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


def retrieve_isolated(query, top_k=RESULT_LIMIT, *, verbose=True, on_start=None):
    """Retrieve structured records in a fresh child, fully exiting before return.

    ``on_start(pid)`` exposes the owned child to diagnostics. It does not transfer
    ownership: this call joins the child on success and terminates it on failure
    or interruption. Importing the module never loads models or changes caches.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be an integer of at least 1.")
    return _run_in_process(_retrieve, (query, top_k, verbose), on_start=on_start)
