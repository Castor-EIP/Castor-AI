"""Manual test client for the podcast AI module.

Connects to a running castostudio-ai-server, starts a podcast session with
the given sources, and prints every scene switch / status / error event
live. Sends periodic KeepAlive messages so the test also exercises that the
server keeps reading the client stream while analysis is running.

Example (local files, fastest way to validate the VAD/state-machine logic
without any streaming infra):

    uv run castostudio-ai-server &
    uv run python scripts/test_podcast_client.py \\
        --source "s1:Cam Hote:/path/to/host.wav" \\
        --source "s2:Cam Invite:/path/to/guest.wav" \\
        --source "s3:Plan Large:/path/to/wide.mp4"

url accepts anything PyAV can open: local file path, rtsp(s)://, rtmp(s)://.
Ctrl+C sends a StopSignal and ends the session cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time

import grpc

from castostudio_ai_server.proto import ia_analysis_pb2, ia_analysis_pb2_grpc

LOGGER = logging.getLogger("test_podcast_client")

EVENT_TYPE_NAMES = {
    ia_analysis_pb2.SERVER_EVENT_UNKNOWN: "UNKNOWN",
    ia_analysis_pb2.SERVER_EVENT_SWITCH_SUGGESTED: "SWITCH_SUGGESTED",
    ia_analysis_pb2.SERVER_EVENT_STATUS_CHANGED: "STATUS_CHANGED",
    ia_analysis_pb2.SERVER_EVENT_ERROR: "ERROR",
}


def parse_source(spec: str) -> ia_analysis_pb2.Source:
    # scene_id:label:url - maxsplit=2 keeps any ':' inside the url intact.
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected scene_id:label:url, got {spec!r}")
    scene_id, label, url = parts
    return ia_analysis_pb2.Source(scene_id=scene_id, label=label, url=url)


def parse_config(pairs: list[str]) -> dict[str, str]:
    config = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"expected key=value, got {pair!r}")
        config[key] = value
    return config


async def outgoing_messages(
    session_id: str,
    sources: list[ia_analysis_pb2.Source],
    keep_alive_interval: float,
    stop_event: asyncio.Event,
):
    yield ia_analysis_pb2.ClientMessage(
        session_id=session_id,
        sources=ia_analysis_pb2.SourceList(sources=sources),
    )
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=keep_alive_interval)
        except asyncio.TimeoutError:
            yield ia_analysis_pb2.ClientMessage(
                session_id=session_id,
                keep_alive=ia_analysis_pb2.KeepAlive(timestamp_ms=int(time.time() * 1000)),
            )
    yield ia_analysis_pb2.ClientMessage(
        session_id=session_id,
        stop=ia_analysis_pb2.StopSignal(reason="test client stopped"),
    )


def format_event(event: ia_analysis_pb2.ServerEvent) -> str:
    kind = EVENT_TYPE_NAMES.get(event.event_type, str(event.event_type))
    payload = event.WhichOneof("payload")
    if payload == "switch_suggestion":
        detail = (
            f"scene_id={event.switch_suggestion.scene_id} "
            f"confidence={event.switch_suggestion.confidence:.2f}"
        )
    elif payload == "status":
        detail = f"state={event.status.state} message={event.status.message!r}"
    elif payload == "error":
        detail = (
            f"code={event.error.error_code} fatal={event.error.is_fatal} "
            f"message={event.error.error_message!r}"
        )
    else:
        detail = "<empty>"
    return f"[{kind}] {detail}"


async def run(args: argparse.Namespace) -> None:
    sources = [parse_source(spec) for spec in args.source]
    config = parse_config(args.config)

    async with grpc.aio.insecure_channel(args.addr) as channel:
        stub = ia_analysis_pb2_grpc.IaAnalysisServiceStub(channel)

        start_response = await stub.StartSession(
            ia_analysis_pb2.StartSessionRequest(module_name="podcast", module_config=config)
        )
        if not start_response.success:
            LOGGER.error(
                "StartSession failed: %s (%s)",
                start_response.message,
                start_response.error_code,
            )
            return

        session_id = start_response.session_id
        LOGGER.info("Session started: %s", session_id)
        for source in sources:
            LOGGER.info("  source scene_id=%s label=%r url=%s", source.scene_id, source.label, source.url)

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass  # signal handlers unsupported on this platform (e.g. Windows)

        call = stub.AnalysisStream(
            outgoing_messages(session_id, sources, args.keep_alive_interval, stop_event)
        )

        try:
            async for event in call:
                print(format_event(event))
        finally:
            stop_event.set()
            await stub.EndSession(ia_analysis_pb2.EndSessionRequest(session_id=session_id))
            LOGGER.info("Session ended: %s", session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--addr", default="localhost:50051", help="gRPC server address")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="scene_id:label:url",
        help="Repeatable, one per camera/audio source.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="key=value",
        help="Repeatable. Podcast module_config override, e.g. --config vad_threshold=0.4",
    )
    parser.add_argument(
        "--keep-alive-interval",
        type=float,
        default=5.0,
        help="Seconds between KeepAlive messages sent to the server.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
