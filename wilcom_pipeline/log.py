"""Console progress reporting.

The pipeline makes calls that take minutes, and the previous version printed
nothing while they were in flight -- so a working run and a hung run looked
identical. That is worse than a cosmetic problem: it invites killing a run that is
fine, and it hid a completed call behind a cached reply I only found by accident.

Three things this module exists to do:

  * **Timestamps on every line**, so a slow stage is visibly slow rather than
    apparently stuck.
  * **A heartbeat during long calls.** A thread prints elapsed seconds while a
    request is in flight, which is the only signal available when the provider
    returns nothing until the whole answer is generated.
  * **One place to turn output down.** `QUIET` suppresses info lines but never
    warnings or errors, so `--quiet` cannot hide a problem.

Deliberately not a logging framework. There are no handlers, levels or
formatters to configure; it writes to stdout and stderr and stops.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time

# Set by `set_quiet`. Warnings and errors ignore it.
QUIET = False

# How often the heartbeat prints while a call is in flight. Long enough not to
# spam, short enough that a two-minute call shows several ticks.
HEARTBEAT_SECONDS = 15.0

# Prefer UTF-8 so a Hindi instruction or an Urdu transcript is logged as itself
# rather than as question marks. A deliberate side effect on import: the streams
# are reconfigurable and the alternative is losing the text. `_safe_write` is the
# guarantee, this is only the quality-of-life half -- if a host has already wrapped
# or replaced the streams, the reconfigure fails harmlessly and the fallback holds.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


_T0 = time.time()

# Optional extra destination for every line, used by the GUI to stream progress
# into a textbox. A plain callable rather than a logging.Handler: the point is to
# avoid a framework, and a GUI needs the rendered line, not a LogRecord.
_SINK = None

# A second, structured destination. The text sink is enough for a log box but not
# for a progress bar: a bar needs to know WHICH step is running, how many of how
# many are done, and how long the current call has been in flight. Recovering that
# by parsing rendered lines would break the moment a message contained a bracket,
# so events carry fields instead.
_EVENT_SINK = None


def set_sink(fn) -> None:
    """Send every rendered line to `fn` as well as stdout.

    The GUI cannot read stdout from inside the same process, so without this it
    could only show output after a stage finished -- which is exactly the
    invisibility that made a working run look hung.
    """
    global _SINK
    _SINK = fn


def clear_sink() -> None:
    global _SINK
    _SINK = None


def set_event_sink(fn) -> None:
    """Send structured events to `fn`, for progress bars and step feeds."""
    global _EVENT_SINK
    _EVENT_SINK = fn


def clear_event_sink() -> None:
    global _EVENT_SINK
    _EVENT_SINK = None


def _event(level: str, message: str, **fields) -> None:
    """Publish one structured event. Never raises."""
    if _EVENT_SINK is None:
        return
    payload = {"level": level, "message": message, "at": time.time()}
    payload.update(fields)
    try:
        _EVENT_SINK(payload)
    except Exception:
        # The same rule as the text sink: a broken listener must not take down
        # the run it is reporting on.
        pass


def set_quiet(flag: bool) -> None:
    global QUIET
    QUIET = bool(flag)


def _clock() -> str:
    """Wall-clock time, so a log can be matched against a provider dashboard."""
    return time.strftime("%H:%M:%S")


def _elapsed() -> str:
    """Seconds since the process started, which is what shows relative cost."""
    return "%6.1fs" % (time.time() - _T0)


def _safe_write(stream, text: str) -> None:
    """Write a line, surviving an encoding the text does not fit.

    A Windows console defaults to cp1252, and this pipeline handles Hindi and Urdu
    tutorials: a model's reasoning can legitimately contain Devanagari or Arabic
    script, and `stream.write` would then raise UnicodeEncodeError and take down a
    run that had already spent money. Losing a log line is a far smaller failure
    than losing the run, so an unrepresentable character is replaced rather than
    raised.
    """
    try:
        stream.write(text)
        stream.flush()
        return
    except UnicodeEncodeError:
        pass
    except Exception:
        return

    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        stream.write(text.encode(encoding, "replace").decode(encoding, "replace"))
        stream.flush()
    except Exception:
        # A stream that cannot be written to at all is not worth crashing over:
        # stdout being broken is not the caller's problem.
        pass


def _emit(prefix: str, message: str, stream=None, level: str = "info",
          **fields) -> None:
    stream = stream or sys.stdout
    line = "[%s %s] %s%s" % (_clock(), _elapsed(), prefix, message)
    _safe_write(stream, line + "\n")
    _event(level, message, **fields)
    if _SINK is not None:
        try:
            _SINK(line)
        except Exception:
            # A sink that raises must never take down the run it is reporting on.
            # Whatever the sink's problem is, losing the log line is the smaller
            # failure.
            pass


def info(message: str) -> None:
    """Ordinary progress. Suppressed by --quiet."""
    if not QUIET:
        _emit("", message, level="info")


def step(message: str) -> None:
    """A stage or sub-step boundary. Suppressed by --quiet."""
    if not QUIET:
        _emit("", message, level="phase")


phase = step


def task(message: str, detail: str = "") -> None:
    """An intermediate step inside a stage.

    A stage runs for minutes, so "stage 2 is running" tells a user nothing about
    whether anything is happening. Tasks are the units in between: cutting a clip,
    extracting frames, sending a request, validating a reply. They are what the UI
    lists so each step is visibly accounted for.
    """
    if not QUIET:
        _emit("  ", message, level="task", detail=detail)


def detail(message: str) -> None:
    """Indented supporting detail. Suppressed by --quiet."""
    if not QUIET:
        _emit("    ", message, level="detail")


def ok(message: str) -> None:
    """A completed unit of work. Suppressed by --quiet."""
    if not QUIET:
        _emit("ok  ", message, level="ok")


def warn(message: str) -> None:
    """Never suppressed. Something is off but the run continues."""
    _emit("WARN ", message, stream=sys.stderr, level="warn")


def error(message: str) -> None:
    """Never suppressed."""
    _emit("ERR  ", message, stream=sys.stderr, level="error")


def progress(index: int, total: int, message: str = "") -> None:
    """A numbered counter, e.g. '[3/21]'. Carries the numbers as event fields."""
    if not QUIET:
        _emit("", "[%d/%d] %s" % (index, total, message),
              level="progress", index=index, total=total, text=message)


def gauge(pct: float, message: str = "") -> None:
    """Progress that is reported many times a second.

    Unlike `progress`, this writes NO log line: ffmpeg reports every few hundred
    milliseconds and one line per report would bury everything else. It still
    publishes an event, which is what a progress bar needs and a text log does not.
    """
    _event("progress", message, pct=max(0.0, min(100.0, float(pct))),
           text=message)


@contextlib.contextmanager
def heartbeat(label: str, interval: float = HEARTBEAT_SECONDS):
    """Print elapsed time every `interval` seconds while the block runs.

    A daemon thread, so it cannot keep the process alive if the main thread dies.
    The final line reports the true duration, which is the number worth reading:
    the ticks exist only to prove something is happening.
    """
    stop = threading.Event()
    start = time.time()

    def tick():
        while not stop.wait(interval):
            if not QUIET:
                elapsed = time.time() - start
                _emit("    ", "... %s still running, %.0fs elapsed"
                      % (label, elapsed),
                      level="tick", label=label, elapsed=elapsed)

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def money(usd: float) -> str:
    """Format a small amount, keeping precision where it matters."""
    if usd < 0.01:
        return "$%.5f" % usd
    return "$%.4f" % usd


def tokens(n) -> str:
    """Format a token count with thousands separators."""
    try:
        return "{:,}".format(int(n))
    except (TypeError, ValueError):
        return str(n)


def table(rows: list, headers: list) -> None:
    """Aligned columns. Widths are computed from the data, not guessed."""
    if not rows:
        return
    widths = [len(h) for h in headers]
    cells = [[str(c) for c in row] for row in rows]
    for row in cells:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    if not QUIET:
        _emit("", "  ".join(h.ljust(widths[i])
                            for i, h in enumerate(headers)))
        _emit("", "  ".join("-" * w for w in widths))
        for row in cells:
            _emit("", "  ".join(c.ljust(widths[i])
                                for i, c in enumerate(row)))
