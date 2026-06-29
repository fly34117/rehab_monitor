"""Unified fall-alert popup — OpenCV window for desktop visibility.

Each fall event spawns a short-lived Python subprocess that owns its own
OpenCV window, avoiding GTK threading conflicts with the main pipeline.
A 3-second cooldown deduplicates near-simultaneous alerts.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time

log = logging.getLogger("fall_popup")

# ---- shared state ----
_last_enqueue_time = 0.0
_current_popup = None               # Popen handle of the active popup subprocess
ENQUEUE_COOLDOWN = 3.0              # seconds — global dedup across sources
POPUP_DISMISS_SEC = 8

# Written to a temp .py file so the subprocess is a proper script
# (easier to debug than inline -c code).
_POPUP_SOURCE = r'''
import cv2, json, numpy as np, sys, time, traceback

try:
    with open(sys.argv[1], "r") as fh:
        data = json.load(fh)

    W, H = 440, 240
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # -- bright-red header banner --
    cv2.rectangle(canvas, (0, 0), (W, 60), (0, 0, 220), -1)
    cv2.putText(canvas, "!! FALL DETECTED !!", (50, 44),
                cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2)

    y = 95
    cv2.putText(canvas, f"Source : {data['source']}", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (220, 220, 220), 1)
    y += 42
    cv2.putText(canvas, f"Score  : {data['score']:.2f}", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (220, 220, 220), 1)
    y += 42

    detail = data.get('detail', '')
    if detail:
        cv2.putText(canvas, detail, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (160, 160, 160), 1)
        y += 38

    cv2.putText(canvas, data.get('ts', ''), (W - 130, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 120), 1)
    cv2.putText(canvas, "auto-close in {}s".format(data.get('dismiss', 8)),
                (14, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    cv2.namedWindow("FALL ALERT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FALL ALERT", W, H)
    try:
        cv2.setWindowProperty("FALL ALERT", cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.imshow("FALL ALERT", canvas)

    dismiss = data.get('dismiss', 8)
    deadline = time.time() + dismiss
    while time.time() < deadline:
        key = cv2.waitKey(100) & 0xFF
        if key in (ord('q'), 27):
            break
    cv2.destroyWindow("FALL ALERT")

except Exception:
    traceback.print_exc()
    # Also write to stderr so parent can capture
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()
    # Keep window open a moment so the error is visible
    try:
        time.sleep(2)
    except Exception:
        pass
'''


def enqueue_fall(source, score, detail=None):
    """Thread-safe push from any source (camera / wristband).

    Launches a standalone subprocess that shows the OpenCV popup window.
    """
    global _last_enqueue_time, _current_popup

    now = time.time()
    if now - _last_enqueue_time < ENQUEUE_COOLDOWN:
        log.debug("fall popup suppressed by cooldown (%.1fs < %ss)",
                  now - _last_enqueue_time, ENQUEUE_COOLDOWN)
        return
    _last_enqueue_time = now

    detail_text = ""
    if detail:
        if "location" in detail:
            loc = detail["location"]
            detail_text = f"Loc    : ({loc[0]:.1f}, {loc[1]:.1f}) m"
        elif "peak_magnitude" in detail:
            detail_text = f"Peak Mag: {detail['peak_magnitude']:.1f} m/s2"

    event_json = json.dumps({
        "source": source.upper(),
        "score": float(score),
        "ts": time.strftime("%H:%M:%S", time.localtime(now)),
        "detail": detail_text,
        "dismiss": POPUP_DISMISS_SEC,
    })

    # Kill any still-running popup
    if _current_popup is not None and _current_popup.poll() is None:
        try:
            _current_popup.kill()
        except Exception:
            pass
        _current_popup = None

    # Write event to a temp file so we don't hit cmdline-length limits
    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="fall_popup_", delete=False)
        tmp.write(event_json)
        tmp.close()

        popup_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_popup_window.py")

        # Ensure the helper script exists on disk
        if not os.path.exists(popup_script):
            _write_popup_script(popup_script)

        _current_popup = subprocess.Popen(
            [sys.executable, popup_script, tmp.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        print(f"[弹窗] 跌倒弹窗已启动 (pid={_current_popup.pid}, "
              f"source={source}, score={score:.2f})")

        # Clean up temp file after subprocess exits (fire-and-forget)
        def _cleanup(popup, tmp_path):
            # Read stderr BEFORE wait() to avoid pipe deadlock
            try:
                _, stderr_bytes = popup.communicate(timeout=POPUP_DISMISS_SEC + 5)
            except Exception:
                popup.kill()
                _, stderr_bytes = popup.communicate()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            if popup.returncode != 0 and stderr_bytes:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                if stderr_text:
                    print(f"[弹窗] 子进程异常 (rc={popup.returncode}): {stderr_text[:300]}")

        import threading
        threading.Thread(target=_cleanup, args=(_current_popup, tmp.name),
                         daemon=True).start()

    except Exception:
        print(f"[弹窗] 启动子进程失败: {sys.exc_info()[1]}")


def _write_popup_script(path):
    """Write the popup helper script to disk (one-time)."""
    with open(path, "w") as fh:
        fh.write(_POPUP_SOURCE)
    log.info("popup helper script written to %s", path)


def start_popup_thread():
    """Compatibility stub — no persistent thread needed."""
    return None


def stop_popup():
    """Kill any active popup subprocess."""
    global _current_popup
    if _current_popup is not None:
        try:
            _current_popup.kill()
        except Exception:
            pass
        _current_popup = None
