
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
