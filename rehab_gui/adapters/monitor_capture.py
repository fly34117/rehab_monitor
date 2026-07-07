"""诊断监控器捕获适配 — monkey-patch 4个诊断渲染器

零修改 rehab_monitor/ 源码：
1. 抑制 __init__ 中的 cv2.namedWindow / cv2.resizeWindow（不弹独立窗口）
2. 捕获 render() 中的 cv2.imshow 输出的 canvas（推送到 GUI bridge）
"""
import cv2
import numpy as np
from functools import wraps

from rehab_monitor.logging_setup import get_logger

logger = get_logger("monitor_capture")


def suppress_cv_windows(cls):
    """Patch 类的 __init__，抑制 cv2.namedWindow / cv2.resizeWindow 调用

    Args:
        cls: 要 patch 的类（如 AngleMonitor）
    """
    original_init = cls.__init__

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        # 临时替换 cv2 窗口函数为空操作
        orig_named = cv2.namedWindow
        orig_resize = cv2.resizeWindow
        cv2.namedWindow = lambda *a, **k: None
        cv2.resizeWindow = lambda *a, **k: None
        try:
            original_init(self, *args, **kwargs)
        finally:
            cv2.namedWindow = orig_named
            cv2.resizeWindow = orig_resize

    cls.__init__ = patched_init
    logger.debug(f"已抑制 {cls.__name__} 的独立窗口创建")


def capture_render(cls, push_func, render_name="render"):
    """Patch 类的 render 方法，捕获 canvas 并推送到 GUI bridge

    原始 render() 内部调用 cv2.imshow(window_name, canvas) 显示画面但不返回。
    此 patch 临时替换 cv2.imshow，在调用时捕获 canvas 副本，
    调用原始 render 后将 canvas 推送到 bridge。

    Args:
        cls: 要 patch 的类
        push_func: bridge.push_diag_xxx 函数
        render_name: render 方法名（默认 "render"）
    """
    original_render = getattr(cls, render_name)

    @wraps(original_render)
    def patched_render(self, *args, **kwargs):
        captured = {"canvas": None}

        orig_imshow = cv2.imshow

        def capture_imshow(win_name, mat):
            if mat is not None and isinstance(mat, np.ndarray) and mat.size > 0:
                captured["canvas"] = mat.copy()

        cv2.imshow = capture_imshow
        try:
            result = original_render(self, *args, **kwargs)
        finally:
            cv2.imshow = orig_imshow

        # 推送捕获的 canvas
        canvas = captured["canvas"]
        if canvas is not None and canvas.size > 0:
            push_func(canvas)

        return result

    setattr(cls, render_name, patched_render)
    logger.debug(f"已捕获 {cls.__name__}.{render_name} 的渲染输出")


def patch_all_monitors(bridge):
    """对 4 个诊断监控器应用全部 patch

    Args:
        bridge: ThreadBridge 实例
    """
    from rehab_monitor.angle_monitor import AngleMonitor
    from rehab_monitor.trajectory_monitor import TrajectoryMonitor
    from rehab_monitor.skeleton_viewer import SkeletonViewer
    from rehab_monitor.gait_metrics_viewer import GaitMetricsViewer

    suppress_cv_windows(AngleMonitor)
    capture_render(AngleMonitor, bridge.push_diag_angle)

    suppress_cv_windows(TrajectoryMonitor)
    capture_render(TrajectoryMonitor, bridge.push_diag_trajectory)

    suppress_cv_windows(SkeletonViewer)
    capture_render(SkeletonViewer, bridge.push_diag_skeleton)

    suppress_cv_windows(GaitMetricsViewer)
    capture_render(GaitMetricsViewer, bridge.push_diag_gait)

    logger.info("已 patch 全部 4 个诊断监控器（抑制窗口 + 捕获渲染）")
