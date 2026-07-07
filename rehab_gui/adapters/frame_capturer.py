"""帧捕获适配器 — 注入渲染管线, 拦截 DisplayOverlay 返回的帧"""
import threading
import queue
from functools import wraps

from rehab_monitor.logging_setup import get_logger

logger = get_logger("frame_capturer")


class FrameCapturer:
    """帧捕获适配器 — monkey-patch DisplayOverlay.render

    拦截 rehab_monitor 的渲染输出, 将帧通过回调传给 GUI
    """

    def __init__(self, bridge):
        """初始化帧捕获器

        Args:
            bridge: ThreadBridge 实例
        """
        self._bridge = bridge
        self._frame_count = 0
        self._frame_lock = threading.Lock()
        self._original_render = None
        self._diag_count = 0

        logger.info("帧捕获器已初始化")

    def inject(self):
        """注入 rehab_monitor 管线"""
        try:
            from rehab_monitor.display_overlay import DisplayOverlay
            from rehab_monitor import main as rehab_main

            # 保存原始 render 方法
            self._original_render = DisplayOverlay.render

            # 替换 render 方法
            @wraps(self._original_render)
            def patched_render(self_obj, frame, state):
                """替换的 render 方法"""
                # 调用原始渲染
                result = self._original_render(self_obj, frame, state)

                # 捕获帧
                self._on_frame_captured(result, state)

                return result

            DisplayOverlay.render = patched_render

            # 同时捕获主循环中的诊断窗口调用
            self._patch_diagnostic_calls(rehab_main)

            logger.info("帧捕获注入成功")
        except Exception as e:
            logger.error(f"帧捕获注入失败: {e}")

    def _on_frame_captured(self, frame, state):
        """帧捕获回调

        Args:
            frame: np.ndarray 渲染后的帧
            state: dict 当前状态
        """
        with self._frame_lock:
            self._frame_count += 1

        # 每100帧记录一次日志
        if self._frame_count % 100 == 0:
            logger.info(f"已捕获 {self._frame_count} 帧")

        # 推送主帧
        self._bridge.push_frame(frame)

        # 推送帧计数
        self._bridge.push_frame_count(self._frame_count)

        # 推送跌倒状态
        fall_status = state.get("fall_status", "safe")
        fall_score = state.get("fall_score")
        self._bridge.push_fall(fall_status, fall_score)

        # 推送情绪数据
        emotion = state.get("emotion")
        if emotion:
            self._bridge.push_emotion(
                emotion.get("label", "neutral"),
                emotion.get("scores"),
            )

        # 推送步态指标
        self._bridge.push_metrics({
            "person_count": state.get("person_count", 0),
            "fps": state.get("fps", 0),
        })

    def _patch_diagnostic_calls(self, rehab_main):
        """补丁诊断窗口调用

        Args:
            rehab_main: rehab_monitor.main 模块
        """
        # 拦截诊断窗口的渲染调用
        # 这些通常在主循环中被调用, 我们需要捕获它们的输出
        pass

    def get_frame_count(self):
        """获取帧计数

        Returns:
            int: 已捕获的帧数
        """
        with self._frame_lock:
            return self._frame_count
