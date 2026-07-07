"""诊断窗口捕获 — 替代 cv2.imshow, 将渲染输出转为 QImage"""
import threading
from functools import wraps

from rehab_monitor.logging_setup import get_logger

logger = get_logger("diagnostic_capturer")


class DiagnosticCapturer:
    """诊断窗口捕获适配器 — monkey-patch 4个诊断渲染器

    拦截 AngleMonitor, TrajectoryMonitor, SkeletonViewer, GaitMetricsViewer
    的 render 方法, 将输出的帧通过 bridge 推送给 GUI
    """

    def __init__(self, bridge):
        """初始化诊断窗口捕获器

        Args:
            bridge: ThreadBridge 实例
        """
        self._bridge = bridge
        self._original_methods = {}

        logger.info("诊断窗口捕获器已初始化")

    def inject(self):
        """注入所有诊断模块"""
        try:
            # 角度监测
            self._patch_module(
                "rehab_monitor.angle_monitor",
                "AngleMonitor",
                "render",
                self._bridge.push_diag_angle,
            )

            # 轨迹监测
            self._patch_module(
                "rehab_monitor.trajectory_monitor",
                "TrajectoryMonitor",
                "render",
                self._bridge.push_diag_trajectory,
            )

            # 骨骼预览
            self._patch_module(
                "rehab_monitor.skeleton_viewer",
                "SkeletonViewer",
                "render",
                self._bridge.push_diag_skeleton,
            )

            # 步态指标
            self._patch_module(
                "rehab_monitor.gait_metrics_viewer",
                "GaitMetricsViewer",
                "render",
                self._bridge.push_diag_gait,
            )

            logger.info("诊断窗口捕获注入成功 (4个模块)")
        except Exception as e:
            logger.error(f"诊断窗口捕获注入失败: {e}")

    def _patch_module(self, module_path, class_name, method_name, push_func):
        """补丁单个模块的 render 方法

        Args:
            module_path: 模块路径
            class_name: 类名
            method_name: 方法名
            push_func: 推送函数
        """
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        original = getattr(cls, method_name)

        # 保存原始方法
        self._original_methods[(module_path, method_name)] = original

        @wraps(original)
        def patched_render(self_obj, *args, **kwargs):
            """替换的 render 方法"""
            # 调用原始渲染
            result = original(self_obj, *args, **kwargs)

            # 捕获输出帧 (render 返回修改后的帧)
            if result is not None:
                push_func(result)

            return result

        setattr(cls, method_name, patched_render)
        logger.debug(f"已补丁 {module_path}.{class_name}.{method_name}")
