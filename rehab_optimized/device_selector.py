"""
硬件自动检测与设备优先级选择
=============================
自动探测 NPU / GPU / CPU 可用性, 按优先级回退
"""
import os
import grp
import logging

logger = logging.getLogger("rehab.device")


class DeviceSelector:
    """Intel 异构设备检测器

    检测顺序: NPU → GPU → CPU
    每个设备需要:
      - NPU: /dev/accel/accel0 + render 组权限 + OpenVINO NPU 插件
      - GPU: /dev/dri/renderD128 + OpenVINO GPU 插件
      - CPU: 始终可用
    """

    def __init__(self, priority=None):
        """
        Args:
            priority: 设备优先级列表, 默认 ['npu', 'gpu', 'cpu']
        """
        self.priority = priority or ["npu", "gpu", "cpu"]
        self._ov_devices = None   # 延迟加载
        self._available = self._detect_all()

    # ------------------------------------------------------------------
    # 硬件检测
    # ------------------------------------------------------------------

    @property
    def ov_devices(self):
        """获取 OpenVINO 可用设备列表 (延迟加载)"""
        if self._ov_devices is None:
            try:
                import openvino as ov
                self._ov_devices = ov.Core().available_devices
            except Exception as e:
                logger.warning("OpenVINO 初始化失败: %s", e)
                self._ov_devices = []
        return self._ov_devices

    def _check_npu(self):
        """检查 NPU 可用性"""
        # 1. 设备节点存在
        if not os.path.exists("/dev/accel/accel0"):
            logger.debug("NPU: /dev/accel/accel0 不存在")
            return False

        # 2. render 组权限
        try:
            user_groups = {g.gr_name for g in grp.getgrall()
                          if os.getuid() in g.gr_mem}
            if "render" not in user_groups:
                logger.warning("NPU: 用户不在 render 组")
                return False
        except Exception:
            pass

        # 3. OpenVINO 插件
        if "NPU" not in self.ov_devices:
            logger.debug("NPU: OpenVINO 未识别")
            return False

        # 4. 设备可读写
        if not os.access("/dev/accel/accel0", os.RDWR):
            logger.warning("NPU: 无读写权限")
            return False

        return True

    def _check_gpu(self):
        """检查 GPU 可用性"""
        if "GPU" not in self.ov_devices:
            logger.debug("GPU: OpenVINO 未识别")
            return False
        return True

    def _check_cpu(self):
        """CPU 始终可用"""
        return True

    def _detect_all(self):
        """检测所有设备"""
        available = {}
        for dev in self.priority:
            check = getattr(self, f"_check_{dev}", None)
            if check and check():
                available[dev] = True
            else:
                available[dev] = False
        return available

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def get_best_device(self):
        """按优先级返回最佳可用设备"""
        for dev in self.priority:
            if self._available.get(dev, False):
                return dev
        return "cpu"  # 最后回退

    def get_available_devices(self):
        """返回所有可用设备"""
        return [dev for dev in self.priority if self._available.get(dev, False)]

    def is_available(self, device):
        """检查指定设备是否可用"""
        return self._available.get(device, False)

    def report(self):
        """打印设备状态报告"""
        lines = ["硬件设备检测:"]
        status_icon = {True: "[OK]", False: "[--]"}
        for dev in self.priority:
            ok = self._available.get(dev, False)
            icon = status_icon[ok]
            detail = self._device_detail(dev) if ok else "不可用"
            lines.append(f"  {icon} {dev.upper():4s}  {detail}")
        best = self.get_best_device()
        lines.append(f"  → 选择: {best.upper()}")
        return "\n".join(lines)

    def _device_detail(self, device):
        """设备详细信息"""
        details = {
            "npu": "Intel AI Boost",
            "gpu": "Intel Arc iGPU",
            "cpu": f"Intel Core Ultra ({os.cpu_count()} 线程)",
        }
        return details.get(device, "")


# ---- 模块级便捷函数 ----
_global_selector = None


def get_selector() -> DeviceSelector:
    global _global_selector
    if _global_selector is None:
        _global_selector = DeviceSelector()
    return _global_selector
