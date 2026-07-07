"""设备自动检测 — NPU/GPU/CPU"""
import os
import subprocess
import shutil


class DeviceDetector:
    """硬件设备自动检测器

    检测优先级: NPU > GPU > CPU
    """

    # NPU 设备路径
    NPU_DEVICE_PATH = "/dev/accel/accel0"
    # GPU 设备路径
    GPU_DEVICE_PATH = "/dev/dri/renderD128"

    def detect(self):
        """检测可用的推理设备

        Returns:
            str: "npu" / "gpu" / "cpu"
        """
        if self._has_npu():
            return "npu"
        if self._has_gpu():
            return "gpu"
        return "cpu"

    def _has_npu(self):
        """检查 NPU 是否可用

        Returns:
            bool: NPU 是否可用
        """
        # 检查设备文件
        if not os.path.exists(self.NPU_DEVICE_PATH):
            return False
        # 检查用户是否在 render 组
        try:
            groups = subprocess.check_output(
                ["groups"], text=True
            ).strip()
            if "render" not in groups.split():
                return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
        # 检查设备权限
        try:
            stat = os.stat(self.NPU_DEVICE_PATH)
            # render 组 GID 通常为 109
            import grp
            try:
                render_gid = grp.getgrnam("render").gr_gid
                if stat.st_gid == render_gid:
                    # 检查组权限是否有读写
                    return bool(stat.st_mode & 0o060)
            except KeyError:
                pass
            # 检查是否为 root:render 且权限为 660
            return bool(stat.st_mode & 0o660)
        except OSError:
            return False

    def _has_gpu(self):
        """检查 Intel iGPU 是否可用

        Returns:
            bool: GPU 是否可用
        """
        # 检查设备文件
        if not os.path.exists(self.GPU_DEVICE_PATH):
            return False
        # 尝试 vainfo 检查 VA-API 支持
        try:
            result = subprocess.run(
                ["vainfo"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # vainfo 退出码 0 或包含 "VA-API" 表示可用
            return result.returncode == 0 or "VA-API" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # vainfo 不存在，回退到检查设备文件
            return True

    def get_info(self):
        """获取设备详细信息

        Returns:
            dict: 设备信息字典
        """
        device = self.detect()
        info = {
            "device": device,
            "npu_available": self._has_npu(),
            "gpu_available": self._has_gpu(),
            "cpu_available": True,
        }
        # 尝试获取更详细的信息
        try:
            # 检查 OpenVINO 版本
            import openvino
            info["openvino_version"] = openvino.__version__
        except ImportError:
            info["openvino_version"] = "unknown"
        try:
            # 检查 vainfo 输出
            result = subprocess.run(
                ["vainfo"], capture_output=True, text=True, timeout=5
            )
            info["vainfo_output"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            info["vainfo_output"] = ""
        return info
