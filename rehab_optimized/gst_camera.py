"""
摄像头捕获 — 支持 GStreamer VA-API 硬件解码 + OpenCV V4L2 优化回退
=====================================================================
优先级:
  1. GStreamer VA-API 硬件解码 (iGPU 零拷贝)
  2. OpenCV V4L2 MJPEG 异步采集   (摄像头硬件压缩)
  3. OpenCV V4L2 YUYV 异步采集    (原始数据)
  4. OpenCV 默认后端

统一接口: CameraCapture.read() → (ret, frame, timestamp_ms)
"""
import logging
import threading
import time
import queue
import os
import numpy as np

logger = logging.getLogger("rehab.camera")


# ============================================================
# 基类
# ============================================================
class CameraCapture:
    """摄像头抽象基类"""

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.is_opened = False

    def read(self):
        """返回 (ret, frame, timestamp_ms)"""
        raise NotImplementedError

    def release(self):
        raise NotImplementedError


# ============================================================
# GStreamer VA-API 硬件解码 (需驱动 ≥ 24.4.0)
# ============================================================
class GstCamera(CameraCapture):
    """GStreamer VA-API 硬件解码 — Arrow Lake iGPU 加速

    管道: v4l2src → vaapipostproc → videoconvert → appsink
    优势: 解码/缩放/色彩转换全在 iGPU, CPU 几乎零负载
    要求: intel-media-va-driver >= 24.4.0 (Arrow Lake)
    """

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        super().__init__(camera_id, width, height, fps)
        self._frame_queue = queue.Queue(maxsize=2)
        self._pipeline = None
        self._bus = None

        # 确保 VA-API 环境变量
        os.environ.setdefault("LIBVA_DRIVER_NAME", "iHD")
        os.environ.setdefault("GST_VAAPI_ALL_DRIVERS", "1")

        self._open()

    def _open(self):
        """构建并启动 GStreamer 管道"""
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
            Gst.init(None)
        except Exception as e:
            logger.warning("GStreamer Python 绑定不可用: %s", e)
            self.is_opened = False
            return

        pipeline_str = (
            f"v4l2src device=/dev/video{self.camera_id} ! "
            f"video/x-raw,width={self.width},height={self.height},"
            f"framerate={self.target_fps}/1,format=YUY2 ! "
            f"vaapipostproc ! "
            f"video/x-raw,format=BGR ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink name=sink max-buffers=1 drop=true emit-signals=true"
        )

        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
            sink = self._pipeline.get_by_name("sink")
            if not sink:
                logger.error("GStreamer: 无法获取 appsink")
                self.is_opened = False
                return

            sink.set_property("emit-signals", True)
            sink.connect("new-sample", self._on_new_sample)

            # 消息总线
            self._bus = self._pipeline.get_bus()
            self._bus.add_signal_watch()
            self._bus.connect("message::error", self._on_error)

            self._pipeline.set_state(Gst.State.PLAYING)
            self.is_opened = True
            logger.info("摄像头已打开 (GStreamer VA-API, ID=%d, %dx%d@%d)",
                         self.camera_id, self.width, self.height, self.target_fps)

        except Exception as e:
            logger.error("GStreamer 管道创建失败: %s", e)
            self.is_opened = False

    def _on_new_sample(self, sink):
        """appsink 回调 — 丢帧策略 (队列满时丢弃旧帧)"""
        try:
            import gi
            from gi.repository import Gst

            sample = sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR

            buf = sample.get_buffer()
            caps = sample.get_caps()
            struct = caps.get_structure(0)
            h = struct.get_value("height")
            w = struct.get_value("width")

            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR

            frame = np.ndarray(
                shape=(h, w, 3), dtype=np.uint8, buffer=map_info.data
            ).copy()

            buf.unmap(map_info)

            ts = time.time() * 1000
            try:
                self._frame_queue.put_nowait((True, frame, ts))
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait((True, frame, ts))
                except queue.Empty:
                    pass

        except Exception as e:
            logger.error("GStreamer 帧回调异常: %s", e)

        return Gst.FlowReturn.OK

    def _on_error(self, bus, msg):
        import gi
        from gi.repository import Gst
        err, debug = msg.parse_error()
        logger.error("GStreamer 错误: %s", err)

    def read(self):
        if not self.is_opened:
            return False, None, 0
        try:
            return self._frame_queue.get(timeout=2.0)
        except queue.Empty:
            return False, None, 0

    def release(self):
        if self._pipeline:
            import gi
            from gi.repository import Gst
            self._pipeline.set_state(Gst.State.NULL)
        self.is_opened = False


# ============================================================
# OpenCV V4L2 优化捕获 (MJPEG 优先 + 异步采集)
# ============================================================
class OpenCVCamera(CameraCapture):
    """OpenCV V4L2 摄像头 — MJPEG 优先, 异步采集线程

    即使没有 VA-API, MJPEG 也能显著降低 CPU/USB 带宽:
      YUYV 640x480@30fps → ~18 MB/s USB
      MJPEG 640x480@30fps →  ~3 MB/s USB
    """

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        super().__init__(camera_id, width, height, fps)
        self._frame_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread = None
        self._cap = None
        self._fourcc_str = "YUYV"
        self._open()

    def _open(self):
        import cv2

        self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self.is_opened = False
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 尝试 MJPEG
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        actual = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        if actual == cv2.VideoWriter_fourcc(*'MJPG'):
            self._fourcc_str = "MJPG"
        else:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

        self.is_opened = True

        # 启动异步采集线程
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info("摄像头已打开 (V4L2 %s 异步, ID=%d, %dx%d@%d)",
                     self._fourcc_str, self.camera_id, self.width, self.height, self.target_fps)

    def _capture_loop(self):
        """后台采集线程 — 持续读取最新帧到队列"""
        while not self._stop_event.is_set():
            if not self._cap or not self._cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("摄像头读取失败, 等待重试...")
                time.sleep(0.1)
                continue

            ts = time.time() * 1000
            try:
                self._frame_queue.put_nowait((True, frame, ts))
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait((True, frame, ts))
                except queue.Empty:
                    pass

    def read(self):
        if not self.is_opened:
            return False, None, 0
        try:
            return self._frame_queue.get(timeout=2.0)
        except queue.Empty:
            return False, None, 0

    def release(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
        self.is_opened = False

    def __del__(self):
        self.release()


# ============================================================
# 检测辅助函数 (带缓存, 避免重复调用外部命令)
# ============================================================

_vaapi_cache = None
_gst_cache = None


def _check_vaapi_working():
    """检测 VA-API 是否可用 (结果缓存)"""
    global _vaapi_cache
    if _vaapi_cache is not None:
        return _vaapi_cache

    try:
        import subprocess
        logger.debug("检测 VA-API...")
        result = subprocess.run(
            ["vainfo"], capture_output=True, text=True, timeout=2,
            env={**os.environ, "LIBVA_DRIVER_NAME": "iHD"}
        )
        _vaapi_cache = "driver version" in (result.stdout + result.stderr).lower()
    except Exception:
        _vaapi_cache = False

    return _vaapi_cache


def _check_gst_vaapi():
    """检测 GStreamer VA-API 插件是否有可用元素 (结果缓存)"""
    global _gst_cache
    if _gst_cache is not None:
        return _gst_cache

    # 如果 VA-API 不可用，GStreamer 必然不可用，跳过
    if not _check_vaapi_working():
        _gst_cache = False
        return False

    try:
        import subprocess
        result = subprocess.run(
            ["gst-inspect-1.0", "vaapi"],
            capture_output=True, text=True, timeout=2
        )
        for line in (result.stdout + result.stderr).split("\n"):
            if "features" in line:
                _gst_cache = int(line.strip().split()[0]) > 0
                break
        else:
            _gst_cache = False
    except Exception:
        _gst_cache = False

    return _gst_cache


def invalidate_camera_cache():
    """清除硬件检测缓存 (驱动更新后调用)"""
    global _vaapi_cache, _gst_cache
    _vaapi_cache = None
    _gst_cache = None


# ============================================================
# 工厂函数 — 自动选择最优后端
# ============================================================

def create_camera(use_gstreamer=False, camera_id=0, width=640, height=480, fps=30):
    """创建摄像头实例 — 自动选择最优后端

    优先级:
      1. GStreamer VA-API (use_gstreamer=True 且硬件可用)
      2. OpenCV V4L2 异步采集 (MJPEG 优先)
      3. OpenCV 默认 (回退)
    """
    # GStreamer VA-API
    if use_gstreamer:
        if not _check_vaapi_working() or not _check_gst_vaapi():
            logger.warning("GStreamer VA-API 不可用 (驱动不支持当前 GPU), 回退 V4L2")
            logger.warning("  提示: sudo bash ubuntu/setup_hw_decode.sh 安装最新驱动")
        else:
            try:
                cam = GstCamera(camera_id, width, height, fps)
                if cam.is_opened:
                    return cam
                logger.warning("GStreamer 无法打开摄像头, 回退 V4L2")
            except Exception as e:
                logger.warning("GStreamer 失败 (%s), 回退 V4L2", e)

    # OpenCV V4L2 (MJPEG 优先, 异步采集)
    try:
        cam = OpenCVCamera(camera_id, width, height, fps)
        if cam.is_opened:
            return cam
    except Exception as e:
        logger.warning("OpenCV V4L2 失败 (%s), 回退默认", e)

    # 终极回退
    import cv2
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        logger.info("摄像头已打开 (OpenCV 默认后端, ID=%d)", camera_id)

    # 包装为 CameraCapture 接口
    class _OpenCVCameraWrapper(CameraCapture):
        def __init__(self, cap):
            super().__init__(camera_id, width, height, fps)
            self._cap = cap
            self.is_opened = cap.isOpened()

        def read(self):
            ret, frame = self._cap.read()
            return ret, frame, time.time() * 1000

        def release(self):
            self._cap.release()

    return _OpenCVCameraWrapper(cap)
