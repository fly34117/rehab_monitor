"""
硬件加速摄像头模块 — 隔离在 ubuntu/，不修改 rehab_monitor/
============================================================
自动选择最优解码后端:
  1. GStreamer VA-API 硬件解码 (iGPU 零拷贝解码 + resize)
  2. OpenCV V4L2 MJPEG      (摄像头硬件压缩, CPU 解压)
  3. OpenCV V4L2 YUYV       (原始数据, 纯 CPU)

特性:
  - 异步采集线程 (解码与处理解耦)
  - 丢帧保护 (队列满时丢弃旧帧)
  - 自动回退 (驱动不可用时降级)
  - 与 cv2.VideoCapture 接口兼容 (供 monkey-patch 使用)

使用方式:
  from ubuntu.hw_camera import create_hw_camera
  cap = create_hw_camera(0, width=640, height=480, fps=30)
  ret, frame = cap.read()
"""

import os
import sys
import time
import queue
import threading
import logging
import numpy as np

logger = logging.getLogger("rehab.hw_camera")

# OpenCV 属性常量 (cv2 可能还没导入时也能用)
CAP_PROP_FRAME_WIDTH  = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS          = 5
CAP_PROP_FOURCC       = 6
CAP_PROP_BUFFERSIZE   = 38
CAP_PROP_BACKEND      = 42

# ============================================================
# 检测函数 (带缓存, 避免重复调用外部命令)
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


def invalidate_hw_cache():
    """清除硬件检测缓存 (驱动更新后调用)"""
    global _vaapi_cache, _gst_cache
    _vaapi_cache = None
    _gst_cache = None


def _check_camera_mjpeg(device_id=0):
    """检测摄像头是否支持 MJPEG"""
    try:
        import cv2
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        actual = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()
        return actual == cv2.VideoWriter_fourcc(*'MJPG')
    except Exception:
        return False


def detect_best_backend(device_id=0):
    """自动检测并返回最优可用后端

    Returns: 'gstreamer_vaapi' | 'v4l2_mjpeg' | 'v4l2_yuyv' | 'opencv'
    """
    # 1. GStreamer VA-API (硬件解码最佳)
    if _check_vaapi_working() and _check_gst_vaapi():
        return "gstreamer_vaapi"

    # 2. V4L2 MJPEG (摄像头内压缩, CPU 解压)
    if _check_camera_mjpeg(device_id):
        return "v4l2_mjpeg"

    # 3. V4L2 YUYV (标准方式)
    try:
        import cv2
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        ok = cap.isOpened()
        cap.release()
        if ok:
            return "v4l2_yuyv"
    except Exception:
        pass

    # 4. OpenCV 默认后端
    return "opencv"


# ============================================================
# GStreamer VA-API 硬件解码后端
# ============================================================

class _GstVaapiCamera:
    """GStreamer VA-API 管道 — Arrow Lake iGPU 硬件解码

    管道: v4l2src → rawvideoparse → vaapipostproc → videoconvert → appsink
    优势: 解码/缩放/色彩转换全在 iGPU，CPU 几乎零负载
    """

    def __init__(self, device_id=0, width=640, height=480, fps=30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.is_opened = False

        self._frame_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._pipeline = None
        self._thread = None

        # 确保环境变量正确
        os.environ.setdefault("LIBVA_DRIVER_NAME", "iHD")
        os.environ.setdefault("GST_VAAPI_ALL_DRIVERS", "1")

        self._open()

    def _open(self):
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst, GLib
            Gst.init(None)
        except Exception as e:
            logger.warning("GStreamer Python 绑定不可用: %s", e)
            self.is_opened = False
            return

        # 管道: v4l2 采集 → VA-API 后处理 (解码/缩放/色彩) → BGR 输出 → appsink
        pipeline_str = (
            f"v4l2src device=/dev/video{self.device_id} ! "
            f"video/x-raw,width={self.width},height={self.height},"
            f"framerate={self.fps}/1,format=YUY2 ! "
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

            # 启动线程处理 GStreamer 消息总线
            self._bus = self._pipeline.get_bus()
            self._bus.add_signal_watch()
            self._bus.connect("message::error", self._on_error)
            self._bus.connect("message::warning", self._on_warning)

            self._pipeline.set_state(Gst.State.PLAYING)
            self.is_opened = True
            logger.info("摄像头已打开 (GStreamer VA-API, ID=%d, %dx%d@%d)",
                         self.device_id, self.width, self.height, self.fps)

        except Exception as e:
            logger.error("GStreamer 管道创建失败: %s", e)
            self.is_opened = False

    def _on_new_sample(self, sink):
        """appsink 回调 — 将帧放入队列 (丢帧策略)"""
        try:
            sample = sink.emit("pull-sample")
            if not sample:
                return 0  # GST_FLOW_OK = 0

            buf = sample.get_buffer()
            caps = sample.get_caps()
            struct = caps.get_structure(0)
            h = struct.get_value("height")
            w = struct.get_value("width")

            success, map_info = buf.map(1)  # GST_MAP_READ = 1
            if not success:
                return 0

            frame = np.ndarray(
                shape=(h, w, 3), dtype=np.uint8, buffer=map_info.data
            ).copy()  # 必须 copy

            buf.unmap(map_info)

            ts = time.time() * 1000
            try:
                self._frame_queue.put_nowait((True, frame))
            except queue.Full:
                # 丢弃旧帧，放入新帧
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait((True, frame))
                except queue.Empty:
                    pass

        except Exception as e:
            logger.error("GStreamer 帧回调异常: %s", e)

        return 0  # GST_FLOW_OK

    def _on_error(self, bus, msg):
        err, debug = msg.parse_error()
        logger.error("GStreamer 错误: %s (debug: %s)", err, debug)

    def _on_warning(self, bus, msg):
        err, debug = msg.parse_warning()
        logger.warning("GStreamer 警告: %s (debug: %s)", err, debug)

    def read(self):
        if not self.is_opened:
            return False, None
        try:
            ret, frame = self._frame_queue.get(timeout=2.0)
            return ret, frame
        except queue.Empty:
            return False, None

    def isOpened(self):
        return self.is_opened

    def release(self):
        self._stop_event.set()
        if self._pipeline:
            import gi
            from gi.repository import Gst
            self._pipeline.set_state(Gst.State.NULL)
        self.is_opened = False


# ============================================================
# OpenCV V4L2 优化后端 (MJPEG / YUYV)
# ============================================================

class _V4L2Camera:
    """OpenCV V4L2 摄像头 — MJPEG 优先, 异步采集

    即使没有 VA-API, 使用 MJPEG 也能显著降低 CPU 和 USB 带宽:
      YUYV 640x480@30fps → ~18 MB/s USB
      MJPEG 640x480@30fps →  ~3 MB/s USB (画质几乎无损)
    """

    def __init__(self, device_id=0, width=640, height=480, fps=30,
                 use_mjpeg=True):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.is_opened = False

        self._frame_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread = None

        self._open(use_mjpeg)

    def _open(self, use_mjpeg):
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            self.is_opened = False
            return

        # 设置参数
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 尝试 MJPEG
        fourcc_str = "YUYV"
        if use_mjpeg:
            self._cap.set(cv2.CAP_PROP_FOURCC,
                          cv2.VideoWriter_fourcc(*'MJPG'))
            actual = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            if actual == cv2.VideoWriter_fourcc(*'MJPG'):
                fourcc_str = "MJPG"
            else:
                # 回退 YUYV
                self._cap.set(cv2.CAP_PROP_FOURCC,
                              cv2.VideoWriter_fourcc(*'YUYV'))

        # 启动异步采集线程
        self.is_opened = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._thread.start()

        logger.info("摄像头已打开 (V4L2 %s 异步, ID=%d, %dx%d@%d)",
                     fourcc_str, self.device_id, self.width, self.height, self.fps)

    def _capture_loop(self):
        """后台采集线程 — 持续读取最新帧"""
        while not self._stop_event.is_set():
            if not self._cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("摄像头读取失败, 等待重连...")
                time.sleep(0.1)
                continue

            try:
                self._frame_queue.put_nowait((True, frame))
            except queue.Full:
                # 丢弃旧帧
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait((True, frame))
                except queue.Empty:
                    pass

    def read(self):
        if not self.is_opened:
            return False, None
        try:
            return self._frame_queue.get(timeout=2.0)
        except queue.Empty:
            return False, None

    def isOpened(self):
        return self.is_opened

    def release(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if hasattr(self, '_cap') and self._cap:
            self._cap.release()
        self.is_opened = False


# ============================================================
# 统一接口 — 与 cv2.VideoCapture 兼容
# ============================================================

class HWCamera:
    """硬件加速摄像头 — cv2.VideoCapture 兼容接口

    自动选择最优后端，提供与 OpenCV VideoCapture 完全相同的方法签名。
    设计用于 monkey-patch cv2.VideoCapture，让原 rehab_monitor 无需修改
    即可获得硬件加速。

    用法 (monkey-patch):
        import cv2
        _orig_cap = cv2.VideoCapture
        def _patched_cap(idx, *args, **kwargs):
            try:
                return HWCamera(idx)
            except Exception:
                return _orig_cap(idx, *args, **kwargs)
        cv2.VideoCapture = _patched_cap
    """

    def __init__(self, index=0, apiPreference=None, params=None):
        self._index = index
        self._width = 640
        self._height = 480
        self._fps = 30
        self._params = {}  # 缓存的 set() 参数

        # 保留 cv2 兼容属性
        self._backend = None
        self._impl = None

        # 检测最优后端
        backend = detect_best_backend(index)

        if backend == "gstreamer_vaapi":
            self._try_gstreamer(index)
        elif backend in ("v4l2_mjpeg", "v4l2_yuyv"):
            self._try_v4l2(index, use_mjpeg=(backend == "v4l2_mjpeg"))
        else:
            self._try_opencv(index)

    def _try_gstreamer(self, index):
        try:
            self._impl = _GstVaapiCamera(index, self._width, self._height, self._fps)
            if self._impl.is_opened:
                self._backend = "gstreamer_vaapi"
                return
        except Exception as e:
            logger.warning("GStreamer 失败: %s", e)

        # 回退
        self._try_v4l2(index, use_mjpeg=True)

    def _try_v4l2(self, index, use_mjpeg=True):
        try:
            self._impl = _V4L2Camera(index, self._width, self._height,
                                     self._fps, use_mjpeg=use_mjpeg)
            if self._impl.is_opened:
                self._backend = "v4l2"
                return
        except Exception as e:
            logger.warning("V4L2 失败: %s", e)

        self._try_opencv(index)

    def _try_opencv(self, index):
        import cv2
        self._impl = cv2.VideoCapture(index)
        self._impl.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._impl.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._impl.set(cv2.CAP_PROP_FPS, self._fps)
        self._backend = "opencv"
        logger.info("摄像头: OpenCV 默认后端 (ID=%d)", index)

    @property
    def backend_name(self):
        return self._backend

    # ---- cv2.VideoCapture 兼容接口 ----

    def isOpened(self):
        if self._impl is None:
            return False
        return self._impl.isOpened()

    def read(self):
        """返回 (ret, frame) 与 cv2.VideoCapture.read() 完全一致"""
        if self._impl is None:
            return False, None

        if self._backend == "opencv":
            return self._impl.read()

        return self._impl.read()  # (ret, frame)

    def release(self):
        if self._impl:
            self._impl.release()
        self._impl = None

    def get(self, prop_id):
        """兼容 cv2.VideoCapture.get()"""
        if self._backend == "opencv" and self._impl:
            return self._impl.get(prop_id)

        # 映射常用属性
        mapping = {
            CAP_PROP_FRAME_WIDTH:  self._width,
            CAP_PROP_FRAME_HEIGHT: self._height,
            CAP_PROP_FPS:          self._fps,
            CAP_PROP_BUFFERSIZE:   1,
            CAP_PROP_BACKEND:      0x800,  # V4L2 backend code
        }
        return mapping.get(prop_id, -1)

    def set(self, prop_id, value):
        """兼容 cv2.VideoCapture.set()"""
        if self._backend == "opencv" and self._impl:
            return self._impl.set(prop_id, value)

        # 缓存参数 (管道已在构造时创建, 不支持运行时修改)
        if prop_id == CAP_PROP_FRAME_WIDTH:
            self._width = int(value)
        elif prop_id == CAP_PROP_FRAME_HEIGHT:
            self._height = int(value)
        elif prop_id == CAP_PROP_FPS:
            self._fps = int(value)
        self._params[prop_id] = value
        return True

    def getBackendName(self):
        """返回实际使用的后端名称"""
        if self._backend == "opencv" and self._impl:
            return self._impl.getBackendName()
        return self._backend or "unknown"

    def __del__(self):
        self.release()


# ============================================================
# 工厂函数
# ============================================================

def create_hw_camera(device_id=0, width=640, height=480, fps=30,
                     force_backend=None):
    """创建硬件加速摄像头 — 推荐入口

    Args:
        device_id: 摄像头 ID
        width, height: 分辨率
        fps: 帧率
        force_backend: None=自动, 'gstreamer_vaapi', 'v4l2_mjpeg', 'v4l2_yuyv', 'opencv'

    Returns: HWCamera 实例 (cv2.VideoCapture 兼容接口)
    """
    if force_backend:
        # 强制指定后端 (跳过自动检测)
        cap = HWCamera.__new__(HWCamera)
        cap._index = device_id
        cap._width = width
        cap._height = height
        cap._fps = fps
        cap._params = {}

        if force_backend == "gstreamer_vaapi":
            cap._try_gstreamer(device_id)
        elif force_backend in ("v4l2_mjpeg", "v4l2_yuyv"):
            cap._try_v4l2(device_id, use_mjpeg=(force_backend == "v4l2_mjpeg"))
        else:
            cap._try_opencv(device_id)

        return cap

    return HWCamera(device_id)


# ============================================================
# Monkey-patch 函数 (供 nmu_main.py 使用)
# ============================================================

def monkey_patch_cv2_videocapture():
    """替换 cv2.VideoCapture，注入硬件加速摄像头

    调用此函数后, 所有 cv2.VideoCapture(n) 调用
    都会自动使用硬件加速后端。

    用法 (在 nmu_main.py 中):
        from ubuntu.hw_camera import monkey_patch_cv2_videocapture
        monkey_patch_cv2_videocapture()
        # 之后 rehab_monitor.main.main() 中的 cv2.VideoCapture 自动使用硬件加速
    """
    import cv2
    _orig = cv2.VideoCapture

    def _patched(index=0, apiPreference=None, params=None):
        try:
            cam = HWCamera(index)
            if cam.isOpened():
                logger.info("HWCamera: 已替换 cv2.VideoCapture (后端=%s)", cam.backend_name)
                return cam
        except Exception as e:
            logger.warning("HWCamera 创建失败 (%s), 回退原 cv2.VideoCapture", e)

        return _orig(index, apiPreference, **(params or {}))

    cv2.VideoCapture = _patched
    logger.info("HWCamera: cv2.VideoCapture 已 monkey-patch")
    return _orig
