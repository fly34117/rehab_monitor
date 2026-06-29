"""呼吸检测模块 — 基于胸腔ROI运动信号 + FFT频谱分析的呼吸频率检测

从 YOLO 姿态关键点中提取胸腔区域，通过帧间差分计算呼吸运动信号，
使用 FFT 频谱分析 (含次谐波修正) 或时域峰值检测来估算呼吸频率 (BPM)。

集成方式: 在主循环中按 'b' 键触发 (需先用 't' 锁定目标人脸),
进入呼吸检测子循环, 弹出实时波形窗口, 结束后显示结果 5 秒后销毁。
"""
import time
import cv2
import numpy as np
from collections import deque
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from .logging_setup import get_logger

logger = get_logger("breathing")

# COCO 17 关键点索引 (与 config.py 保持同步)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


class BreathingDetector:
    """呼吸频率检测器 — 从胸腔 ROI 的运动信号中提取呼吸频率

    信号处理流程:
        帧差绝对值 (absdiff) → 去趋势 (移动平均) → 高斯平滑
        → FFT + Hanning窗 + 带通滤波 [min_bpm, max_bpm]
        → 次谐波检查 + 自动半频修正 (整流效应补偿)
    """

    def __init__(self, sigma=1.5, min_bpm=5, max_bpm=100, auto_correct=True):
        self.sigma = sigma                # 高斯平滑核
        self.min_bpm = min_bpm            # 有效呼吸频率下限
        self.max_bpm = max_bpm            # 有效呼吸频率上限
        self.auto_correct = auto_correct  # 自动半频修正 (absdiff 整流)

    # ── 胸腔 ROI 提取 ──────────────────────────────────────────

    @staticmethod
    def get_chest_roi_from_keypoints(kpts, frame_shape):
        """从 COCO 17点姿态关键点计算胸腔 ROI

        使用肩膀 (5,6) 定位上胸部区域 — 与原版 BreathingDetector.auto_detect_roi
        参数完全一致: 宽=肩宽×0.6, 高=肩宽×0.55, 位置紧贴肩膀线下方。

        Args:
            kpts: (17, 2) 或 (17, 3) 关键点数组 [x, y, (conf)]
            frame_shape: 帧尺寸 (h, w)

        Returns:
            (x, y, w, h) 或 None (关键点不足时)
        """
        if kpts is None:
            return None

        h, w = frame_shape[:2]
        k = kpts[:, :2]  # 只取 x, y 坐标

        # 提取肩膀关键点 (需要双肩可见)
        shoulders = k[[LEFT_SHOULDER, RIGHT_SHOULDER]]
        shoulder_valid = shoulders[(shoulders[:, 0] > 0) & (shoulders[:, 1] > 0)]

        if len(shoulder_valid) < 2:
            return None  # 双肩必须都可见, 保证定位精度

        # 肩膀中点 (像素坐标) + 肩宽 (像素)
        shoulder_center = shoulders.mean(axis=0)
        shoulder_width = float(np.linalg.norm(shoulders[0] - shoulders[1]))

        # ROI 尺寸 — 与原版 auto_detect_roi 完全一致
        rw = max(int(shoulder_width * 0.6), 40)
        rh = max(int(shoulder_width * 0.55), 40)

        # ROI 位置: 水平居中于肩膀中点, 垂直紧贴肩膀线下方
        rx = int(shoulder_center[0] - rw / 2)
        ry = int(shoulder_center[1] + shoulder_width * 0.05)

        # 裁剪到帧边界内
        rx = max(0, min(rx, w - 1))
        ry = max(0, min(ry, h - 1))
        rw = int(min(rw, w - rx))
        rh = int(min(rh, h - ry))

        return (rx, ry, rw, rh)

    # ── BPM 计算 ───────────────────────────────────────────────

    def calculate_bpm(self, movements, fps, method='fft'):
        """从运动信号计算呼吸频率

        Args:
            movements: 帧间差分绝对值均值序列
            fps: 实际帧率
            method: 'fft' (频域) 或 'peak' (时域峰值计数)

        Returns:
            BPM 浮点数, 数据不足时返回 0.0
        """
        n = len(movements)
        if n < fps:  # 至少需要 1 秒数据
            logger.warning("呼吸数据不足: %d 帧 (%.1f 秒)",
                          n, n / fps if fps > 0 else 0)
            return 0.0

        movements = np.array(movements, dtype=np.float64)

        # 1. 去趋势 — 自适应窗口移动平均
        window = max(5, min(int(fps * 3), n // 2))
        if window % 2 == 0:
            window += 1
        trend = np.convolve(movements, np.ones(window) / window, mode='same')
        detrended = movements - trend

        # 2. 高斯平滑
        smoothed = gaussian_filter1d(detrended, sigma=self.sigma)

        # 3. 频率估算
        if method == 'fft':
            return self._fft_bpm(smoothed, fps)
        else:
            return self._peak_bpm(smoothed, fps)

    def _fft_bpm(self, signal, fps):
        """FFT 频谱分析 + 次谐波检查 + 自动半频修正

        帧差绝对值 (absdiff) 会引入整流效应, 将真实频率加倍。
        通过次谐波检查 (寻找低频峰值) 和自动半频修正来补偿。
        """
        n = len(signal)

        # 补零到 2 的幂
        n_fft = 2
        while n_fft < n:
            n_fft *= 2

        # Hanning 窗 + 实数 FFT
        fft_vals = np.abs(np.fft.rfft(signal * np.hanning(n), n=n_fft))
        freq = np.fft.rfftfreq(n_fft, 1.0 / fps)

        # 带通滤波 [min_bpm/60, max_bpm/60] Hz
        lo, hi = self.min_bpm / 60.0, self.max_bpm / 60.0
        mask = (freq >= lo) & (freq <= hi)
        if not np.any(mask):
            logger.debug("FFT: 有效范围内无信号 [%.0f-%.0f BPM]", self.min_bpm, self.max_bpm)
            return 0.0

        vf, va = freq[mask], fft_vals[mask]
        peak_i = np.argmax(va)
        peak_hz = vf[peak_i]
        peak_bpm = peak_hz * 60.0
        confidence = va[peak_i] / (np.mean(va) + 1e-10)

        # 次谐波检查: 在峰值频率 60% 以下寻找较低峰值
        # (absdiff 整流会将真实频率加倍, 故真实频率可能是峰值的一半)
        low_mask = (vf >= lo) & (vf <= peak_hz * 0.6)
        if np.any(low_mask):
            low_va = va[low_mask]
            low_vf = vf[low_mask]
            low_i = np.argmax(low_va)
            low_hz = low_vf[low_i]
            low_bpm = low_hz * 60.0
            ratio = low_va[low_i] / (va[peak_i] + 1e-10)
            octave_ratio = peak_hz / (low_hz + 1e-10)
            # 低频峰值足够强 (ratio > 0.15) 且频率比约 2x → 使用低频
            if ratio > 0.15 and 1.5 < octave_ratio < 2.5:
                logger.debug("FFT: 次谐波修正 %.1f → %.1f BPM (ratio=%.2f octave=%.1f)",
                           peak_bpm, low_bpm, ratio, octave_ratio)
                return low_bpm

        # 自动半频修正: 高置信度时峰值可能是真实频率的 2 倍
        if self.auto_correct and confidence > 2.5:
            half_bpm = peak_bpm / 2.0
            if half_bpm >= self.min_bpm:
                logger.debug("FFT: 半频修正 %.1f → %.1f BPM (conf=%.1f)",
                           peak_bpm, half_bpm, confidence)
                return half_bpm

        logger.debug("FFT: %.1f BPM @ %.3f Hz (conf=%.1f)", peak_bpm, peak_hz, confidence)
        return peak_bpm

    def _peak_bpm(self, signal, fps):
        """时域峰值计数 — 通过检测局部极值来估算呼吸频率

        出错时自动回退到 FFT 方法。
        """
        n = len(signal)
        duration_sec = n / fps

        # 最小峰值间距 (基于 max_bpm, *0.5 留有余量)
        min_dist = max(int(fps * 60 / self.max_bpm * 0.5), 3)

        # 峰值检测
        std = np.std(signal)
        peaks, _ = find_peaks(signal, distance=min_dist, prominence=std * 0.3)
        bpm = (len(peaks) / duration_sec) * 60

        logger.debug("Peak: %d 个峰值 → %.1f BPM (时长 %.1fs)",
                    len(peaks), bpm, duration_sec)

        # 超出有效范围时回退到 FFT
        if bpm < self.min_bpm or bpm > self.max_bpm:
            logger.debug("Peak 超出范围, 回退到 FFT...")
            return self._fft_bpm(signal, fps)

        return bpm


# ── 实时波形绘制 (供主循环调用) ──────────────────────────────────

def draw_breathing_waveform(buffer_data, frame_count, duration_sec,
                            fps, min_bpm=5, max_bpm=100):
    """绘制实时呼吸波形画布 (OpenCV BGR 图像)

    Args:
        buffer_data: deque 或 list — 运动信号缓冲
        frame_count: 已采集帧数
        duration_sec: 总检测时长 (秒)
        fps: 帧率
        min_bpm, max_bpm: 有效 BPM 范围 (仅显示用)

    Returns:
        (canvas_h, canvas_w, 3) BGR numpy 数组
    """
    canvas_w, canvas_h = 640, 400
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30

    data = list(buffer_data)
    n = len(data)

    # 数据不足时显示等待提示
    if n < 5:
        cv2.putText(canvas, "Collecting breath signal...", (140, canvas_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        _draw_waveform_info(canvas, frame_count, duration_sec, fps,
                           min_bpm, max_bpm, canvas_w, canvas_h)
        return canvas

    # 去均值 (去除直流分量)
    data_arr = np.array(data, dtype=np.float64)
    data_arr = data_arr - data_arr.mean()

    # 移动平均平滑
    window = min(11, n // 2)
    if window >= 3:
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window) / window
        data_arr = np.convolve(data_arr, kernel, mode='same')

    # 归一化到 [-1, 1]
    abs_max = np.abs(data_arr).max()
    if abs_max > 1e-8:
        data_arr = data_arr / abs_max

    # 绘图区域
    margin = 40
    plot_w = canvas_w - 2 * margin
    plot_h = canvas_h - 2 * margin - 50
    center_y = margin + plot_h // 2

    # 水平网格线
    for i in range(5):
        gy = int(margin + i * plot_h / 4)
        cv2.line(canvas, (margin, gy), (canvas_w - margin, gy), (50, 50, 50), 1)

    # 零线 (较亮)
    cv2.line(canvas, (margin, center_y), (canvas_w - margin, center_y),
             (100, 100, 100), 1)

    # 波形折线
    pts = []
    step = plot_w / max(n - 1, 1) if n > 1 else plot_w
    for i in range(n):
        px = int(margin + i * step)
        py = int(center_y - data_arr[i] * plot_h // 2)
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], (0, 230, 100), 2)

    # 峰值标记 (简易局部极值检测)
    if n > 10:
        for i in range(3, n - 3):
            local_win = data_arr[i - 3:i + 4]
            if data_arr[i] == local_win.max() and data_arr[i] > 0:
                px = int(margin + i * step)
                py = int(center_y - data_arr[i] * plot_h // 2)
                cv2.circle(canvas, (px, py), 5, (0, 180, 255), -1)  # 橙色 → 吸气
            elif data_arr[i] == local_win.min() and data_arr[i] < 0:
                px = int(margin + i * step)
                py = int(center_y - data_arr[i] * plot_h // 2)
                cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)    # 红色 → 呼气

    # 图例
    cv2.circle(canvas, (margin + 10, canvas_h - 14), 4, (0, 180, 255), -1)
    cv2.putText(canvas, "Inhale", (margin + 20, canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.circle(canvas, (margin + 80, canvas_h - 14), 4, (0, 0, 255), -1)
    cv2.putText(canvas, "Exhale", (margin + 90, canvas_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    _draw_waveform_info(canvas, frame_count, duration_sec, fps,
                       min_bpm, max_bpm, canvas_w, canvas_h)

    return canvas


def _draw_waveform_info(canvas, frame_count, duration_sec, fps,
                        min_bpm, max_bpm, canvas_w, canvas_h):
    """在波形画布上叠加状态信息条"""
    elapsed = frame_count / fps if fps > 0 else 0
    remaining = max(0, duration_sec - elapsed)

    # 顶栏: 进度
    cv2.putText(canvas,
                f"Recording: {elapsed:.0f}s / {duration_sec}s  "
                f"Remaining: {remaining:.0f}s  "
                f"Frames: {frame_count}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 底栏: 参数
    cv2.putText(canvas,
                f"BPM Range: {min_bpm}-{max_bpm}  |  "
                f"FPS: {fps:.0f}",
                (10, canvas_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)


def draw_breathing_result_frame(bpm, remaining_sec):
    """生成单帧结果画面 (非阻塞), 由主循环控制倒计时

    Args:
        bpm: 检测到的呼吸频率
        remaining_sec: 窗口关闭剩余秒数 (如 5,4,3,2,1)

    Returns:
        (400, 640, 3) BGR numpy 数组
    """
    canvas = np.ones((400, 640, 3), dtype=np.uint8) * 30

    # 标题
    cv2.putText(canvas, "Breathing Detection Complete", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # BPM 数值 (正常范围 8-30 用绿色, 异常用黄色)
    color = (0, 255, 100) if 8 <= bpm <= 30 else (0, 200, 255)
    status_text = "Normal" if 8 <= bpm <= 30 else "Out of range"
    cv2.putText(canvas, f"BPM: {bpm:.1f}", (200, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(canvas, f"Status: {status_text}", (200, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # 倒计时
    cv2.putText(canvas, f"Window closes in {int(remaining_sec)}s...", (200, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
    cv2.putText(canvas, f"Closing in {int(remaining_sec)}s...", (230, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 120, 255), 2)

    return canvas


def draw_breathing_result(bpm, window_name="Breathing Waveform"):
    """在波形窗口中显示检测结果, 5 秒后自动销毁 (阻塞式, 供独立脚本使用)

    Args:
        bpm: 检测到的呼吸频率
        window_name: OpenCV 窗口名
    """
    for i in range(5, 0, -1):
        canvas = draw_breathing_result_frame(bpm, i)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(1000)

    cv2.destroyWindow(window_name)
    logger.info("呼吸检测结果窗口已销毁 (BPM=%.1f)", bpm)
