"""工具函数 — cv2 Mat→QImage 转换等"""
import os
import cv2
import numpy as np
from PyQt6.QtGui import QImage

# ---- 中文字体加载 ----
_FONT_PATH = None
_FONT_CACHE = {}  # size -> ImageFont
_CJK_FONT_SEARCH = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def _get_cjk_font(size=20):
    """获取 CJK 字体（带缓存）"""
    global _FONT_PATH
    if _FONT_PATH is None:
        for p in _CJK_FONT_SEARCH:
            if os.path.exists(p):
                _FONT_PATH = p
                break
        if _FONT_PATH is None:
            _FONT_PATH = ""  # 标记已搜索过，避免重复查找

    if not _FONT_PATH:
        return None

    key = (_FONT_PATH, size)
    if key not in _FONT_CACHE:
        try:
            from PIL import ImageFont
            _FONT_CACHE[key] = ImageFont.truetype(_FONT_PATH, size)
        except Exception:
            _FONT_CACHE[key] = None
    return _FONT_CACHE[key]


def put_chinese_text(img, text, pos, color=(255, 255, 255), size=20):
    """在 OpenCV BGR 图像上绘制中文文本（使用 PIL）

    Args:
        img: np.ndarray BGR 格式图像
        text: 中文文本
        pos: (x, y) 左上角坐标
        color: BGR 颜色元组
        size: 字体大小

    Returns:
        np.ndarray: 带文字的图像（原地修改）
    """
    return put_chinese_texts(img, [(text, pos, color, size)])


def put_chinese_texts(img, texts):
    """批量在 OpenCV BGR 图像上绘制中文文本（一次 PIL 转换）

    Args:
        img: np.ndarray BGR 格式图像
        texts: list of (text, pos, color, size)
               text: 中文文本
               pos: (x, y) 左上角坐标
               color: BGR 颜色元组
               size: 字体大小

    Returns:
        np.ndarray: 带文字的图像（原地修改）
    """
    if not texts:
        return img

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        for text, pos, color, size in texts:
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return img

    # BGR → RGB → PIL
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    for text, pos, color, size in texts:
        font = _get_cjk_font(size)
        if font is not None:
            pil_color = (color[2], color[1], color[0])
            draw.text(pos, text, font=font, fill=pil_color)

    # PIL → BGR
    result = np.array(pil_img)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    img[:] = result
    return img


def mat_to_qimage(mat):
    """将 OpenCV Mat 转换为 PyQt QImage

    Args:
        mat: np.ndarray OpenCV BGR 格式图像 (H, W, 3) 或 (H, W)

    Returns:
        QImage: 转换后的 QImage 对象
    """
    if mat is None:
        return None

    if len(mat.shape) == 2:
        # 灰度图
        h, w = mat.shape
        bytes_per_line = w
        return QImage(mat.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

    if len(mat.shape) == 3:
        # BGR → RGB（用 numpy 索引避免 cv2.cvtColor 的额外内存分配）
        rgb = mat[:, :, ::-1].copy()  # BGR→RGB in-place slice + copy
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

    return None


def qimage_to_mat(qimage):
    """将 PyQt QImage 转换为 OpenCV Mat

    Args:
        qimage: QImage 对象

    Returns:
        np.ndarray: OpenCV BGR 格式图像
    """
    if qimage is None:
        return None

    # 转换为 RGB32 格式便于处理
    qimage = qimage.convertToFormat(QImage.Format.Format_RGB32)
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    # 获取字节数据
    buf = ptr.asstring(qimage.byteCount())
    # 转为 numpy 数组
    arr = np.frombuffer(buf, np.uint8).reshape((height, width, 4))
    # RGBA → BGR
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)


def scale_mat_to_fit(mat, max_width, max_height):
    """缩放 Mat 以适应目标尺寸, 保持宽高比

    Args:
        mat: np.ndarray OpenCV 图像
        max_width: 最大宽度
        max_height: 最大高度

    Returns:
        np.ndarray: 缩放后的图像
    """
    if mat is None:
        return None
    h, w = mat.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(mat, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return mat


def create_placeholder_frame(width, height, text="无画面"):
    """创建占位帧（支持中文）

    Args:
        width: 宽度
        height: 高度
        text: 显示文本

    Returns:
        np.ndarray: BGR 占位帧
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)  # 深色背景
    # 使用支持中文的渲染函数
    text_w_est = len(text) * 14  # 粗略估算文字宽度
    x = max(10, (width - text_w_est) // 2)
    y = height // 2 - 10
    put_chinese_text(frame, text, (x, y), color=(128, 128, 128), size=18)
    return frame
