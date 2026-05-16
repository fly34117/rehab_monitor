"""统一日志系统 — 控制台 + 文件双输出"""
import logging
import os
import sys
from datetime import datetime

_logger = None
_LOG_DIR = None


def setup_logging(level=logging.INFO, log_dir=None):
    """初始化日志系统，返回 root logger。重复调用安全。"""
    global _logger, _LOG_DIR

    if _logger is not None:
        return _logger

    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    _LOG_DIR = log_dir
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"rehab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # 控制台 handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)

    # 文件 handler (DEBUG 级别，事后排查用)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root = logging.getLogger("rehab")
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    _logger = root
    _logger.info(f"日志文件: {log_file}")
    return _logger


def get_logger(name=None):
    """获取子 logger。如未初始化则自动初始化。"""
    global _logger
    if _logger is None:
        setup_logging()
    full = "rehab"
    if name:
        full += f".{name}"
    return logging.getLogger(full)


def get_log_dir():
    global _LOG_DIR
    if _LOG_DIR is None:
        _LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(_LOG_DIR, exist_ok=True)
    return _LOG_DIR
