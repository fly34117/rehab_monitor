"""工具函数"""
import sys
import os
import logging


def setup_logging(name="rehab", level=logging.INFO):
    """统一日志配置"""
    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    log = logging.getLogger(name)
    log.setLevel(level)
    if not log.handlers:
        log.addHandler(handler)
    return log


def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path
