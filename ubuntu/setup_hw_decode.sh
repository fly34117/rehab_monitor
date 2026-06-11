#!/bin/bash
# ============================================================
# 硬件解码环境安装脚本 — Intel Arrow Lake + Ubuntu 22.04
# ============================================================
# 问题: 系统自带的 intel-media-va-driver 22.3.1 不支持 Arrow Lake
# 解决: 安装 Intel 官方最新 GPGPU/媒体驱动
#
# 用法:
#   sudo bash setup_hw_decode.sh          安装驱动 (需要 root)
#   bash setup_hw_decode.sh --check       仅检查当前状态
#   bash setup_hw_decode.sh --dry-run     预览会做什么
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
fail() { echo -e "${RED}[XX]${NC} $1"; }
info() { echo -e "${CYAN}[--]${NC} $1"; }

# ============================================================
# 检测子命令
# ============================================================
do_check() {
    echo "============================================"
    echo "  硬件解码环境检测"
    echo "============================================"
    echo ""

    # 1. GPU 识别
    echo "--- GPU ---"
    if lspci | grep -i vga | grep -qi intel; then
        lspci | grep -i vga | grep -i intel | while read -r line; do
            ok "Intel GPU: $line"
        done
    else
        fail "未检测到 Intel GPU"
    fi

    # 2. 内核
    echo "--- 内核 ---"
    KVER=$(uname -r)
    KMAJ=$(echo "$KVER" | cut -d. -f1)
    if [ "$KMAJ" -ge 6 ]; then
        ok "内核 $KVER (≥6.x, Arrow Lake 需要)"
    else
        warn "内核 $KVER (建议 ≥6.8, Arrow Lake 可能需要升级)"
    fi

    # 3. VA-API 驱动
    echo "--- VA-API ---"
    VA_VER=$(dpkg -l intel-media-va-driver 2>/dev/null | awk '/^ii/{print $3}' | cut -d- -f1 || echo "未安装")
    echo "  intel-media-va-driver: $VA_VER"
    if [ -n "$VA_VER" ] && dpkg --compare-versions "$VA_VER" ge "24.4.0"; then
        ok "驱动版本满足 Arrow Lake 需求"
    elif [ -n "$VA_VER" ] && dpkg --compare-versions "$VA_VER" ge "23.1.0"; then
        warn "驱动版本较旧, Arrow Lake 可能部分支持, 建议升级"
    else
        fail "驱动版本太旧 (需要 ≥24.4.0 才能支持 Arrow Lake)"
    fi

    # 4. vainfo
    echo "--- vainfo ---"
    if command -v vainfo &>/dev/null; then
        if vainfo 2>&1 | grep -q "driver version"; then
            ok "VA-API 工作正常:"
            vainfo 2>&1 | grep -E "driver|Profile" | sed 's/^/  /'
        else
            fail "VA-API 初始化失败 (驱动不匹配 GPU)"
            vainfo 2>&1 | tail -5 | sed 's/^/  /'
        fi
    else
        warn "vainfo 未安装 (apt install vainfo)"
    fi

    # 5. GStreamer VA-API
    echo "--- GStreamer VA-API ---"
    if gst-inspect-1.0 vaapi 2>/dev/null | grep -q "features"; then
        FEATURES=$(gst-inspect-1.0 vaapi 2>/dev/null | grep -oP '\d+ features' || echo "0 features")
        if [ "$FEATURES" != "0 features" ]; then
            ok "GStreamer VA-API: $FEATURES"
        else
            fail "GStreamer VA-API: 0 features (驱动不支持 GPU)"
        fi
    else
        warn "GStreamer VA-API 插件未安装"
    fi

    # 6. 摄像头格式
    echo "--- 摄像头 ---"
    for dev in /dev/video*; do
        if [ -c "$dev" ]; then
            echo "  $dev:"
            if command -v v4l2-ctl &>/dev/null; then
                v4l2-ctl -d "$dev" --list-formats 2>/dev/null | grep -E "MJPG|YUYV" | sed 's/^/    /' || echo "    (无可用格式)"
            fi
        fi
    done
    if ! command -v v4l2-ctl &>/dev/null; then
        warn "v4l2-ctl 未安装 (apt install v4l-utils)"
    fi

    # 7. 渲染节点权限
    echo "--- 权限 ---"
    if [ -c /dev/dri/renderD128 ]; then
        RPERM=$(stat -c "%a %G" /dev/dri/renderD128)
        echo "  /dev/dri/renderD128: $RPERM"
        if groups "$USER" | grep -q render; then
            ok "用户 $USER 在 render 组"
        else
            warn "用户 $USER 不在 render 组 (sudo usermod -a -G render $USER)"
        fi
    fi
    if [ -c /dev/accel/accel0 ]; then
        echo "  /dev/accel/accel0: $(stat -c "%a %G" /dev/accel/accel0)"
    fi

    # 8. 环境变量
    echo "--- 当前会话 ---"
    echo "  LIBVA_DRIVER_NAME=${LIBVA_DRIVER_NAME:-(未设置)}"
    echo "  LIBVA_DRIVERS_PATH=${LIBVA_DRIVERS_PATH:-(未设置)}"
    echo "  GST_VAAPI_ALL_DRIVERS=${GST_VAAPI_ALL_DRIVERS:-(未设置)}"
}

# ============================================================
# 安装
# ============================================================
do_install() {
    echo "============================================"
    echo "  安装 Intel 最新媒体驱动"
    echo "============================================"
    echo ""

    # ---- 0. 前置检查 ----
    if [ "$(id -u)" -ne 0 ]; then
        fail "需要 root 权限: sudo bash setup_hw_decode.sh"
        exit 1
    fi

    # ---- 0. 添加 Intel GPGPU/Compute 仓库 (提供最新 intel-media 驱动) ----
    echo "添加 Intel 官方仓库..."
    INTEL_REPO="/etc/apt/sources.list.d/intel-gpu.list"

    if [ ! -f "$INTEL_REPO" ]; then
        # Intel GPU driver repository for Ubuntu 22.04
        # https://dgpu-docs.intel.com/driver/installation.html
        wget -qO - https://repositories.intel.com/gpu/intel-graphics.key 2>/dev/null | \
            gpg --dearmor --yes -o /usr/share/keyrings/intel-graphics.gpg 2>/dev/null

        cat > "$INTEL_REPO" <<'EOF'
deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client
EOF
        ok "Intel GPU 仓库已添加"
    else
        info "Intel GPU 仓库已存在"
    fi

    # ---- 1. 更新包列表 ----
    echo ""
    echo "更新包列表..."
    apt-get update -qq 2>&1 | tail -1

    # ---- 2. 升级关键包 ----
    echo ""
    echo "升级 Intel 媒体驱动包..."

    PACKAGES=(
        intel-media-va-driver
        intel-media-va-driver-non-free
        libva2
        libva-drm2
        libva-x11-2
        libmfx1
        libigfxcmrt7
    )

    for pkg in "${PACKAGES[@]}"; do
        if apt-cache show "$pkg" &>/dev/null; then
            CURRENT=$(dpkg -l "$pkg" 2>/dev/null | awk '/^ii/{print $3}' || echo "未安装")
            echo "  $pkg: $CURRENT → 安装最新..."
            apt-get install -y "$pkg" -qq 2>&1 | tail -1
            NEW=$(dpkg -l "$pkg" 2>/dev/null | awk '/^ii/{print $3}' || echo "?")
            ok "  $pkg: $NEW"
        fi
    done

    # ---- 3. 安装/更新 GStreamer VA-API ----
    echo ""
    echo "确保 GStreamer VA-API 插件..."
    apt-get install -y gstreamer1.0-vaapi vainfo -qq 2>&1 | tail -1

    # ---- 4. 验证 ----
    echo ""
    echo "验证安装..."

    # 更新 ld cache
    ldconfig 2>/dev/null || true

    # 设置 LIBVA_DRIVER_NAME 为 iHD (新驱动)
    export LIBVA_DRIVER_NAME=iHD

    if vainfo 2>&1 | grep -q "driver version"; then
        ok "VA-API 启动成功:"
        vainfo 2>&1 | grep -E "driver|Profile" | head -15 | sed 's/^/  /'
    else
        warn "VA-API 仍不可用 (可能需要重启)"
        vainfo 2>&1 | tail -3 | sed 's/^/  /'
    fi

    # 检查 GStreamer
    FEATURES=$(gst-inspect-1.0 vaapi 2>/dev/null | grep -oP '\d+ features' || echo "0 features")
    if [ "$FEATURES" != "0 features" ]; then
        ok "GStreamer VA-API: $FEATURES"
    else
        warn "GStreamer VA-API 仍无可用功能 (可能需要重启或清理 GStreamer 缓存)"
        info "  试试: rm ~/.cache/gstreamer-1.0/registry.*"
    fi

    echo ""
    echo "============================================"
    echo "  安装完成"
    echo "============================================"
    echo ""
    echo "后续步骤:"
    echo "  1. 重启或注销: 使 GPU 驱动生效"
    echo "  2. 加入 render 组: sudo usermod -a -G render \$USER && newgrp render"
    echo "  3. 验证: bash setup_hw_decode.sh --check"
    echo "  4. 启动: bash ubuntu/run.sh --hw-decode"
}

# ============================================================
# 主入口
# ============================================================
case "${1:-}" in
    --check)
        do_check
        ;;
    --dry-run)
        echo "会执行的步骤:"
        echo "  1. 添加 Intel GPU 仓库: https://repositories.intel.com/gpu/ubuntu jammy client"
        echo "  2. 升级 intel-media-va-driver → 最新版 (>= 24.4.0)"
        echo "  3. 升级 libva2, libmfx1, libigfxcmrt7"
        echo "  4. 验证 vainfo + GStreamer VA-API"
        echo ""
        echo "当前状态:"
        do_check
        ;;
    *)
        do_check
        echo ""
        if [ "$(id -u)" -eq 0 ]; then
            read -rp "继续安装? [y/N] " REPLY
            if [ "$REPLY" = "y" ] || [ "$REPLY" = "Y" ]; then
                do_install
            fi
        else
            warn "需要 root 权限安装: sudo bash $0"
        fi
        ;;
esac
