#!/bin/bash
# 安装桌面快捷方式 — 一键启动 Rehab Monitor GUI
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Rehab Monitor GUI 安装程序${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检测 Python 环境
CONDA_PREFIX="/home/ubuntu224/miniconda3/envs/yolov26"
PYTHON_BIN="$CONDA_PREFIX/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo -e "${RED}错误: 未找到 Python 环境 ($PYTHON_BIN)${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python 环境: $PYTHON_BIN"

# 检查 PyQt6
if ! $PYTHON_BIN -c "from PyQt6.QtWidgets import QApplication" 2>/dev/null; then
    echo -e "${YELLOW}!${NC} PyQt6 未安装, 正在安装..."
    $PYTHON_BIN -m pip install PyQt6
    echo -e "${GREEN}✓${NC} PyQt6 已安装"
else
    echo -e "${GREEN}✓${NC} PyQt6 已安装"
fi

# 检查系统依赖
if ! dpkg -l | grep -q libxcb-cursor0 2>/dev/null; then
    echo -e "${YELLOW}!${NC} 安装系统依赖 libxcb-cursor0..."
    sudo apt-get install -y libxcb-cursor0
    echo -e "${GREEN}✓${NC} 系统依赖已安装"
fi

# 创建配置目录
CONFIG_DIR="$HOME/.rehab_gui"
mkdir -p "$CONFIG_DIR"
echo -e "${GREEN}✓${NC} 配置目录: $CONFIG_DIR"

# 创建启动脚本
LAUNCHER="$PROJECT_ROOT/ubuntu/launch_gui.sh"
cat > "$LAUNCHER" << 'LAUNCHER_EOF'
#!/bin/bash
cd /home/ubuntu224/桌面/yolov26
source /home/ubuntu224/miniconda3/etc/profile.d/conda.sh
conda activate yolov26
python -m rehab_gui.main
LAUNCHER_EOF
chmod +x "$LAUNCHER"
echo -e "${GREEN}✓${NC} 启动脚本: $LAUNCHER"

# 安装桌面快捷方式
APP_DIR="$HOME/.local/share/applications"
mkdir -p "$APP_DIR"

cat > "$APP_DIR/rehab-gui.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Rehab Monitor
GenericName=康复监控系统
Comment=三模态智能康复监控系统 - YOLOv26-Pose + OpenVINO
Icon=video-display
Exec=$LAUNCHER
Path=$PROJECT_ROOT
Terminal=false
Categories=Utility;
EOF

chmod +x "$APP_DIR/rehab-gui.desktop"
echo -e "${GREEN}✓${NC} 桌面快捷方式已安装"

# 更新桌面数据库
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database "$APP_DIR" 2>/dev/null || true
fi

# 复制到桌面
cp "$APP_DIR/rehab-gui.desktop" "$HOME/桌面/RehabMonitor.desktop"
chmod +x "$HOME/桌面/RehabMonitor.desktop" 2>/dev/null || true
echo -e "${GREEN}✓${NC} 桌面快捷方式已创建"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  安装完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "双击桌面上的 ${YELLOW}Rehab Monitor${NC} 图标即可启动"
echo ""
echo "手动启动命令:"
echo "  ${YELLOW}$LAUNCHER${NC}"
echo "  或: ${YELLOW}cd $PROJECT_ROOT && $PYTHON_BIN -m rehab_gui.main${NC}"
echo ""
