#!/bin/bash
# ============================================================
# push-all.sh — 同时推送到 GitHub（主）和 Gitee（镜像）
# ============================================================
#
# 用法:
#   ./push-all.sh                    # 推送当前分支
#   ./push-all.sh main               # 推送指定分支
#   ./push-all.sh --force            # 强制推送（慎用）
#   ./push-all.sh --tags             # 推送所有标签
#
# 环境变量:
#   PUSH_BRANCH  — 默认推送的分支 (默认: 当前分支)

set -euo pipefail

# ===== 颜色 =====
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ===== 参数解析 =====
FORCE=""
TAGS=false
BRANCH="${PUSH_BRANCH:-}"

for arg in "$@"; do
    case "$arg" in
        --force|-f)  FORCE="--force" ;;
        --tags|-t)   TAGS=true ;;
        -*)          echo -e "${RED}未知参数: $arg${NC}" && exit 1 ;;
        *)           BRANCH="$arg" ;;
    esac
done

# 获取当前分支
if [ -z "$BRANCH" ]; then
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
fi

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  双远程推送${NC}"
echo -e "${CYAN}  分支: ${YELLOW}$BRANCH${NC}"
echo -e "${CYAN}  GitHub: github.com/fly34117/rehab_monitor.git${NC}"
echo -e "${CYAN}  Gitee:  gitee.com/fiy-placid/rehab_monitor.git${NC}"
echo -e "${CYAN}============================================================${NC}"

# ===== 1. 推送到 GitHub（主仓库） =====
echo -e "\n${GREEN}[1/2] 推送到 GitHub (origin)...${NC}"
git push origin "$BRANCH" $FORCE
echo -e "${GREEN}  GitHub 推送成功 ✓${NC}"

# ===== 2. 推送到 Gitee（镜像） =====
echo -e "\n${GREEN}[2/2] 推送到 Gitee (gitee)...${NC}"
git push gitee "$BRANCH" $FORCE
echo -e "${GREEN}  Gitee 推送成功 ✓${NC}"

# ===== 3. 可选：推送标签 =====
if $TAGS; then
    echo -e "\n${YELLOW}[Tags] 推送标签到两个远程...${NC}"
    git push origin --tags
    git push gitee --tags
    echo -e "${GREEN}  标签推送成功 ✓${NC}"
fi

echo -e "\n${CYAN}============================================================${NC}"
echo -e "${GREEN}  All done! ${NC}"
echo -e "${CYAN}============================================================${NC}"
