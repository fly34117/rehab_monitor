#!/bin/bash
# ============================================================
# 一键从 Gitee 拉取脚本
# 用法: ./pull_gitee.sh
# ============================================================
set -e

GITEE_REMOTE="gitee"
GITEE_BRANCH="main"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  📥 一键拉取 Gitee${NC}"
echo -e "${GREEN}============================================${NC}"

# --- 0. 检查是否在 git 仓库中 ---
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}✗ 当前目录不是 Git 仓库${NC}"
    exit 1
fi

# --- 1. 检查远程是否存在 ---
echo -e "\n${YELLOW}[1/3] 检查 Gitee 远程...${NC}"
if ! git remote get-url "$GITEE_REMOTE" > /dev/null 2>&1; then
    echo -e "${RED}✗ 远程 '$GITEE_REMOTE' 不存在${NC}"
    echo -e "  请先添加: git remote add $GITEE_REMOTE https://gitee.com/fiy-placid/rehab_monitor.git"
    exit 1
fi
echo -e "${GREEN}✓ 远程: $(git remote get-url $GITEE_REMOTE)${NC}"

# --- 2. 检查本地是否有未提交的变更 ---
echo -e "\n${YELLOW}[2/3] 检查本地状态...${NC}"
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}⚠ 本地有未提交的变更:${NC}"
    git status --short
    echo -e "\n${YELLOW}建议先提交或暂存本地变更，避免冲突。${NC}"
    read -p "是否继续拉取？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}已取消${NC}"
        exit 0
    fi
fi

# --- 3. 拉取 ---
echo -e "\n${YELLOW}[3/3] 从 Gitee 拉取...${NC}"

# 先 fetch 看差异
echo -e "  Fetching..."
git fetch "$GITEE_REMOTE" "$GITEE_BRANCH"

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "$GITEE_REMOTE/$GITEE_BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
    echo -e "${GREEN}✓ 已是最新，无需拉取${NC}"
    exit 0
fi

echo -e "  本地: ${LOCAL:0:7}"
echo -e "  远程: ${REMOTE:0:7}"

# 显示远程新增的提交
echo -e "\n${YELLOW}  远程新增提交:${NC}"
git log --oneline --color "$LOCAL..$REMOTE" 2>/dev/null

echo

# 执行 pull
if git pull "$GITEE_REMOTE" "$GITEE_BRANCH"; then
    echo -e "\n${GREEN}============================================${NC}"
    echo -e "${GREEN}  ✓ 拉取成功！${NC}"
    echo -e "${GREEN}============================================${NC}"
else
    echo -e "\n${RED}✗ 拉取失败，可能有冲突，请手动解决${NC}"
    exit 1
fi
