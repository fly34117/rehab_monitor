#!/bin/bash
# ============================================================
# 一键推送 Gitee 脚本
# 用法: ./push_gitee.sh [commit message]
# ============================================================
set -e

GITEE_REMOTE="gitee"
GITEE_LIMIT_MB=100

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  🚀 一键推送 Gitee${NC}"
echo -e "${GREEN}============================================${NC}"

# --- 0. 检查是否在 git 仓库中 ---
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}✗ 当前目录不是 Git 仓库${NC}"
    exit 1
fi

# --- 1. 检查 git 历史中是否有大文件（>100MB）---
echo -e "\n${YELLOW}[1/5] 检查 git 历史大文件 (>${GITEE_LIMIT_MB}MB)...${NC}"

LARGE_FILES=$(git rev-list --objects --all | \
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
    awk '/^blob/ && $3 > '"${GITEE_LIMIT_MB}"'*1024*1024 {print $3, $4}' | \
    sort -rn)

if [ -n "$LARGE_FILES" ]; then
    echo -e "${RED}✗ 仓库历史中存在超过 ${GITEE_LIMIT_MB}MB 的文件:${NC}"
    echo "$LARGE_FILES" | while read size path; do
        size_mb=$(echo "scale=1; $size / 1024 / 1024" | bc 2>/dev/null || echo "$size")
        echo -e "  ${RED}• $path (${size_mb}MB)${NC}"
    done
    echo -e "\n${RED}请先移除大文件再推送！${NC}"
    echo -e "  参考: git filter-branch --index-filter 'git rm --cached --ignore-unmatch <file>' -- --all"
    exit 1
fi
echo -e "${GREEN}✓ git 历史无超限文件${NC}"

# --- 2. 检查工作区中是否有大文件（排除 .gitignore 已屏蔽的）---
echo -e "\n${YELLOW}[2/5] 检查工作区大文件 (>${GITEE_LIMIT_MB}MB)...${NC}"

WS_LARGE_FILES=$(find . -type f -size +"${GITEE_LIMIT_MB}"M \
    -not -path './.git/*' \
    -not -path './.claude/*' \
    -printf '%s %p\n' 2>/dev/null | sort -rn)

WARN_FILES=""
if [ -n "$WS_LARGE_FILES" ]; then
    while read size path; do
        # 跳过已被 .gitignore 屏蔽的文件（git add -A 不会提交它们）
        if git check-ignore -q "$path" 2>/dev/null; then
            continue
        fi
        size_mb=$(echo "scale=1; $size / 1024 / 1024" | bc 2>/dev/null || echo "$size")
        WARN_FILES="${WARN_FILES}  ${RED}• $path (${size_mb}MB)${NC}\n"
    done <<< "$WS_LARGE_FILES"
fi

if [ -n "$WARN_FILES" ]; then
    echo -e "${RED}✗ 工作区存在超过 ${GITEE_LIMIT_MB}MB 且未被 .gitignore 屏蔽的文件:${NC}"
    echo -e "$WARN_FILES"
    echo -e "${RED}请先移除大文件或将它们加入 .gitignore！${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 工作区无超限文件（已排除 .gitignore 屏蔽的）${NC}"

# --- 3. 检查变更 ---
echo -e "\n${YELLOW}[3/5] 检查本地变更...${NC}"

if [ -z "$(git status --porcelain)" ] && [ -z "$(git log gitee/main..HEAD --oneline 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ 没有需要推送的内容${NC}"
    exit 0
fi

# 显示待提交的文件
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}  未暂存/未提交的变更:${NC}"
    git status --short
fi

# 显示待推送的提交
if [ -n "$(git log gitee/main..HEAD --oneline 2>/dev/null)" ]; then
    echo -e "${YELLOW}  待推送的提交:${NC}"
    git log gitee/main..HEAD --oneline --color 2>/dev/null
fi

# --- 3. 自动提交（如果有未提交的变更） ---
if [ -n "$(git status --porcelain)" ]; then
    echo -e "\n${YELLOW}[4/5] 提交变更...${NC}"
    git add -A

    if [ -n "$1" ]; then
        COMMIT_MSG="$1"
    else
        COMMIT_MSG="auto: $(date '+%Y-%m-%d %H:%M') 自动提交"
    fi

    git commit -m "$COMMIT_MSG"
    echo -e "${GREEN}✓ 已提交: $COMMIT_MSG${NC}"
else
    echo -e "\n${YELLOW}[4/5] 无需提交，跳过${NC}"
fi

# --- 4. 推送到 Gitee ---
echo -e "\n${YELLOW}[5/5] 推送到 Gitee...${NC}"

if git push "$GITEE_REMOTE" main 2>&1; then
    echo -e "\n${GREEN}============================================${NC}"
    echo -e "${GREEN}  ✓ 推送成功！${NC}"
    echo -e "${GREEN}  📦 https://gitee.com/fiy-placid/rehab_monitor${NC}"
    echo -e "${GREEN}============================================${NC}"
else
    echo -e "\n${RED}✗ 推送失败，请检查上方错误信息${NC}"
    exit 1
fi
