# ============================================================
# push-all.ps1 — 同时推送到 GitHub（主）和 Gitee（镜像）
# ============================================================
#
# 用法:
#   .\push-all.ps1                  # 推送当前分支
#   .\push-all.ps1 main             # 推送指定分支
#   .\push-all.ps1 -Force           # 强制推送（慎用）
#   .\push-all.ps1 -Tags            # 推送所有标签
#
# 环境变量:
#   $env:PUSH_BRANCH  — 默认推送的分支

param(
    [string]$Branch = "",
    [switch]$Force = $false,
    [switch]$Tags = $false
)

# 获取当前分支
if (-not $Branch) {
    $Branch = $env:PUSH_BRANCH
    if (-not $Branch) {
        $Branch = git rev-parse --abbrev-ref HEAD
    }
}

$ForceFlag = if ($Force) { "--force" } else { "" }

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  双远程推送" -ForegroundColor Cyan
Write-Host "  分支: $Branch" -ForegroundColor Yellow
Write-Host "  GitHub: github.com/fly34117/rehab_monitor.git" -ForegroundColor Cyan
Write-Host "  Gitee:  gitee.com/fiy-placid/rehab_monitor.git" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ===== 1. GitHub =====
Write-Host ""
Write-Host "[1/2] 推送到 GitHub (origin)..." -ForegroundColor Green
git push origin $Branch $ForceFlag
if ($LASTEXITCODE -eq 0) {
    Write-Host "  GitHub 推送成功" -ForegroundColor Green
} else {
    Write-Host "  GitHub 推送失败 (exit=$LASTEXITCODE)" -ForegroundColor Red
}

# ===== 2. Gitee =====
Write-Host ""
Write-Host "[2/2] 推送到 Gitee (gitee)..." -ForegroundColor Green
git push gitee $Branch $ForceFlag
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Gitee 推送成功" -ForegroundColor Green
} else {
    Write-Host "  Gitee 推送失败 (exit=$LASTEXITCODE)" -ForegroundColor Red
}

# ===== 3. Tags =====
if ($Tags) {
    Write-Host ""
    Write-Host "[Tags] 推送标签到两个远程..." -ForegroundColor Yellow
    git push origin --tags
    git push gitee --tags
    Write-Host "  标签推送成功" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  All done!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
