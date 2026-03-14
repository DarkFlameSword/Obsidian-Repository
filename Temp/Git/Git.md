---
date: 2025-08-17
author:
  - Siyuan Liu
tags:
  - 随笔
aliases:
  - summary
---
# 常用命令
- 查看远程仓库是否添加成功：
```
git remote -v
```
- 新建并切换到分支
```
git checkout -b <branch-name>
```
- 查看当前分支
```
git branch
```
带 `*` 的就是你当前所在的分支

# 场景
## 推送本地修改后的版本到github
## 如果是第一次将本地文件remote到github，则需要提前配置用户信息
### 配置用户信息（身份标识）
Git 需要知道是谁在提交代码。这步配置会影响你提交记录中的名字和邮箱。

1. 打开你的终端（Terminal / Git Bash / PowerShell）
2. 输入以下命令（替换双引号内的内容为你自己的信息）：

```
# 设置你的用户名（可以是英文名或 GitHub ID）
git config --global user.name "你的名字"

# 设置你的邮箱（必须是你注册 GitHub 账户时使用的邮箱）
git config --global user.email "你的邮箱@example.com"
```

> **注意：** 这里的邮箱非常重要。如果和 GitHub 账号的邮箱不一致，你的 Commits 将不会计入 GitHub 的贡献图（绿格子）中
### 1. 添加修改过的文件到暂存区：
```
git add .
```
### 2. 提交改动到git：
```
git commit -m "描述本次更新内容"
```
### 3.第一次推送（并且指定默认推送分支）：
```
git push -u origin main
```
以后再推送更新就可以直接用：
```
git push
```
## 拉取最新项目代码到本地
### 1. 拉取远程最新代码：
```
git pull origin main
```
### 2. **查看提交历史：**
```
git log --oneline
```
### 3. 切换分支：
```
git checkout -b new-feature   # 新建并切换分支 
git checkout main             # 回到主分支
```

# 解决Git Conflict
**查看冲突文件：**
`git status`

**先尝试合并：**
```
获取远端更新：
git fetch origin

合并远端分支到本地：
git merge origin/main

解决冲突（见“冲突处理”）
git commit
git push
```

**如果无法合并，用本地覆盖远端历史**
```
git push --force-with-lease origin main
```
请慎用，先确认无人依赖远端现有提交

**如果无法合并，用远端覆盖本地**
```
git fetch origin
git reset --hard origin/main 会丢弃本地未推送的提交
```
