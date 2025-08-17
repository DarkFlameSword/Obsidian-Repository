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
