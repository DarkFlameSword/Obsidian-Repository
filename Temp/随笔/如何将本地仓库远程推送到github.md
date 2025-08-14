---
date: 2025-08-14
author:
  - Siyuan Liu
tags:
  - 随笔
aliases:
  - summary
---
# 给GitHub账号添加SSH
# 安装Git
# 在Github创建空白仓库
==注意:==
- 不要勾选 Initialize this repository with a README
# 在本地为上传目标初始化Git数据文件
1. 鼠标右键上传目标文件夹
2. 点击`Open Git Bash here`
3. `git init`初始化 Git 仓库
4. `git add .`添加所有文件到暂存区
5. `git commit -m "Initial commit"`提交
# 移除因为失败推送而产生的推送文件
`git remote remove origin`
# 推送目标文件夹
==注意:==
- 请逐行运行, 不要全部复制
```
git remote add origin git@github.com:DarkFlameSword/FIT5047-Assignment1.git
git branch -M main
git push -u origin main
```