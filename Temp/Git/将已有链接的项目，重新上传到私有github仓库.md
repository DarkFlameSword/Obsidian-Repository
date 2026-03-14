---
date: 2026-03-14
author:
  - Siyuan Liu
tags:
  - summary
---
## 方法一：创建新仓库后push（推荐）

### 1. **在你的GitHub上创建新仓库**

- 登录GitHub
- 点击右上角 `+` → `New repository`
- 填写仓库名称、描述等信息
- 点击 `Create repository`（先不要初始化README）

### 2. **本地操作**

bash

```bash
# 进入clone下来的项目目录
cd your-project

# 移除原来的远程仓库地址
git remote remove origin

# 添加你自己的仓库地址（替换USERNAME和REPO_NAME）
git remote add origin https://github.com/USERNAME/REPO_NAME.git

# 或者使用SSH（需要配置SSH密钥）
git remote add origin git@github.com:USERNAME/REPO_NAME.git

# 重命名分支为main（如果需要）
git branch -M main

# 推送所有内容到你的仓库
git push -u origin main
```

---

## 方法二：镜像克隆（保留完整历史）

如果想保留原项目的完整commit历史：

bash

```bash
# 使用--mirror选项克隆
git clone --mirror https://github.com/ORIGINAL_OWNER/ORIGINAL_REPO.git

# 进入镜像目录
cd ORIGINAL_REPO.git

# 推送到你的仓库
git push --mirror https://github.com/USERNAME/YOUR_REPO.git

# 删除镜像目录
cd ..
rm -rf ORIGINAL_REPO.git
```

---

## 方法三：处理有下游仓库的情况

如果想保留原仓库的分支和标签：

bash

```bash
# clone项目
git clone https://github.com/ORIGINAL_OWNER/ORIGINAL_REPO.git
cd ORIGINAL_REPO

# 修改远程地址
git remote set-url origin https://github.com/USERNAME/YOUR_REPO.git

# 推送所有分支和标签
git push -u origin --all
git push -u origin --tags
```

---

## 常见问题

**Q: 推送时提示"rejected"？**

bash

```bash
# 如果远程仓库已有内容，可以尝试
git push -u origin main --force  # 强制推送（谨慎使用）
```

**Q: 想同时保留原仓库的更新链接？**

bash

```bash
# 添加上游仓库
git remote add upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPO.git

# 后续想同步上游更新
git fetch upstream
git merge upstream/main
```

**Q: 需要认证？**

bash

```bash
# 使用个人访问令牌(Personal Access Token)替代密码
# 在HTTPS URL中：https://USERNAME:TOKEN@github.com/USERNAME/REPO.git
```