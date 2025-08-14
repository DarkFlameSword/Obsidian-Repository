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
## 1.检查您电脑上是否已有SSH密钥
```
ls -al ~/.ssh
```
如果命令返回结果中包含 `id_rsa.pub`、`id_ecdsa.pub` 或 `id_ed25519.pub` 等文件，说明您已经有了一个密钥，可以跳到第三步
## 2.生成新的SSH密钥
1. 打开终端或Git Bash
2. 运行以下命令。建议使用 `Ed25519` 算法，它比 `RSA` 更安全、性能更好。请将命令中的邮箱地址换成您自己的GitHub邮箱
```
  ssh-keygen -t ed25519 -C "your_email@example.com"
```
3. 当提示 "Enter a file in which to save the key" 时，直接按 **Enter** 键接受默认文件位置即可
4. 接下来会提示您输入一个安全密码（passphrase）。您可以直接按 **Enter** 键跳过，但为了安全起见，强烈建议您设置一个密码。这个密码将在您每次使用SSH密钥时要求输入，能有效保护您的密钥安全
## 3.将SSH公钥复制到剪贴板
==对于 Ed25519算法生成的密钥==
```
cat ~/.ssh/id_ed25519.pub
```
命令执行后，终端会显示一串以 `ssh-ed25519` 或 `ssh-rsa` 开头的文本。这就是您的公钥。请完整地复制这串文本
## 4.将SSH公yect添加到您的GitHub账户
1. 登录您的 [GitHub](https://github.com/) 账户。
2. 点击右上角的个人头像，然后选择 **Settings** (设置)。
3. 在左侧菜单中，点击 **SSH and GPG keys** (SSH和GPG密钥)。
4. 点击 **New SSH key** (新建SSH密钥) 或 **Add SSH key** (添加SSH密钥) 按钮。
5. 在 **Title** (标题) 字段中，为您的密钥起一个容易识别的名字，例如 "My MacBook Pro" 或 "Work Laptop"。
6. 在 **Key** (密钥) 字段中，粘贴您在第三步中复制的完整公钥内容。
7. 点击 **Add SSH key** (添加SSH密钥) 按钮。如果系统提示，请输入您的GitHub密码进行确认。
## 5.测试SSH连接
1. 在终端或Git Bash中输入以下命令：
```
ssh -T git@github.com
```
2. 您可能会看到一条警告信息，询问您是否信任该主机的真实性。输入 **yes** 并按 **Enter**。
```
> The authenticity of host 'github.com (IP ADDRESS)' can't be established.
> ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA09iUeA6vgridUulV4.
> Are you sure you want to continue connecting (yes/no)? yes
```
3. 如果一切顺利，您将看到一条欢迎信息：
```
Hi <your name>! You've successfully authenticated, but GitHub does not provide shell acce
```
# 安装Git
==自行百度==
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
- 在Github上新建的空白仓库页面, 会提供完全正确的命令行, 下列代码行需要自行替换`YourUsername`和`RepoName`
```
git remote add origin git@github.com:YourUsername/RepoName.git
git branch -M main
git push -u origin main
```