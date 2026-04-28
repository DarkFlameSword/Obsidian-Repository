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

^480f0b

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
==或者在`C:\Users\<你的用户名>\.ssh\`路径下寻找.pub文件，然后用记事本打开.pub==
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
## 1. 下载 Git 安装包[Git - Downloads](https://git-scm.com/downloads)
==注意:==
- 选择组件**Git Bash Here**、**Git GUI Here** 必须勾上
## 2.验证安装
```
git --version
```
如果显示类似以下消息,则表示安装成功
```
git version 2.45.1.windows.1
```
## 3.配置用户名和邮箱
```
git config --global user.name "your GitHub username"
git config --global user.email "your GitHub E-mail"
```
# 在Github创建空白仓库
==注意:==
- 不要勾选 Initialize this repository with a README
# 在本地为上传目标初始化Git数据文件
1. 鼠标右键上传目标文件夹
2. 点击`Open Git Bash here`
3. `git init`初始化 Git 仓库
4. `git add .`添加所有文件到暂存区
5. `git commit -m "Initial commit"`提交
# 推送目标文件夹
==注意:==
- 请逐行运行, 不要全部复制
- 在Github上新建的空白仓库页面, **会提供官方完全正确的命令行**
```
类似
git remote add origin https://github.com/DarkFlameSword/Simple-Time-Countdown.git
git branch -M main
git push -u origin main
```

- 如果是推送文件到私有仓库，需要先使用下面代码测试 SSH 是否连通
```
ssh -T git@github.com

git remote add origin git@github.com:用户名/仓库名.git
```


# 标准软件开发

## 一、上传前必须补充

```bash
.gitignore
README.md
LICENSE
CHANGELOG.md
CONTRIBUTING.md
SECURITY.md
```

其中：

|文件|作用|
|---|---|
|`.gitignore`|排除构建产物、缓存、密钥、虚拟环境|
|`README.md`|项目介绍、安装、运行、截图、常见问题|
|`LICENSE`|明确别人能否使用、修改、分发你的代码|
|`CHANGELOG.md`|记录每个版本改了什么|
|`CONTRIBUTING.md`|贡献规范、分支规则、提交规则|
|`SECURITY.md`|漏洞报告方式|

GitHub 官方也建议用 `LICENSE` 明确开源权限；没有 license 的公开仓库并不等于别人可以自由使用。([GitHub Docs](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-license-to-a-repository?utm_source=chatgpt.com "Adding a license to a repository"))

## 二、正式 App 应该补充版本管理

建议采用语义化版本：

```text
v1.0.0
v1.0.1
v1.1.0
v2.0.0
```

基本规则：

```text
主版本号：破坏性变更
次版本号：新增功能
修订号：Bug 修复
```

例如：

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

然后在 GitHub 上创建 Release，并上传安装包，例如：

```text
Windows: .exe / .msi
macOS: .dmg
Linux: .AppImage / .deb
```

GitHub Release 可以用于打包软件、发布说明和二进制文件。([GitHub Docs](https://docs.github.com/en/repositories/releasing-projects-on-github?utm_source=chatgpt.com "Releasing projects on GitHub"))

## 三、建议补充分支策略

不要长期直接在 `main` 上开发。

推荐：

```text
main        稳定发布版
dev         开发版
feature/*   新功能分支
fix/*       Bug 修复分支
release/*   发布准备分支
```

常见流程：

```bash
git checkout -b dev
git checkout -b feature/pvp-recommendation
```

完成后通过 Pull Request 合并，而不是直接 push 到 `main`。

## 四、建议补充 GitHub 仓库设置

正式项目建议打开：

```text
Issues
Discussions
Wiki，可选
Branch protection
Pull Request review
Dependabot
Security policy
Releases
```

尤其是：

```text
main 分支禁止直接 push
合并前必须通过检查
Release 使用 tag 管理
```

## 五、建议补充构建与发布流程

如果是本地 App，最好有：

```text
build 脚本
打包脚本
版本号写入程序
自动生成 release 包
校验安装包
```

例如项目根目录可以有：

```text
scripts/
  build.bat
  build.sh
  package.bat
  release.bat
```

如果后期成熟，可以加 GitHub Actions 自动构建。

## 六、推荐的正式版流程

```bash
git init

git add README.md LICENSE .gitignore CHANGELOG.md CONTRIBUTING.md SECURITY.md
git add .

git commit -m "Initial commit"

git branch -M main

git remote add origin git@github.com:用户名/仓库名.git

git push -u origin main
```

发布第一个版本：

```bash
git tag -a v0.1.0 -m "Initial public release"
git push origin v0.1.0
```

然后去 GitHub 创建 Release，上传本地 App 安装包。

---

# 之后如果该如何更新？

## 1. 普通更新代码

```bash
git status
git add .
git commit -m "Update feature"
git push origin main
```

这只是更新 GitHub 上的代码，不会生成新的 Release。

## 2. 要发布新版本

假设之前已经发布：

```bash
v0.1.0
```

这次更新后可以发布：

```bash
v0.1.1
```

流程：

```bash
git status
git add .
git commit -m "Fix bugs in v0.1.1"
git push origin main

git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

然后到 GitHub：

```text
Releases → Draft a new release → 选择 v0.1.1 → 写更新说明 → 上传安装包 → Publish release
```

## 3. 版本号怎么加

```text
v0.1.0 → v0.1.1    小修复
v0.1.0 → v0.2.0    新功能
v0.1.0 → v1.0.0    第一个正式稳定版
v1.0.0 → v2.0.0    不兼容的大改动
```

## 4. 不要重复使用旧 tag

不要这样做：

```bash
git tag -a v0.1.0 -m "New release"
git push origin v0.1.0
```

`v0.1.0` 已经代表旧版本了。  
有更新就创建新 tag，例如：

```bash
v0.1.1
v0.2.0
v1.0.0
```

## 5. 推荐实际流程

每次修改后：

```bash
git add .
git commit -m "Describe your change"
git push origin main
```

准备发布时：

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

然后创建新的 GitHub Release。