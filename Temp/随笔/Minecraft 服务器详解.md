---
date: 2025-09-30
author:
  - Siyuan Liu
tags:
  - 随笔
---
# 客户端操作
## JAVA 环境
1. [[[https://download.oracle.com/java/17/archive/jdk-17.0.12_windows-x64_bin.msi](https://download.oracle.com/java/17/archive/jdk-17.0.12_windows-x64_bin.msi) ([sha256](https://download.oracle.com/java/17/archive/jdk-17.0.12_windows-x64_bin.msi.sha256) )|下载JAVA17.0.12]]
2. 打开**系统环境变量设置**，右键开始菜单 → 选择 **系统** → **关于** → **高级系统设置** → 点击 **环境变量**
3. 在 **系统变量** 中点击 **新建**
    - 变量名：`JAVA_HOME`
    -  变量值：JDK 安装路径，例如：`C:\Program Files\Java\jdk-17.0.12`
4. 在 **系统变量** 中找到 `Path` → 点击 **编辑** → 点击 **新建**。
5. 添加：`%JAVA_HOME%\bin`
6. 保存并关闭所有窗口。
7. 验证安装是否成功, 打开**命令提示符**, 输入：`java -version`
	- 预期输出：`java version "17.0.x" Java(TM) SE Runtime Environment ...`

---
## HMCL启动器
1. 从[[https://github.com/HMCL-dev/HMCL/releases|HMCL下载地址]]下载win版本.exe文件
2. 点击**账户**，点击**离线模式**，创建用户名并且点击**登录**
3. 返回到HMCL的主界面，点击**下载**
4. 选择**正式版1.20.1**
5. 点击Forge，选择**47.4.0**，然后点击**安装**

---
## 添加mod
1. 点击**实例管理**
2. 点击**更新整合包**
3. 将`落幕曲1.5.1安装包（拖入启动器安装）(1).zip`拖入HMCL界面
## 进入游戏
1. 进入HMCL主界面，点击右下角![[Pasted image 20250930200053.png]]
2. 选择`落幕曲`服务器
3. 点击**启动游戏**
4. 选择**直接连接**![[Pasted image 20250930213942.png]]
5. 输入服务器IP`43.160.196.62:25565`

---
# 服务端操作
