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
2. windows搜索框搜索**环境变量**，点击**编辑系统环境变量**
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
3. ~~返回到HMCL的主界面，点击**下载**~~
4. ~~选择**正式版1.20.1**~~
5. ~~点击Forge，选择**47.4.0**，然后点击**安装**~~

---
## 添加mod
1. 点击**实例列表**
2. 点击**安装整合包**
3. 将`落幕曲1.5.1安装包（拖入启动器安装）(1).zip`拖入HMCL界面


---
## 进入游戏
1. 进入HMCL主界面，点击右下角![[Pasted image 20250930200053.png]]
2. 选择`落幕曲`服务器
3. 点击**启动游戏**
4. 选择**直接连接**![[Pasted image 20250930213942.png]]
5. 输入服务器IP`124.223.25.184:25565`

---
# 服务端操作



--- 

# Minecraft 收费须知
1. 当您在3/10/2025进入本服务器一次后，默认您参与付费项目
2. 本服务器采购的是腾讯云`上海 CPU - 4核 内存 - 16GB 系统盘 - SSD云硬盘 180GB峰值带宽 - 200Mbps`轻量云服务器，月订阅价格为320![[Pasted image 20251002220544.png]]
3. 当所有付费用户都同意关闭服务器时/当云服务器到期时，开始计算收费流程
4. 收费计算公式如下：
$$pay = \frac{320-R}{N}$$
- $R$: 退还服务器的退款费用
- $N$: 参与付费的用户

1. 本服务器会在北京时间14：00-23：30开启，其余时间会关闭服务器（如需在服务器关闭期间登录，请联系管理员）
2. 在服务器到期后可以考虑本地保存所有用户数据，方便以后使用
3. 服务器IP地址`124.223.25.184:25565`