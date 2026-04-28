---
date: 2026-04-01
author:
  - Siyuan Liu
tags:
  - summary
---
# 🐧 Ubuntu 常用命令大全

---

## 📁 文件与目录操作

| 命令                             | 说明                    |
| ------------------------------ | --------------------- |
| `ls -la`                       | 列出所有文件（含隐藏文件）及详细信息    |
| `cd <目录>`                      | 切换目录                  |
| `pwd`                          | 显示当前路径                |
| `mkdir -p <目录>`                | 创建目录（含多级目录）           |
| `rm -rf <目录/文件>`               | 强制删除目录或文件             |
| `cp -r <源> <目标>`               | 递归复制目录或文件             |
| `mv <源> <目标>`                  | 移动或重命名文件              |
| `touch <文件>`                   | 创建空文件或更新时间戳           |
| `find / -name <文件名>`           | 全局搜索文件                |
| `find . -type f -name "*.log"` | 搜索当前目录下所有 `.log` 文件   |
| `locate <文件名>`                 | 快速查找文件（需先 `updatedb`） |
| `tree`                         | 以树形结构显示目录             |
| `du -sh <目录>`                  | 查看目录占用空间              |
| `df -h`                        | 查看磁盘使用情况              |
| `ln -s <源> <链接>`               | 创建软链接                 |
| `stat <文件>`                    | 查看文件详细元信息             |
| `file <文件>`                    | 查看文件类型                |

---

## 📝 文件内容查看与编辑

| 命令 | 说明 |
|---|---|
| `cat <文件>` | 查看文件内容 |
| `less <文件>` | 分页查看文件（推荐） |
| `head -n 20 <文件>` | 查看文件前 20 行 |
| `tail -n 20 <文件>` | 查看文件后 20 行 |
| `tail -f <文件>` | 实时追踪文件末尾（看日志神器） |
| `grep -rn "关键词" <目录>` | 递归搜索关键词并显示行号 |
| `grep -i "关键词" <文件>` | 忽略大小写搜索 |
| `grep -v "关键词" <文件>` | 反向过滤（不含关键词的行） |
| `awk '{print $1}' <文件>` | 按列提取内容 |
| `sed -i 's/旧/新/g' <文件>` | 全局替换文件内容 |
| `wc -l <文件>` | 统计文件行数 |
| `sort <文件>` | 排序文件内容 |
| `uniq -c` | 去重并统计重复次数 |
| `diff <文件1> <文件2>` | 比较两个文件差异 |
| `vim <文件>` | 用 Vim 编辑文件 |
| `nano <文件>` | 用 Nano 编辑文件（新手友好） |

---

## 👤 用户与权限管理

| 命令 | 说明 |
|---|---|
| `whoami` | 显示当前用户 |
| `id` | 显示用户 UID/GID 信息 |
| `adduser <用户名>` | 创建新用户 |
| `userdel -r <用户名>` | 删除用户及其主目录 |
| `passwd <用户名>` | 修改用户密码 |
| `usermod -aG <组> <用户>` | 将用户加入指定组 |
| `groups <用户>` | 查看用户所属组 |
| `su - <用户名>` | 切换用户 |
| `sudo <命令>` | 以 root 权限执行命令 |
| `sudo -i` | 切换到 root 用户环境 |
| `chmod 755 <文件>` | 设置文件权限 |
| `chmod -R 755 <目录>` | 递归设置目录权限 |
| `chown user:group <文件>` | 修改文件所有者 |
| `chown -R user:group <目录>` | 递归修改目录所有者 |
| `umask 022` | 设置默认权限掩码 |
| `visudo` | 安全编辑 sudoers 文件 |

---

## 📦 软件包管理（APT）

| 命令 | 说明 |
|---|---|
| `apt update` | 更新软件包索引 |
| `apt upgrade -y` | 升级所有已安装软件包 |
| `apt full-upgrade -y` | 完整升级（含依赖变更） |
| `apt install <包名>` | 安装软件包 |
| `apt remove <包名>` | 卸载软件包（保留配置） |
| `apt purge <包名>` | 彻底卸载软件包（含配置） |
| `apt autoremove` | 清理无用依赖 |
| `apt search <关键词>` | 搜索软件包 |
| `apt show <包名>` | 查看软件包详情 |
| `dpkg -l` | 列出所有已安装的包 |
| `dpkg -i <包.deb>` | 手动安装 `.deb` 包 |
| `dpkg -r <包名>` | 卸载 `.deb` 包 |
| `apt-cache policy <包名>` | 查看包的可用版本 |
| `add-apt-repository <ppa>` | 添加 PPA 源 |

---

## ⚙️ 系统服务管理（Systemd）

| 命令 | 说明 |
|---|---|
| `systemctl start <服务>` | 启动服务 |
| `systemctl stop <服务>` | 停止服务 |
| `systemctl restart <服务>` | 重启服务 |
| `systemctl reload <服务>` | 重载配置（不中断服务） |
| `systemctl enable <服务>` | 设置开机自启 |
| `systemctl disable <服务>` | 取消开机自启 |
| `systemctl status <服务>` | 查看服务状态 |
| `systemctl list-units --type=service` | 列出所有服务 |
| `journalctl -u <服务>` | 查看指定服务日志 |
| `journalctl -u <服务> -f` | 实时追踪服务日志 |
| `journalctl -xe` | 查看系统错误日志 |
| `journalctl --since "1 hour ago"` | 查看最近 1 小时日志 |

---

## 🔍 进程管理

| 命令 | 说明 |
|---|---|
| `ps aux` | 查看所有进程 |
| `ps aux | grep <进程名>` | 搜索指定进程 |
| `top` | 实时查看进程资源占用 |
| `htop` | 更友好的进程监控（需安装） |
| `kill <PID>` | 终止进程 |
| `kill -9 <PID>` | 强制终止进程 |
| `killall <进程名>` | 按名称终止所有同名进程 |
| `pkill <进程名>` | 按名称发送信号 |
| `pgrep <进程名>` | 按名称查找进程 PID |
| `nice -n 10 <命令>` | 以低优先级运行命令 |
| `renice -n 5 -p <PID>` | 调整运行中进程优先级 |
| `nohup <命令> &` | 后台运行命令（关闭终端不中断） |
| `jobs` | 查看后台任务 |
| `bg %1` | 将任务放到后台运行 |
| `fg %1` | 将后台任务调到前台 |
| `screen` / `tmux` | 终端多路复用（持久会话） |

---

## 🌐 网络管理

| 命令 | 说明 |
|---|---|
| `ip a` | 查看所有网络接口 IP |
| `ip r` | 查看路由表 |
| `ip link set <网卡> up/down` | 启用/禁用网卡 |
| `ping -c 4 <地址>` | 测试网络连通性 |
| `traceroute <地址>` | 路由追踪 |
| `nslookup <域名>` | DNS 查询 |
| `dig <域名>` | 详细 DNS 查询 |
| `curl -I <URL>` | 查看 HTTP 响应头 |
| `curl -O <URL>` | 下载文件 |
| `wget <URL>` | 下载文件 |
| `wget -r <URL>` | 递归下载网站 |
| `ss -tulnp` | 查看所有监听端口及进程 |
| `netstat -tulnp` | 查看端口占用（需安装 net-tools） |
| `lsof -i :<端口>` | 查看指定端口占用进程 |
| `iptables -L -n -v` | 查看防火墙规则 |
| `ufw status` | 查看 UFW 防火墙状态 |
| `ufw allow <端口>` | 开放端口 |
| `ufw deny <端口>` | 关闭端口 |
| `ufw enable` / `ufw disable` | 启用/禁用 UFW |
| `scp <文件> user@host:<路径>` | 远程复制文件 |
| `rsync -avz <源> user@host:<目标>` | 高效远程同步文件 |
| `ssh user@host` | SSH 连接远程主机 |
| `ssh-keygen -t rsa` | 生成 SSH 密钥对 |
| `ssh-copy-id user@host` | 复制公钥到远程主机 |

---

## 💾 磁盘与存储管理

| 命令 | 说明 |
|---|---|
| `lsblk` | 查看所有块设备 |
| `fdisk -l` | 查看磁盘分区信息 |
| `fdisk /dev/sdb` | 对磁盘进行分区操作 |
| `mkfs.ext4 /dev/sdb1` | 格式化分区为 ext4 |
| `mount /dev/sdb1 /mnt` | 挂载分区 |
| `umount /mnt` | 卸载分区 |
| `blkid` | 查看分区 UUID |
| `df -h` | 查看磁盘使用情况 |
| `du -sh /*` | 查看各目录占用空间 |
| `ncdu` | 交互式磁盘使用分析（���安装） |
| `swapoff -a` / `swapon -a` | 关闭/开启所有 Swap |
| `free -h` | 查看内存和 Swap 使用情况 |

---

## 📊 系统监控与性能

| 命令 | 说明 |
|---|---|
| `uptime` | 查看系统运行时长和负载 |
| `w` | 查看当前登录用户及负载 |
| `vmstat 1` | 每秒输出系统性能统计 |
| `iostat -x 1` | 查看磁盘 I/O 统计 |
| `sar -u 1 5` | 查看 CPU 使用率历史 |
| `mpstat` | 查看多核 CPU 使用情况 |
| `free -h` | 查看内存使用情况 |
| `cat /proc/cpuinfo` | 查看 CPU 详情 |
| `cat /proc/meminfo` | 查看内存详情 |
| `lscpu` | 查看 CPU 架构信息 |
| `lshw -short` | 查看硬件概要信息 |
| `dmesg | tail` | 查看内核日志（硬件报错等） |
| `uname -a` | 查看系统内核版本 |
| `lsb_release -a` | 查看 Ubuntu 发行版信息 |
| `timedatectl` | ��看/设置系统时间 |

---

## 🗜️ 压缩与解压

| 命令 | 说明 |
|---|---|
| `tar -czvf <包.tar.gz> <目录>` | 压缩为 `.tar.gz` |
| `tar -xzvf <包.tar.gz>` | 解压 `.tar.gz` |
| `tar -cjvf <包.tar.bz2> <目录>` | 压缩为 `.tar.bz2` |
| `tar -xjvf <包.tar.bz2>` | 解压 `.tar.bz2` |
| `zip -r <包.zip> <目录>` | 压缩为 `.zip` |
| `unzip <包.zip> -d <目录>` | 解压 `.zip` 到指定目录 |
| `gzip <文件>` | 压缩单个文件 |
| `gunzip <文件.gz>` | 解压 `.gz` 文件 |
| `xz -z <文件>` | 压缩为 `.xz` |
| `xz -d <文件.xz>` | 解压 `.xz` |

---

## 🔐 安全与审计

| 命令 | 说明 |
|---|---|
| `last` | 查看用户登录历史 |
| `lastb` | 查看登录失败记录 |
| `who` | 查看当前登录用户 |
| `cat /var/log/auth.log` | 查看认证日志 |
| `cat /var/log/syslog` | 查看系统日志 |
| `fail2ban-client status` | 查看 Fail2Ban 状态 |
| `openssl genrsa -out key.pem 2048` | 生成 RSA 私钥 |
| `openssl req -new -x509 ...` | 生成自签名证书 |
| `md5sum <文件>` | 计算 MD5 校验值 |
| `sha256sum <文件>` | 计算 SHA256 校验值 |
| `gpg --gen-key` | 生成 GPG 密钥 |
| `gpg -e -r <收件人> <文件>` | GPG 加密文件 |
| `chmod 600 ~/.ssh/id_rsa` | 确保私钥权限安全 |

---

## 🐳 Docker 常用（生产必备）

| 命令 | 说明 |
|---|---|
| `docker ps -a` | 查看所有容器 |
| `docker images` | 查看本地镜像 |
| `docker run -d -p 80:80 <镜像>` | 后台运行容器并映射端口 |
| `docker exec -it <容器> bash` | 进入运行中的容器 |
| `docker logs -f <容器>` | 实时查看容器日志 |
| `docker stop <容器>` | 停止容器 |
| `docker rm <容器>` | 删除容器 |
| `docker rmi <镜像>` | 删除镜像 |
| `docker-compose up -d` | 后台启动 Compose 服务 |
| `docker-compose down` | 停止并删除 Compose 服务 |
| `docker system prune -af` | 清理所有无用资源 |
| `docker inspect <容器>` | 查看容器详细信息 |
| `docker stats` | 实时查看容器资源占用 |

---

## 🔧 Shell 与效率技巧

| 命令 | 说明 |
|---|---|
| `history` | 查看命令历史 |
| `!!` | 执行上一条命令 |
| `!<关键词>` | 执行最近匹配的命令 |
| `Ctrl + R` | 搜索历史命令 |
| `alias ll='ls -la'` | 创建命令别名 |
| `export VAR=value` | 设置环境变量 |
| `echo $VAR` | 输出变量值 |
| `env` | 查看所有环境变量 |
| `which <命令>` | 查看命令路径 |
| `type <命令>` | 查看命令类型 |
| `xargs` | 将输入转为命令参数 |
| `tee <文件>` | 同时输出到屏幕和文件 |
| `watch -n 2 <命令>` | 每 2 秒执行一次命令 |
| `crontab -e` | 编辑定时任务 |
| `crontab -l` | 查看定时任务列表 |
| `at now + 5 min` | 5 分钟后执行一次性任务 |

---

## ⏰ Crontab 时间格式速查

```
* * * * *  命令
│ │ │ │ │
│ │ │ │ └── 星期 (0-7, 0和7都是周日)
│ │ │ └──── 月份 (1-12)
│ │ └────── 日期 (1-31)
│ └──────── 小时 (0-23)
└────────── 分钟 (0-59)

示例：
0 2 * * *     每天凌晨 2 点执行
*/5 * * * *   每 5 分钟执行
0 0 * * 1     每周一午夜执行
```

---

> 💡 **建议：** 生产环境操作前先 `--dry-run` 或在测试环境验证，`rm -rf` / `dd` 等危险命令务必二次确认！