---
date: 2026-04-01
author:
  - Siyuan Liu
tags:
  - 随笔
---
### **安装与配置**
| 命令 | 说明 |
|---|---|
| `wsl --install` | 安装 WSL（默认安装 Ubuntu） |
| `wsl --install -d <发行版>` | 安装指定发行版（如 `Ubuntu-22.04`、`Debian`） |
| `wsl --list --online` | 列出所有可安装的发行版 |

### **管理发行版**
| 命令 | 说明 |
|---|---|
| `wsl --list --verbose` / `wsl -l -v` | 列出已安装的发行版及其状态和 WSL 版本 |
| `wsl --set-version <发行版> 2` | 将指定发行版转换为 WSL 2 |
| `wsl --set-default-version 2` | 将 WSL 2 设为新安装的默认版本 |
| `wsl --set-default <发行版>` | 设置默认发行版 |
| `wsl --unregister <发行版>` | 删除某个发行版（⚠️ 会清除所有数据） |

### **启动与停止**
| 命令 | 说明 |
|---|---|
| `wsl` | 启动默认发行版 |
| `wsl -d <发行版>` | 启动指定发行版 |
| `wsl --shutdown` | 关闭所有 WSL 实例 |
| `wsl -t <发行版>` | 终止指定的运行中发行版 |
| `wsl --status` | 查看 WSL 状态与配置信息 |

### **执行命令**
| 命令 | 说明 |
|---|---|
| `wsl <命令>` | 在 Windows 中直接运行 Linux 命令（如 `wsl ls -la`） |
| `wsl -d <发行版> -e <命令>` | 在指定发行版中运行命令 |
| `wsl -u root` | 以 root 用户启动发行版 |

### **导出与导入（备份）**
| 命令 | 说明 |
|---|---|
| `wsl --export <发行版> <文件.tar>` | 将发行版导出为 `.tar` 文件 |
| `wsl --import <发行版> <安装目录> <文件.tar>` | 从 `.tar` 文件导入发行版 |

### **网络与版本**
| 命令 | 说明 |
|---|---|
| `wsl --version` | 查看 WSL 版本信息 |
| `wsl --update` | 更新 WSL 内核 |
| `wsl --mount <磁盘>` | 将物理磁盘挂载到 WSL 2 |

---

### 💡 实用技巧

- **在 WSL 中访问 Windows 文件：** 路径为 `/mnt/c/`、`/mnt/d/` 等
- **在 Windows 资源管理器中访问 WSL 文件：** 地址栏输入 `\\wsl$`
- **配置文件：**
  - `~/.wslconfig`（Windows 用户目录）— 全局配置
  - `/etc/wsl.conf`（发行版内部）— 单个发行版配置

