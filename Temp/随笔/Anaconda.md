---
date: 2026-04-01
author:
  - Siyuan Liu
tags:
  - summary
---
### 1. 环境管理 (Environment Management)
这是 `conda` 最核心的功能，用于隔离不同项目的依赖。

*   **查看所有环境**：
    ```bash
    conda env list
    # 或者 conda info --envs
    ```
*   **创建新环境**（指定 Python 版本）：
    ```bash
    conda create --name myenv python=3.13.12
    ```
*   **激活环境**：
    ```bash
    conda activate myenv
    ```
*   **退出当前环境**（回到 base 环境）：
    ```bash
    conda deactivate
    ```
*   **删除环境**（删除整个环境及其所有包）：
    ```bash
    conda remove --name myenv --all
    ```
*   **克隆环境**（复制一个与旧环境完全相同的新环境）：
    ```bash
    conda create --name newenv --clone oldenv
    ```
*   **导出与导入环境**（方便在其他机器上重建环境）：
    ```bash
    # 导出环境配置
    conda env export > environment.yml
    # 根据配置文件创建环境
    conda env create -f environment.yml
    ```

### 2. 包管理 (Package Management)
在**已激活**的环境中安装、卸载和查看库。

*   **查看当前环境下已安装的包**：
    ```bash
    conda list
    ```
*   **搜索包**（查看有哪些版本可用）：
    ```bash
    conda search numpy
    ```
*   **安装包**：
    ```bash
    conda install numpy
    # 指定版本安装
    conda install numpy=1.21.0
    # 同时安装多个包
    conda install numpy pandas scipy
    ```
*   **卸载包**：
    ```bash
    conda remove numpy
    ```
*   **更新包**：
    ```bash
    conda update numpy
    # 更新当前环境下的所有包
    conda update --all
    ```

### 3. 配置与系统管理 (Configuration & System)
*   **查看 conda 版本**：
    ```bash
    conda --version
    ```
*   **更新 conda 自身**：
    ```bash
    conda update conda
    ```
*   **清理缓存**（释放硬盘空间，清理未使用的包和压缩包）：
    ```bash
    conda clean --all
    ```

### 4. 镜像源管理 (Channels 配置)
由于官方源下载速度较慢，国内通常会配置清华、中科大等镜像源。

*   **查看当前配置的源**：
    ```bash
    conda config --show channels
    ```
*   **添加镜像源**（以清华源为例）：
    ```bash
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    # 设置搜索时显示通道地址
    conda config --set show_channel_urls yes
    ```
*   **删除镜像源**：
    ```bash
    conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    ```
*   **恢复默认源**：
    ```bash
    conda config --remove-key channels
    ```