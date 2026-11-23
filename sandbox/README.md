# 沙箱篇

## 快速开始

在运行 Docker 守护进程的 Linux 环境中，从仓库根目录运行第 01 期检查。脚本使用 [`common/images.py`](common/images.py) 中固定了 digest 的镜像，首次运行时按需拉取。

```bash
cd sandbox
python3 chapter_01/run_checks.py
```

各期的检查入口均为 `chapter_XX/run_checks.py`。本地工作文件写入各期的 `data_local/`，检查结果写入 `results_local/`。第 01 期还可以单独跑基准测试或启动本地服务。

```bash
python3 chapter_01/benchmark.py
# 启动后在本机访问 http://127.0.0.1:8101/health
python3 chapter_01/sandbox_service.py 8101
```

### 第 14 期 microVM

运行第 14 期前，准备可访问的 `/dev/kvm`、`curl` 和 `debugfs`。将 Firecracker v1.17.0、guest 内核、initramfs、根磁盘和结果盘放到 `chapter_14/data_local/fc/`。具体文件名见 [第 14 期检查脚本](chapter_14/run_checks.py)。

## 实验索引

| 期 | 主题 |
| --- | --- |
| [01](chapter_01/) | 最小执行闭环 |
| [02](chapter_02/) | 会话状态 |
| [03](chapter_03/) | 进程树和取消 |
| [04](chapter_04/) | 资源配额 |
| [05](chapter_05/) | 文件与产物 |
| [06](chapter_06/) | 网络出站 |
| [07](chapter_07/) | 凭据与外部操作 |
| [08](chapter_08/) | 输出与回压 |
| [09](chapter_09/) | 镜像冷启动 |
| [10](chapter_10/) | 环境组合 |
| [11](chapter_11/) | 空闲会话回收 |
| [12](chapter_12/) | 准入与放置 |
| [13](chapter_13/) | 持久状态机与幂等性 |
| [14](chapter_14/) | 容器与 microVM |
| [15](chapter_15/) | 训练 rollout 与评测 |
| [16](chapter_16/) | 在线云端 Agent |
| [17](chapter_17/) | 综合验收 |
