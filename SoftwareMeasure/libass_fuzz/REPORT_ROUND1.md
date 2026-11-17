# libass Fuzzing 第一轮报告（ASan + libFuzzer）

## 运行参数

| 参数 | 值 |
|------|---|
| 工具 | libFuzzer + ASan |
| 并发 | 单进程 |
| `-rss_limit_mb` | 51200（50GB） |
| `-malloc_limit_mb` | 40960（40GB） |
| `-max_total_time` | 43200（12h） |
| 启动时间 | 2026-05-15 04:26 UTC |
| 结束时间 | 2026-05-15 ~16:20 UTC（约 11h 54min，接近 12h） |
| 退出方式 | 系统 cgroup 回收（无 SUMMARY，非正常超时退出） |

## 语料库

| 项目 | 数量 |
|------|------|
| 初始种子 | 1290 个 |
| 最终语料库 | 1136 个有效输入 |

## 覆盖率

| 指标 | 初始 | 最终 |
|------|------|------|
| cov（边覆盖数） | 2172 | **2262** |
| ft（特征数） | 11328 | **12959** |
| 总执行次数 | - | ~33432 |
| exec/s | ~3 | ~1~2 |
| 峰值 RSS | - | **14.15 GB** |

## 发现的问题

### crash（内存破坏类漏洞）
**无**。未发现 heap-buffer-overflow / use-after-free / stack-overflow 等内存安全漏洞。

### slow-unit（DoS 慢路径，共 8 个）

所有 slow-unit 均包含 `{\t(\fscx...)}` 动画变换 Override tag，结合损坏的 section 头（如 `PlayResY: 7:0`、乱码字段），导致 libass 渲染时间异常长。

| 文件 | 触发 tag | 特征 |
|------|---------|------|
| slow-unit-5ee60147... | `{\pos}{\move}{\fad}{\t(\fscx)}` | PlayResY 乱码 |
| slow-unit-47e0dfce... | `{\pos}{\move}{\fad}{\t(\fscx)}` | PlayResY 乱码 |
| slow-unit-58673368... | `{\t(\fscx200)}` | Star 字段损坏 |
| slow-unit-a79169a6... | `{\t(\fscx...)}` + 重复 Dialogue | 多行重复 |
| slow-unit-23407f68... | `{\t(\fs}}}}}}cx2}` | 嵌套畸形括号 |
| slow-unit-a57d385a... | `{\t(\fscx002)}` | 同上变体 |
| slow-unit-d69a8b01... | （上轮遗留） | - |
| slow-unit-acf37e79... | （本轮新增，08:37） | Styles 节重复 |

**根因分析：** `\t()` 动画变换 tag 在解析畸形参数时存在**超线性时间复杂度路径**，属于 **CWE-400（不受控资源消耗）**，可用于构造 DoS 攻击。

### oom artifact（上轮遗留，共 2 个）
- `oom-0b3373c2...`、`oom-9812c17f...`：上一轮（fuzz_run2）无限制内存下触发，本轮（50GB 上限）未复现新 OOM。

## 结论

- libass 在标准 `.ass` 格式变异下**未暴露内存破坏漏洞**
- 发现**慢路径 DoS 问题**，集中于 `\t()` 动画 tag + 畸形输入组合
- 覆盖率增长已趋于平缓（cov 2172→2262，增幅 4%），说明字节级盲变异已接近瓶颈

## 下一步计划

| 阶段 | 目标 | 预期收益 |
|------|------|---------|
| 阶段七 | 结构感知 fuzzing（custom mutator） | 突破覆盖率瓶颈，触达深层渲染路径 |
| 阶段八 | UBSan 联合插桩 | 检测整数溢出、未定义行为 |
| 阶段九 | CodeQL 静态分析 | 跨函数数据流漏洞挖掘 |
