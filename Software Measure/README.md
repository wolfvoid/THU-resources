# Software Measurement（软件度量期末大作业）

> **团队**：wolfvoid 小组

本项目为清华大学《软件度量》课程期末大作业归档。实验选取了三个具有代表性且输入特征差异显著的工业级开源 C 语言项目——多媒体字幕渲染库 **libass**、网络传输库 **curl** 以及轻量级 JSON 解析库 **cJSON**，采用**“静态分析先行 + 动态模糊测试（Fuzzing）深挖 + LLM Agent 辅助降噪与根因溯源”**的协同工作流开展安全评测与缺陷挖掘，并向上游社区提交了真实缺陷报告（[libass#928](https://github.com/libass/libass/issues/928)、[libass#931](https://github.com/libass/libass/issues/931)）。

---

## 1. 目录结构

仓库对原始实验工程进行了精简归档，剔除了上游源码副本、插桩二进制产物及数万个 AFL 中间语料队列文件，仅保留最终实验报告、手写 Fuzzing 测试桩/变异器、关键漏洞复现样本（PoC）与课程说明文档：

```text
Software Measure/
├── README.md                        # 项目全景说明文档
├── report/                          # 课程最终实验报告与配图
│   ├── 软件度量实验报告.pdf          # 最终实验报告（10 页 PDF 终稿）
│   ├── main.tex                     # 实验报告 LaTeX 源码
│   └── figures/                     # 报告静态分析、动态模糊测试与 Agent 界面截图
├── docs/                            # 课程题目与参考资料
│   ├── homework.md                  # 《软件度量》期末大作业要求说明
│   └── 模糊测试介绍.pdf              # 课程助教模糊测试与 AFL 入门讲义
└── libass_fuzz/                     # libass 深度模糊测试核心工程产物
    ├── REPORT_FINAL.md              # libass 静态分析 + 五轮 Fuzzing 综合技术报告
    ├── REPORT_ROUND1.md             # libFuzzer 第一轮（ASan + 字节盲变异）阶段报告
    ├── REPORT_ROUND2.md             # libFuzzer 第二轮（结构感知 + UBSan）阶段报告
    ├── BUG_REPORT.md                # 核心 OOM / 慢路径漏洞调用链分析与复现记录
    ├── Agent.md                     # Agent 辅助漏洞挖掘与分阶段执行规程
    ├── drivers/                     # 手写 Fuzzing 测试桩、自定义变异器与协议字典
    │   ├── target.c                 # libFuzzer 测试桩（LLVMFuzzerTestOneInput 多时间点渲染）
    │   ├── custom_mutator.c         # ASS/SSA 语法结构感知变异器（LLVMFuzzerCustomMutator）
    │   ├── afl_target.c             # AFL++ 双解析路径（内存整块 + 流式分块）测试桩
    │   └── ass.dict                 # ASS/SSA 字幕协议专属字典（96 个关键 token）
    ├── artifacts/                   # 动态测试捕获的真实异常输入与 PoC 样本
    │   ├── leak/                    # CWE-401 字体回退路径内存泄漏触发样本
    │   ├── oom/                     # 极端缩放 Bitmap 过度分配 OOM 样本（含 poc_oom.ass）
    │   └── slow_unit/               # CWE-400 \t() 动画标签超线性慢路径 DoS 样本
    └── plots/                       # AFL++ 多轮覆盖率、语料库演化与 Hangs 趋势对比图
        ├── afl_round1.png           # 第一轮：单实例 + ASan + 字典（12h，53.59% edge cov）
        ├── afl_round2.png           # 第二轮：CMPLOG + 4 核并行（56.82% edge cov）
        └── afl_comparison.png       # 两轮覆盖率与 Hangs 发现效率对比
```

---

## 2. 评测对象与核心结果汇总

| 评测目标 | 项目类型 | 静态分析工具 | 动态测试工具与策略 | 主要发现与结论 |
| :--- | :--- | :--- | :--- | :--- |
| **libass** (`v0.17.4`) | ASS/SSA 字幕渲染库（约 5 万行 C） | `cppcheck 1.90` | **libFuzzer**（盲变异 → 结构感知 `CustomMutator` → `\p` 矢量绘图专项）+ **AFL++**（字典 → `CMPLOG` + 4 核多策略并行），配合 `ASan` / `UBSan` | 挖掘出 **CWE-400** 动画标签 `\t()` 超线性慢路径与极端缩放 OOM（DoS，AFL++ 复现 541 次 Hangs）、**CWE-190** 矢量包围盒有符号整数溢出（`ass_render.c:1264-1265`）、**CWE-681** 浮点转整数未定义行为（`ass_render.c:3026`）及 **CWE-401** 字体匹配内存泄漏；向官方提交 2 个 Issue（#928、#931） |
| **curl** | 工业级网络传输库 | Clang `scan-build` | **AFL / AFL++** 围绕 `parseconfig` 与 URL 状态机编写 Harness，结合 HTTP/SOCKS5 协议字典与 `CMPLOG` 持续测试超 23 小时 | 静态分析定位并人工确认 **CWE-404** 异常分支文件流未关闭资源泄漏（`tool_parsecfg.c:264`）；动态测试主要表现为网络等待引起的 Timeout/Hang，核心内存安全防御稳健 |
| **cJSON** | 轻量级 JSON 解析库 | `cppcheck` | 三轮递进：官方 Harness 基线 → 自定义 4 字节模式控制 Driver + `ASan` → **AFL++ 4.33c** + `CMPLOG` + 4 实例并发扩展 API 覆盖（12h+） | 静态分析 121 条告警经核查均为跨平台宏定义（如 `NAN`）误报；12 小时高强度并发测试下 **0 Crashes / 0 Hangs**，验证了成熟精简解析器的高鲁棒性 |

---

## 3. 技术亮点

### 3.1 二维分支 + 多轮递进式模糊测试（以 libass 为代表）
- **突破强格式约束瓶颈**：针对 `.ass` 字幕包含 `[Script Info]`、`[V4+ Styles]`、`[Events]` 及复杂 Override Tags 的强语法约束，编写了 `libass_fuzz/drivers/custom_mutator.c`（实现 `LLVMFuzzerCustomMutator`），在字段与标签语义层进行变异，结合 `UBSan` 将覆盖边数从盲变异的 `2,262` 提升至 `12,480`。
- **双解析路径与多时间点驱动**：在 `libass_fuzz/drivers/afl_target.c` 中同时覆盖 `ass_read_memory()`（整块内存读取）和 `ass_process_data()`（流媒体分块喂入）两条解析链，并在事件起止及多个关键插值帧调用 `ass_render_frame()`，充分触发 `\t()`、`\move()`、`\fad()` 与 `\p` 贝塞尔绘图逻辑。
- **CMPLOG 与多策略协同放大**：AFL++ 第二轮引入 `AFL_LLVM_CMPLOG` 自动攻克多字节字符串比较，配合 `afl-cmin` 语料精简与 1 主（CMPLOG）+ 3 副（`explore` / `exploit` / `fast`）并行实例，总执行量提升至 **1,290 万次**，边覆盖率提升至 **56.82%**。

### 3.2 LLM Agent 辅助漏洞挖掘工作站
围绕传统评测中“手写 Harness 成本高、静态扫描误报多、动态堆栈分诊繁琐”三大痛点，基于 DeepSeek-V3 API 构建了三阶段辅助工作流：
1. **前置靶点分析与字典生成**：输入源码 AST 切片，辅助生成初始化测试桩（`LLVMFuzzerTestOneInput`）及协议关键字字典。
2. **静态告警清洗降噪**：自动过滤跨平台宏与环境兼容性误报（如 cJSON 的 121 条宏诊断），精准聚焦如 `curl` `tool_parsecfg.c` 的 CWE-404 资源未释放路径。
3. **动态崩溃分诊与根因分析（RCA）**：对 AFL++ / Sanitizer 产生的海量原始堆栈做聚类去重与调用链溯源，辅助快速构造最小可复现 PoC。

---

## 4. 快速复现与编译指南

### 4.1 重新编译实验报告 PDF
如需修改报告源码并在配备 `XeLaTeX`（支持 `ctexart`）的环境中重新生成 PDF：
```bash
cd report
xelatex main.tex
xelatex main.tex
```

### 4.2 复现 libass 模糊测试驱动
将 `libass_fuzz/drivers/` 下的测试桩放入克隆好的 [libass](https://github.com/libass/libass) 源码根目录即可插桩编译：

```bash
# 1. 编译结构感知 + ASan/UBSan 版 libFuzzer 目标
./autogen.sh
CC=clang CFLAGS="-g -O1 -fsanitize=address,undefined,fuzzer-no-link -fno-omit-frame-pointer" LDFLAGS="-fsanitize=address,undefined" ./configure --disable-shared --enable-static
make -j$(nproc)
clang -g -O1 -fsanitize=address,undefined,fuzzer target.c custom_mutator.c -I. ./libass/.libs/libass.a -lfreetype -lfontconfig -lfribidi -lharfbuzz -lm -o fuzzer_app_ubsan

# 重放 artifacts 中的异常样本（如 OOM / slow-unit / leak）
./fuzzer_app_ubsan ../artifacts/oom/poc_oom.ass

# 2. 编译 AFL++ ASan 版目标并结合 ass.dict 启动测试
AFL_USE_ASAN=1 CC=afl-clang-fast CFLAGS="-g -O1 -fsanitize=address" ./configure --disable-shared --enable-static && make -j$(nproc)
AFL_USE_ASAN=1 afl-clang-fast -g -O1 -fsanitize=address afl_target.c -I. ./libass/.libs/libass.a -lfreetype -lfontconfig -lfribidi -lharfbuzz -lm -o afl_target
AFL_USE_ASAN=1 afl-fuzz -i seeds -o afl_out -x ass.dict -t 10000 -m none -- ./afl_target @@
```
