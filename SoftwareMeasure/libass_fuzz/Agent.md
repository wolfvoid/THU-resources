# Agent 执行计划：libass 漏洞挖掘评测

## 环境基线（已确认）
- OS：Ubuntu 20.04.6 LTS，内核 4.18，x86_64
- 编译器：clang-10 / gcc-9（libFuzzer 支持已验证）
- 静态分析：cppcheck 1.90
- 依赖：freetype2 / fontconfig / fribidi / harfbuzz（pkg-config 全部就绪）
- 工作目录：`/workspace/wolfvoid/fuzz_project/`
- 权限：root（无需 sudo）

---

## 阶段一：获取源码与静态分析

### 1.1 克隆 libass 最新源码
```bash
cd /workspace/wolfvoid/fuzz_project
git clone https://github.com/libass/libass.git
cd libass
```

### 1.2 执行静态分析（报告必备截图）
```bash
cppcheck --enable=all --inconclusive --force --output-file=static_analysis.txt .
cat static_analysis.txt | head -60
```
**【暂停点 A】**：截取 cppcheck 终端扫描过程 + static_analysis.txt 告警内容截图。

---

## 阶段二：手写 Fuzzing Driver（报告必备截图）

### 2.1 driver 设计要点
- 入口：`LLVMFuzzerTestOneInput()`（libFuzzer 标准接口）
- 全局单例：`ass_library_init()` + `ass_renderer_init()` 仅初始化一次，避免重复开销
- 接口穿透：`ass_read_memory()` 载入变异字节流 → `ass_render_frame()` 多时间点渲染
- 内存安全：每次调用后严格 `ass_free_track()`

### 2.2 driver 文件路径
`/workspace/wolfvoid/fuzz_project/libass/target.c`

**【暂停点 B】**：driver 写入完成后截取源码截图。

---

## 阶段三：编译插桩

### 3.1 生成构建脚本
```bash
cd /workspace/wolfvoid/fuzz_project/libass
./autogen.sh
```

### 3.2 配置（ASan + fuzzer-no-link 插桩）
```bash
./configure \
    CC="clang" \
    CFLAGS="-g -O1 -fsanitize=address,fuzzer-no-link -fno-omit-frame-pointer" \
    LDFLAGS="-fsanitize=address" \
    --disable-shared \
    --enable-static
```

### 3.3 编译静态库
```bash
make clean && make -j$(nproc)
```

### 3.4 链接 fuzzer 可执行文件
```bash
clang -g -O1 -fsanitize=address,fuzzer \
    target.c \
    -I./libass \
    ./libass/.libs/libass.a \
    -lfreetype -lfontconfig -lfribidi -lharfbuzz -lm \
    -o fuzzer_app
```
> 注：include 路径 `-I./libass` 可能需根据实际头文件位置调整为 `-I.`

---

## 阶段四：构建种子语料库

```bash
mkdir -p /workspace/wolfvoid/fuzz_project/libass/corpus
cp test/subtitles/*.ass corpus/ 2>/dev/null || true
cp test/*.ass corpus/ 2>/dev/null || true
ls corpus/
```
若官方测试集不足，补充手动创建最小合法 .ass 种子：
```bash
cat > corpus/minimal.ass << 'SEED'
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,Bold,Italic,Underline,StrikeOut,Alignment
Style: Default,Arial,40,&H00FFFFFF,0,0,0,0,2

[Events]
Format: Layer,Start,End,Style,Text
Dialogue: 0,0:00:00.00,0:00:05.00,Default,Hello fuzzer!
SEED
```

---

## 阶段五：12 小时 Fuzzing（报告必备截图）

### 5.1 后台启动（nohup 防止断连）
```bash
mkdir -p /workspace/wolfvoid/fuzz_project/libass/crashes

nohup ./fuzzer_app \
    -max_total_time=43200 \
    -rss_limit_mb=51200 \
    -malloc_limit_mb=40960 \
    -print_final_stats=1 \
    -artifact_prefix=./crashes/ \
    ./corpus/ \
    > fuzz_run3.log 2>&1 &
echo $! > fuzzer.pid
```

### 5.2 监控与截图要求
- **【暂停点 C】启动截图**：引擎输出 `cov: ... ft: ... exec/s: ...` 时截图
- **【暂停点 D】结束截图**：12 小时后终端 Summary 面板截图（含崩溃数 + 覆盖边数）

---

## 阶段六：崩溃分析（Agent 开放题加分项）

若 `./crashes/` 目录中存在崩溃文件：

```bash
ls crashes/
# 用 ASan 报告重放崩溃
./fuzzer_app crashes/<crash_file> 2>&1 | head -80
```

Agent 将自动解析 `ERROR: AddressSanitizer` 堆栈，判断根因类型：
- `heap-buffer-overflow`
- `use-after-free`
- `stack-buffer-overflow`

并输出结构化 Markdown 漏洞分析报告，供实验报告"开放题"章节使用。

---

## 阶段七：结构感知 Fuzzing（Structure-aware Fuzzing）

当前 libFuzzer 是字节级盲变异，大量变异输入会被 libass 解析器在早期拒绝，无法触达渲染深层逻辑。结构感知 fuzzing 让变异器始终生成语法合法的 `.ass` 文件，从而覆盖更深的代码路径。

### 7.1 思路
使用自定义 mutator（`LLVMFuzzerCustomMutator`），将输入解析为 `.ass` 结构体后在字段级别变异：
- `[Script Info]`：随机化 `PlayResX/Y`、`ScriptType`
- `[V4+ Styles]`：变异字体名、字号、颜色、对齐方式等字段
- `[Events]`：变异时间码、Override tags（`{\an\fs\pos\move\t\clip}` 等）

### 7.2 实现方式
编写 `custom_mutator.c`，注册 `LLVMFuzzerCustomMutator`，内部维护 `.ass` 模板，随机替换字段值后序列化输出：

```bash
clang -g -O1 -fsanitize=address,fuzzer \
    target.c custom_mutator.c \
    -I. \
    ./libass/.libs/libass.a \
    -lfreetype -lfontconfig -lfribidi -lharfbuzz -lm \
    -o fuzzer_app_structured
```

### 7.3 预期收益
- 覆盖率从当前 cov ~2200 有望提升 30%+
- 能触达 `\t()`（动画变换）、`\clip()`（矢量裁剪）等复杂 Override tag 的渲染路径

---

## 阶段八：UBSan 插桩检测未定义行为

ASan 检测内存安全问题，但**整数溢出、数组越界（编译期已知大小）、空指针解引用**等未定义行为需要 UBSan 才能发现。

### 8.1 重新编译（ASan + UBSan 联合插桩）

```bash
cd /workspace/wolfvoid/fuzz_project/libass
make clean
./configure \
    CC="clang" \
    CFLAGS="-g -O1 -fsanitize=address,undefined,fuzzer-no-link -fno-omit-frame-pointer" \
    LDFLAGS="-fsanitize=address,undefined" \
    --disable-shared \
    --enable-static
make -j$(nproc)

clang -g -O1 -fsanitize=address,undefined,fuzzer \
    target.c \
    -I. \
    ./libass/.libs/libass.a \
    -lfreetype -lfontconfig -lfribidi -lharfbuzz -lm \
    -o fuzzer_app_ubsan
```

### 8.2 启动

```bash
mkdir -p crashes_ubsan
nohup ./fuzzer_app_ubsan \
    -max_total_time=43200 \
    -rss_limit_mb=51200 \
    -malloc_limit_mb=40960 \
    -print_final_stats=1 \
    -artifact_prefix=./crashes_ubsan/ \
    ./corpus/ \
    > fuzz_ubsan.log 2>&1 &
```

### 8.3 UBSan 重点关注的告警类型
- `signed-integer-overflow`：字幕时间码/坐标计算整数溢出
- `shift-exponent`：位移操作超出类型宽度
- `null-pointer-dereference`：空轨道/空样式指针解引用
- `array-bounds`：固定大小数组越界访问

---

## 阶段九：CodeQL 静态语义分析

CodeQL 可对 libass 源码做全程序数据流分析，发现 cppcheck 无法检测的跨函数漏洞模式。

### 9.1 本地运行（无需 GitHub Actions）

```bash
# 安装 CodeQL CLI
cd /workspace/wolfvoid
wget https://github.com/github/codeql-action/releases/latest/download/codeql-bundle-linux64.tar.gz
tar -xzf codeql-bundle-linux64.tar.gz
export PATH=$PATH:/workspace/wolfvoid/codeql

# 创建数据库
cd /workspace/wolfvoid/fuzz_project/libass
make clean
codeql database create ../codeql_db \
    --language=cpp \
    --command="make -j$(nproc)"

# 运行 C/C++ 安全查询套件
codeql database analyze ../codeql_db \
    --format=sarif-latest \
    --output=../codeql_results.sarif \
    codeql/cpp-queries:codeql-suites/cpp-security-and-quality.qls
```

### 9.2 GitHub Actions 方式（备选）

在 libass fork 仓库的 `.github/workflows/codeql.yml` 中配置：

```yaml
name: CodeQL
on: [push]
jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: github/codeql-action/init@v2
        with:
          languages: cpp
      - run: ./autogen.sh && ./configure && make -j4
      - uses: github/codeql-action/analyze@v2
```

### 9.3 重点查询规则
- `cpp/overflow-buffer`：缓冲区溢出
- `cpp/use-after-free`：释放后使用
- `cpp/uncontrolled-format-string`：格式化字符串注入
- `cpp/integer-overflow-tainted`：污点整数溢出

---

## 执行顺序与暂停点汇总

| 阶段 | 关键动作 | 暂停点 | 报告用途 |
|------|---------|--------|---------|
| 一 | cppcheck 扫描 | A | 静态分析截图 |
| 二 | 写入 target.c | B | 手写 driver 截图 |
| 五（启动） | fuzzer 起跑 | C | 运行初始截图 |
| 五（结束） | 12h Summary | D | 覆盖趋势/最终截图 |
| 七 | 结构感知 fuzzer | E | 覆盖率对比截图 |
| 八 | UBSan 联合插桩 | F | UBSan 告警截图 |
| 九 | CodeQL 分析 | G | SARIF 结果截图 |

---

## 当前执行状态

- [x] 环境确认（clang-10, cppcheck, 依赖库）
- [x] 阶段一：克隆源码 + 静态分析
- [x] 阶段二：写入 target.c driver
- [x] 阶段三：编译插桩
- [x] 阶段四：构建种子语料库
- [x] 阶段五：12h fuzzing（已完成，fuzz_run3.log，约 11h 54min）
- [x] 阶段六：崩溃分析报告（见 REPORT_ROUND1.md）
- [x] 阶段七：结构感知 Fuzzing（已完成，发现内存泄漏 CWE-401）
- [x] 阶段八：UBSan 联合插桩（已完成，发现整数溢出 CWE-190 x3）
- [ ] 阶段九：CodeQL 静态语义分析
- [ ] 阶段十：专项攻击整数溢出点（极端坐标 + \p 矢量路径深挖）
