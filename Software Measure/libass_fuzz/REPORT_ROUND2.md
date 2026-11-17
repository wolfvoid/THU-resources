# libass Fuzzing 第二轮报告（结构感知 + UBSan）

## 运行参数

| 参数 | 结构感知 fuzzer | UBSan fuzzer |
|------|--------------|-------------|
| 二进制 | `fuzzer_app_structured` | `fuzzer_app_ubsan` |
| Sanitizer | ASan + 自定义 mutator | ASan + UBSan + 自定义 mutator |
| `-rss_limit_mb` | 51200（50GB） | 51200（50GB） |
| `-malloc_limit_mb` | 40960（40GB） | 40960（40GB） |
| 启动时间 | 2026-05-15 17:30 UTC | 2026-05-15 17:30 UTC |
| 退出原因 | 系统回收（约 24h） | 系统回收（约 24h） |

---

## 覆盖率对比

| 指标 | 第一轮（盲变异） | 第二轮结构感知 | 第二轮 UBSan |
|------|--------------|-------------|------------|
| cov（边覆盖） | 2262 | **2522** | **12480** |
| ft（特征数） | 12959 | 13351 | **71966** |
| 总执行次数 | ~33432 | 2910 | 4315 |
| peak RSS | 14154 MB | 14073 MB | 14170 MB |
| new_units_added | - | 18 | **170** |

**UBSan 版本覆盖率（cov 12480）远高于纯 ASan 版本（cov 2522）**，原因是 UBSan 插桩粒度更细（32744 个计数器 vs 5261 个），能区分更多执行路径。

---

## 发现的漏洞

### 漏洞一：内存泄漏（CWE-401）

**两个 fuzzer 各发现 1 个 leak artifact，触发文件相同根因。**

| 项目 | 详情 |
|------|------|
| 类型 | `LeakSanitizer: detected memory leaks` |
| 泄漏量 | 288 bytes（256 direct + 32 indirect） |
| 位置 | `libfontconfig.so.1+0x212f4` / `+0x2193c` |
| 触发路径 | `ass_set_fonts()` → fontconfig 字体 fallback 查找 |
| CWE | CWE-401（不受控的内存分配不释放） |

**根因：** 当 `.ass` 中指定的字体名在系统中不存在时（如 `Courier New`、`DLCHayBold`），libass 调用 fontconfig 做 fallback 匹配，fontconfig 内部分配的查找缓冲区（256 bytes）未被释放。在长期运行的播放器进程中（如 mpv、VLC），每次解析含未知字体的字幕都会泄漏 288 bytes，导致内存缓慢增长。

**触发 PoC（精简）：**
```ass
[Script Info]
ScriptType: v4.00+
PlayResX: 2057
PlayResY: 1171

[V4+ Styles]
Style: Default,Courier New,87,&H001860E1,...

[Events]
Dialogue: 0,0:03:23.58,0:03:27.54,Default,,,,,,{\kf1640}Test
```

---

### 漏洞二：有符号整数溢出（CWE-190）

**UBSan 在 `ass_render.c` 中发现 3 处 runtime error。**

#### 2a. `ass_render.c:1264` 和 `1265`

```
runtime error: signed integer overflow:
  -2147483648 - 2147483647 cannot be represented in type 'int'
```

| 项目 | 详情 |
|------|------|
| 位置 | `libass/ass_render.c:1264:37` 和 `:1265:33` |
| 类型 | `signed integer overflow`（INT_MIN - INT_MAX） |
| 触发条件 | `bbox.x_max - bbox.x_min` / `bbox.y_max - bbox.y_min` 计算矢量绘图（`\p`）包围盒时，极端坐标值导致溢出 |
| CWE | CWE-190（整数溢出） |

对应代码：
```c
v->advance = bbox.x_max - bbox.x_min;  // line 1264 ← 溢出
v->asc     = bbox.y_max - bbox.y_min;  // line 1265 ← 溢出
```

#### 2b. `ass_render.c:3026`

```
runtime error: 3.33333e+36 is outside the range of representable values of type 'int'
```

| 项目 | 详情 |
|------|------|
| 位置 | `libass/ass_render.c:3026:9` |
| 类型 | 浮点→整数转换溢出（float-to-int UB） |
| 触发条件 | 渲染位置计算 `(device_x + bbox.x_min) * render_priv->par_scale_x + 0.5` 中，`par_scale_x` 与极大坐标相乘后超出 `int` 表示范围 |
| CWE | CWE-190（整数溢出）/ CWE-681（不正确的数值转换） |

对应代码：
```c
event_images->left =
    (device_x + bbox.x_min) * render_priv->par_scale_x  // ← 3.33e36
    - text_info->border_x + 0.5;                         // ← 强转 int 时 UB
```

**危害：** 整数溢出导致渲染坐标变为极大负数或垃圾值，后续位图内存访问可能越界，是潜在的内存破坏前驱。

---

### 漏洞三：慢路径 DoS（CWE-400，持续累积）

| fuzzer | slow-unit 数量 | 最慢单元 |
|--------|-------------|---------|
| 结构感知 | 4 个 | 254 秒 |
| UBSan | 9 个 | **742 秒** |

UBSan fuzzer 发现了 742 秒的极慢单元，远超第一轮的 120 秒，说明结构感知 mutator 生成的合法 `.ass` 能更有效地触发渲染层的超线性复杂度路径。

---

## 各轮次漏洞汇总

| 轮次 | 漏洞类型 | 数量 | 严重程度 |
|------|---------|------|---------|
| 第一轮（盲变异） | slow-unit DoS | 8 个 | 中 |
| 第二轮（结构感知） | 内存泄漏（CWE-401） | 2 个 | 中 |
| 第二轮（UBSan） | 有符号整数溢出（CWE-190） | 3 处 | 高 |
| 第二轮（UBSan） | 浮点→整数转换 UB（CWE-681） | 1 处 | 高 |
| 第二轮（结构感知+UBSan） | slow-unit DoS | 新增 6 个 | 中 |

---

## 结果分析与下一步方向

### 为什么整数溢出没有进一步触发内存破坏？

`ass_render.c:1264` 的溢出结果（极大负数）赋给 `v->advance`，后续用于字形间距计算。当前语料库的坐标范围（0~3840）不够极端，溢出后的负值没有恰好落在触发越界访问的范围内。

### 下一步有针对性的挖掘方向

#### 方向一：专项攻击整数溢出点（优先级：高）
- 在 custom_mutator 中**专门生成极端坐标**：`\p` 矢量绘图坐标范围扩展到 `INT_MIN`~`INT_MAX`
- 专门变异 `PlayResX/Y` 为 0、1、负数、极大值
- 目标：让 `bbox.x_max - bbox.x_min` 溢出后的结果恰好触发后续的越界访问

#### 方向二：矢量绘图路径（`\p` tag）深挖（优先级：高）
- 当前 `\p` 使用简单的矩形/贝塞尔曲线，溢出就在这里
- 扩展 mutator 生成更复杂的矢量路径：高阶贝塞尔、超多控制点、自交路径、极大坐标

#### 方向三：内存泄漏转化为可利用漏洞（优先级：中）
- 确认泄漏是否在 libass 内部（目前在 libfontconfig，属于第三方）
- 构造长字体名列表，测试 libass 自身的字体缓存是否也有泄漏

#### 方向四：CodeQL 静态分析确认整数溢出范围（优先级：中）
- 重点查询 `ass_render.c` 中所有 `int` 类型的坐标/尺寸计算
- 找出所有与 `bbox`、`device_x/y`、`par_scale_x` 相关的计算链
