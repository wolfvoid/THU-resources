# libass Fuzzing Bug Report

## Bug 1: Out-of-Memory (OOM) via Bitmap Over-Allocation

**类型**: 内存耗尽 (Denial of Service / OOM)
**严重程度**: 高
**发现方式**: libFuzzer + ASan，运行约 30 分钟后触发
**触发文件**: `crashes/oom-9812c17fc8a541305a6d65cfca282e992c868990`

### 根因分析

触发输入为包含 `\fscx200\fscy200`（极端字体缩放）与 `\t(\fscx200)`（动画变换标签）组合的 ASS 字幕。

**调用链**:
```
LLVMFuzzerTestOneInput (target.c:54)
  → ass_render_frame (ass_render.c:3399)
    → ass_render_event (ass_render.c:3017)
      → render_and_combine_glyphs (ass_render.c:2571)
        → get_bitmap_glyph (ass_render.c:1412 / 1536)
          → ass_bitmap_construct (ass_render.c:1567)
            → ass_cache_get (ass_cache.c:424)
              → ass_alloc_bitmap (ass_bitmap.c:111)
                → ass_aligned_alloc (ass_utils.c:65)  ← 无限分配
```

**问题位置**: `libass/ass_bitmap.c:111` — `ass_alloc_bitmap()` 在极端缩放参数下未做 bitmap 尺寸上限检查，导致单次渲染分配超过 2.5GB 的 bitmap 内存（共 14 次分配合计 ~4.9GB）。

**复现 PoC**:
```ass
[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, ...
Style: Default,Arial,36,&H00FFFFFF,...

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:15.00,0:00:20.00,Default,,0,0,0,,{\an5\fscx200\fscy200}Scaled
Dialogue: 0,0:00:20.00,0:00:25.00,Default,,0,0,0,,{\t(\fscx200)}Animated scale
```

**危害**: 攻击者提供恶意 ASS 字幕文件，可触发播放器/渲染器 OOM crash（DoS）。

**已提交 Issue**: https://github.com/libass/libass/issues/928

---

## Bug 2: Slow Processing (Performance Vulnerability)

**类型**: 算法复杂度漏洞（潜在 DoS）
**触发文件**: 3 个 slow-unit（>60s 单次处理时间）

同样由极端缩放+动画标签组合触发，渲染时间超过 60 秒/帧，可用于 CPU 耗尽攻击。

---

## 静态分析发现的额外问题（cppcheck）

| 文件 | 行号 | 类型 | 描述 |
|------|------|------|------|
| `libass/ass_font.c` | 414 | memleak | `fs` 指针内存泄漏 |
| `libass/ass_render.c` | 535-536 | nullPointer | `offset` 可能为空指针解引用 |
| `libass/ass_directwrite.c` | 50 | ctunullpointer | `shared_hdc` 空指针解引用 |

---

## 统计数据

| 指标 | 值 |
|------|----|
| 运行时间 | ~30 分钟（因 OOM 中止） |
| 执行次数 | 15,578 次 |
| 平均 exec/s | 7 |
| 代码覆盖 (cov) | 1640 → 2180 (+33%) |
| 新增语料 | 1,021 个 |
| 发现崩溃 | 1 个 OOM + 3 个 slow-unit |
| peak_rss | 5,011 MB |
