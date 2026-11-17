#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

static unsigned int rng_state = 0;

static unsigned int rng_next(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static int rng_range(int lo, int hi)
{
    return lo + (int)(rng_next() % (unsigned int)(hi - lo + 1));
}

static const char *FONTNAMES[] = {
    "Arial", "sans-serif", "serif", "monospace",
    "DFKaiShu", "MingLi43", "DLCKaiMedium", "DLCHayBold",
    "DLCLiShu", "DLCRoundBold", "HuaTianKaiTi",
    "Times New Roman", "Courier New", "Verdana",
};
#define FONTNAMES_N (int)(sizeof(FONTNAMES)/sizeof(FONTNAMES[0]))

static const char *OVERRIDE_TAGS[] = {
    "",
    "{\\an5}",
    "{\\an%d}",
    "{\\fs%d}",
    "{\\fscx%d\\fscy%d}",
    "{\\pos(%d,%d)}",
    "{\\move(%d,%d,%d,%d)}",
    "{\\move(%d,%d,%d,%d,%d,%d)}",
    "{\\fad(%d,%d)}",
    "{\\fade(%d,%d,%d,%d,%d,%d,%d)}",
    "{\\clip(%d,%d,%d,%d)}",
    "{\\iclip(%d,%d,%d,%d)}",
    "{\\t(%d,%d,\\fscx%d)}",
    "{\\t(\\fs%d)}",
    "{\\t(%d,%d,\\alpha&H%02X&)}",
    "{\\blur%d}",
    "{\\be%d}",
    "{\\bord%d}",
    "{\\shad%d}",
    "{\\c&H%06X&}",
    "{\\1c&H%06X&}",
    "{\\3c&H%06X&}",
    "{\\alpha&H%02X&}",
    "{\\r}",
    "{\\k%d}",
    "{\\kf%d}",
    "{\\ko%d}",
};
#define TAGS_N (int)(sizeof(OVERRIDE_TAGS)/sizeof(OVERRIDE_TAGS[0]))

#define OUTLINE_MAX 268435455

static const int EXTREME_COORDS[] = {
    0, 1, -1, 64, -64,
    OUTLINE_MAX, -OUTLINE_MAX,
    OUTLINE_MAX / 2, -OUTLINE_MAX / 2,
    OUTLINE_MAX - 1, -(OUTLINE_MAX - 1),
    100, 1000, 10000, 100000,
};
#define EXTREME_COORDS_N (int)(sizeof(EXTREME_COORDS)/sizeof(EXTREME_COORDS[0]))

static const double EXTREME_FLOATS[] = {
    0.0, 1.0, -1.0,
    33554431.0,   -33554431.0,
    33554432.0,   -33554432.0,
    1e9, -1e9, 1e10, -1e10,
    1e18, -1e18, 1e19, -1e19,
    2147483647.0, -2147483648.0,
};
#define EXTREME_FLOATS_N (int)(sizeof(EXTREME_FLOATS)/sizeof(EXTREME_FLOATS[0]))

static int rand_time_ms(void)
{
    return rng_range(0, 3600) * 1000 + rng_range(0, 999);
}

static void fmt_time(char *buf, int ms)
{
    int h = ms / 3600000; ms %= 3600000;
    int m = ms / 60000;   ms %= 60000;
    int s = ms / 1000;    ms %= 1000;
    int cs = ms / 10;
    snprintf(buf, 16, "%d:%02d:%02d.%02d", h, m, s, cs);
}

static size_t build_drawing_path(char *buf, size_t bsz)
{
    int scenario = rng_range(0, 7);
    size_t pos = 0;

    int scale = rng_range(1, 4);

    switch (scenario) {
    case 0: {
        /* 场景 A1：bbox 极端差值，x_max - x_min 接近 INT32_MAX */
        int x1 = -OUTLINE_MAX;
        int x2 =  OUTLINE_MAX;
        int y1 = rng_range(-OUTLINE_MAX, 0);
        int y2 = rng_range(0, OUTLINE_MAX);
        pos += snprintf(buf + pos, bsz - pos,
            "{\\p%d}m %d %d l %d %d{\\p0}", scale, x1, y1, x2, y2);
        break;
    }
    case 1: {
        /* 场景 A2：多段线，累积 bbox 覆盖整个 int32 范围 */
        pos += snprintf(buf + pos, bsz - pos,
            "{\\p1}m %d %d l %d %d l %d %d l %d %d{\\p0}",
            -OUTLINE_MAX, -OUTLINE_MAX,
             OUTLINE_MAX,  OUTLINE_MAX,
            -OUTLINE_MAX,  OUTLINE_MAX,
             OUTLINE_MAX, -OUTLINE_MAX);
        break;
    }
    case 2: {
        /* 场景 B1：超大浮点，触发 double_to_d6() UB */
        int fi = rng_range(10, EXTREME_FLOATS_N - 1);
        double fx = EXTREME_FLOATS[fi];
        double fy = EXTREME_FLOATS[rng_range(10, EXTREME_FLOATS_N - 1)];
        pos += snprintf(buf + pos, bsz - pos,
            "{\\p1}m %.0f %.0f l 0 0{\\p0}", fx, fy);
        break;
    }
    case 3: {
        /* 场景 B2：混合极端浮点，x_max 超大、x_min 超小 */
        pos += snprintf(buf + pos, bsz - pos,
            "{\\p1}m %.0f 0 m %.0f 0{\\p0}", 1e18, -1e18);
        break;
    }
    case 4: {
        /* 场景 C1：仅 m 命令，x_max < x_min，v->advance 为负 */
        /* 结合 \bord 触发 opaque box 负 width 路径 */
        int bord = rng_range(1, 20);
        pos += snprintf(buf + pos, bsz - pos,
            "{\\bord%d\\3c&H000000&\\p1}m %d 0 m %d 0{\\p0}",
            bord, OUTLINE_MAX, -OUTLINE_MAX);
        break;
    }
    case 5: {
        /* 场景 C2：极端 y 方向，触发 v->asc 溢出 */
        int bord = rng_range(1, 20);
        pos += snprintf(buf + pos, bsz - pos,
            "{\\bord%d\\p1}m 0 %d m 0 %d{\\p0}",
            bord, OUTLINE_MAX, -OUTLINE_MAX);
        break;
    }
    case 6: {
        /* 场景 D1：\t 动画 + 极大缩放 + \p，放大渲染压力 */
        int scx = rng_range(1000, 9999);
        int scy = rng_range(1000, 9999);
        int dur = rng_range(100, 5000);
        pos += snprintf(buf + pos, bsz - pos,
            "{\\fscx%d\\fscy%d\\t(0,%d,\\fscx1\\fscy1)\\p1}"
            "m 0 0 l %d 0 %d %d 0 %d{\\p0}",
            scx, scy, dur,
            rng_range(100, 10000),
            rng_range(100, 10000), rng_range(100, 10000),
            rng_range(100, 10000));
        break;
    }
    case 7: {
        /* 场景 D2：\clip 矢量裁剪 + 极端坐标 \p 组合 */
        int cx1 = EXTREME_COORDS[rng_range(0, EXTREME_COORDS_N - 1)];
        int cx2 = EXTREME_COORDS[rng_range(0, EXTREME_COORDS_N - 1)];
        int cy1 = EXTREME_COORDS[rng_range(0, EXTREME_COORDS_N - 1)];
        int cy2 = EXTREME_COORDS[rng_range(0, EXTREME_COORDS_N - 1)];
        pos += snprintf(buf + pos, bsz - pos,
            "{\\clip(%d,%d,%d,%d)\\p%d}m %d %d l %d %d b %d %d %d %d %d %d{\\p0}",
            cx1, cy1, cx2, cy2, scale,
            rng_range(-OUTLINE_MAX, 0),  rng_range(-OUTLINE_MAX, 0),
            rng_range(0, OUTLINE_MAX),   rng_range(0, OUTLINE_MAX),
            rng_range(-OUTLINE_MAX, OUTLINE_MAX), rng_range(-OUTLINE_MAX, OUTLINE_MAX),
            rng_range(-OUTLINE_MAX, OUTLINE_MAX), rng_range(-OUTLINE_MAX, OUTLINE_MAX),
            rng_range(0, OUTLINE_MAX),   rng_range(0, OUTLINE_MAX));
        break;
    }
    default:
        pos += snprintf(buf + pos, bsz - pos,
            "{\\p1}m 0 0 l 100 0 100 100 0 100{\\p0}");
        break;
    }

    return pos;
}

static char *build_override_tag(char *buf, size_t bsz)
{
    int idx = rng_range(0, TAGS_N - 1);
    const char *tmpl = OVERRIDE_TAGS[idx];

    int a = rng_range(1, 9);
    int b = rng_range(0, 2000);
    int c = rng_range(0, 2000);
    int d = rng_range(0, 1920);
    int e = rng_range(0, 1080);
    int f = rng_range(0, 1920);
    int g = rng_range(0, 1080);
    int col = rng_range(0, 0xFFFFFF);
    int alpha = rng_range(0, 0xFF);
    int dur1 = rng_range(0, 5000);
    int dur2 = rng_range(0, 5000);

    if (strstr(tmpl, "%d,%d,%d,%d,%d,%d,%d")) {
        snprintf(buf, bsz, tmpl, alpha, alpha, alpha, 0, dur1, 5000-dur1, 5000);
    } else if (strstr(tmpl, "%d,%d,%d,%d,%d,%d")) {
        snprintf(buf, bsz, tmpl, d, e, f, g, dur1, dur2);
    } else if (strstr(tmpl, "%d,%d,%d,%d")) {
        snprintf(buf, bsz, tmpl, d, e, f, g);
    } else if (strstr(tmpl, "\\t(%d,%d,\\alpha")) {
        snprintf(buf, bsz, tmpl, dur1, dur2, alpha);
    } else if (strstr(tmpl, "\\t(%d,%d,\\fscx")) {
        snprintf(buf, bsz, tmpl, dur1, dur2, b);
    } else if (strstr(tmpl, "\\t(\\fs")) {
        snprintf(buf, bsz, tmpl, b);
    } else if (strstr(tmpl, "\\fscx%d\\fscy%d")) {
        snprintf(buf, bsz, tmpl, b, c);
    } else if (strstr(tmpl, "\\pos(%d,%d)")) {
        snprintf(buf, bsz, tmpl, d, e);
    } else if (strstr(tmpl, "\\move(%d,%d,%d,%d)")) {
        snprintf(buf, bsz, tmpl, 0, 0, f, g);
    } else if (strstr(tmpl, "\\an%d")) {
        snprintf(buf, bsz, tmpl, a);
    } else if (strstr(tmpl, "%06X")) {
        snprintf(buf, bsz, tmpl, col);
    } else if (strstr(tmpl, "%02X")) {
        snprintf(buf, bsz, tmpl, alpha);
    } else if (strstr(tmpl, "%d")) {
        snprintf(buf, bsz, tmpl, b);
    } else {
        snprintf(buf, bsz, "%s", tmpl);
    }
    return buf;
}

static size_t build_ass(char *out, size_t outsz)
{
    int play_x = rng_range(320, 3840);
    int play_y = rng_range(240, 2160);
    const char *font = FONTNAMES[rng_range(0, FONTNAMES_N - 1)];
    int fontsize = rng_range(8, 200);
    int bold = rng_range(0, 1);
    int italic = rng_range(0, 1);
    int align = rng_range(1, 9);
    int border = rng_range(0, 4);
    int shadow = rng_range(0, 4);
    int scalex = rng_range(50, 300);
    int scaley = rng_range(50, 300);
    int spacing = rng_range(0, 10);
    int primary = rng_range(0, 0xFFFFFF);
    int secondary = rng_range(0, 0xFFFFFF);
    int outline_c = rng_range(0, 0xFFFFFF);
    int back_c = rng_range(0, 0xFFFFFF);
    int wrap_style = rng_range(0, 3);

    size_t pos = 0;

    pos += snprintf(out + pos, outsz - pos,
        "[Script Info]\r\n"
        "ScriptType: v4.00+\r\n"
        "PlayResX: %d\r\n"
        "PlayResY: %d\r\n"
        "WrapStyle: %d\r\n"
        "ScaledBorderAndShadow: yes\r\n"
        "\r\n"
        "[V4+ Styles]\r\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
        " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
        " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
        " Alignment, MarginL, MarginR, MarginV, Encoding\r\n"
        "Style: Default,%s,%d,&H00%06X,&H00%06X,&H00%06X,&H00%06X,"
        "%d,%d,0,0,%d,%d,%d,0,%d,%d,%d,%d,10,10,10,1\r\n"
        "\r\n"
        "[Events]\r\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\r\n",
        play_x, play_y, wrap_style,
        font, fontsize,
        primary, secondary, outline_c, back_c,
        bold, italic,
        scalex, scaley, spacing,
        border, border, shadow, align
    );

    int n_events = rng_range(1, 6);
    for (int i = 0; i < n_events && pos < outsz - 512; i++) {
        int start_ms = rand_time_ms();
        int dur_ms = rng_range(100, 10000);
        int end_ms = start_ms + dur_ms;

        char tstart[16], tend[16];
        fmt_time(tstart, start_ms);
        fmt_time(tend, end_ms);

        int layer = rng_range(0, 5);

        int use_drawing = (rng_range(0, 9) < 7);

        if (use_drawing) {
            char drawing[512];
            build_drawing_path(drawing, sizeof(drawing));
            pos += snprintf(out + pos, outsz - pos,
                "Dialogue: %d,%s,%s,Default,,%d,%d,%d,,%s\r\n",
                layer, tstart, tend,
                rng_range(0, 100), rng_range(0, 100), rng_range(0, 100),
                drawing);
        } else {
            char tag[256];
            build_override_tag(tag, sizeof(tag));

            static const char *TEXTS[] = {
                "Hello fuzzer!", "Test", "libass", "A",
                "The quick brown fox", "12345",
                "!@#$%^&*()",
            };
            const char *text = TEXTS[rng_range(0, 6)];

            pos += snprintf(out + pos, outsz - pos,
                "Dialogue: %d,%s,%s,Default,,%d,%d,%d,,%s%s\r\n",
                layer, tstart, tend,
                rng_range(0, 100), rng_range(0, 100), rng_range(0, 100),
                tag, text);
        }
    }

    return pos;
}

size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size,
                                size_t max_size, unsigned int seed)
{
    rng_state = seed ? seed : (unsigned int)time(NULL);
    if (rng_state == 0) rng_state = 1;

    int use_structured = (rng_range(0, 11) >= 2);

    if (use_structured || size < 16) {
        size_t built = build_ass((char *)data, max_size);
        return built > 0 ? built : size;
    }

    size_t new_size = size;
    int op = rng_range(0, 4);
    if (op == 0 && size > 1) {
        size_t pos = rng_next() % size;
        data[pos] = (uint8_t)(rng_next() & 0xFF);
    } else if (op == 1 && size < max_size) {
        size_t pos = rng_next() % size;
        memmove(data + pos + 1, data + pos, size - pos);
        data[pos] = (uint8_t)(rng_next() & 0xFF);
        new_size = size + 1;
    } else if (op == 2 && size > 1) {
        size_t pos = rng_next() % (size - 1);
        memmove(data + pos, data + pos + 1, size - pos - 1);
        new_size = size - 1;
    } else if (op == 3) {
        char drawing[512];
        build_drawing_path(drawing, sizeof(drawing));
        size_t tlen = strlen(drawing);
        if (size + tlen < max_size) {
            size_t pos = rng_next() % size;
            memmove(data + pos + tlen, data + pos, size - pos);
            memcpy(data + pos, drawing, tlen);
            new_size = size + tlen;
        }
    } else {
        size_t built = build_ass((char *)data, max_size);
        new_size = built > 0 ? built : size;
    }

    return new_size;
}
