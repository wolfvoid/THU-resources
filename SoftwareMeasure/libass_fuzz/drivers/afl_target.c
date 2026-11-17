/*
 * AFL Fuzzing Driver for libass
 *
 * libass 核心 API 调用链（理解待测对象的 API）：
 *
 *   ass_library_init()          -- 初始化全局库实例
 *   ass_set_message_cb()        -- 设置日志回调（抑制杂乱输出）
 *   ass_renderer_init()         -- 初始化渲染器
 *   ass_set_frame_size()        -- 设置输出帧尺寸
 *   ass_set_fonts()             -- 设置字体提供者（触发 fontconfig 路径）
 *   ass_read_memory()           -- 从内存解析 .ass 字幕文件（主攻解析器）
 *   ass_process_data()          -- 流式输入解析（模拟 MKV demuxer 场景）
 *   ass_render_frame()          -- 渲染指定时间点（触发排版/渲染器路径）
 *   ass_free_track()            -- 释放字幕轨道
 *   ass_renderer_done()         -- 释放渲染器
 *   ass_library_done()          -- 释放库实例
 *
 * 选择测试的 API（覆盖 libass 两条独立解析路径）：
 *   Path A: ass_read_memory     -- 一次性解析完整文件（文件读取场景）
 *   Path B: ass_process_data    -- 分块流式输入（流媒体场景）
 *
 * 如何将 AFL 数据灌入 API：
 *   AFL 通过 @@ 将变异后的文件路径传入，driver 读取文件内容后
 *   分别传入上述两条 API，覆盖 libass 解析器和渲染器的完整代码路径。
 *
 * 渲染时间点覆盖策略：
 *   - 每个 event 的 Start / 1/4 / 1/2 / 3/4 / End-1 处各渲染一次
 *   - 额外在固定时间点渲染，覆盖 \t() 动画插值、\fad() 淡入淡出等路径
 *
 * 编译（ASan + AFL 插桩）：
 *   AFL_USE_ASAN=1 afl-clang-fast -g -O1 -fsanitize=address \
 *       afl_target.c -I. ./libass/.libs/libass.a \
 *       -lfreetype -lfontconfig -lfribidi -lharfbuzz -lm \
 *       -o afl_target
 *
 * 运行：
 *   AFL_USE_ASAN=1 afl-fuzz -i afl_in -o afl_out -x ass.dict \
 *       -t 10000 -m none -V 43200 -- ./afl_target @@
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "libass/ass.h"

static void msg_cb(int level, const char *fmt, va_list va, void *data)
{
    (void)level; (void)fmt; (void)va; (void)data;
}

/*
 * 对同一个 track 在多个时间点调用 ass_render_frame，
 * 覆盖 \t() 动画插值、\move() 运动、\fad() 淡入淡出、
 * \clip() 裁剪等时间相关的渲染路径。
 */
static void render_track(ASS_Renderer *renderer, ASS_Track *track)
{
    if (!track || track->n_events == 0)
        return;

    int changed;

    for (int i = 0; i < track->n_events; i++) {
        ASS_Event *ev = &track->events[i];
        long long s = ev->Start;
        long long d = ev->Duration;

        ass_render_frame(renderer, track, s, &changed);
        if (d > 0) {
            ass_render_frame(renderer, track, s + d / 4,     &changed);
            ass_render_frame(renderer, track, s + d / 2,     &changed);
            ass_render_frame(renderer, track, s + d * 3 / 4, &changed);
            ass_render_frame(renderer, track, s + d - 1,     &changed);
        }
    }

    static const long long pts[] = {
        0, 100, 500, 1000, 2000, 3000, 5000, 10000, 30000
    };
    for (int t = 0; t < (int)(sizeof(pts)/sizeof(pts[0])); t++)
        ass_render_frame(renderer, track, pts[t], &changed);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    /* 读取 AFL 传入的变异文件 */
    FILE *f = fopen(argv[1], "rb");
    if (!f)
        return 1;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size < 16 || size > (1 << 20)) {
        fclose(f);
        return 0;
    }

    char *data = (char *)malloc(size);
    if (!data) { fclose(f); return 1; }
    if ((long)fread(data, 1, size, f) != size) {
        free(data); fclose(f); return 1;
    }
    fclose(f);

    /* 初始化 libass 库和渲染器 */
    ASS_Library *library = ass_library_init();
    if (!library) { free(data); return 1; }

    ass_set_message_cb(library, msg_cb, NULL);
    ass_set_extract_fonts(library, 0);

    ASS_Renderer *renderer = ass_renderer_init(library);
    if (!renderer) {
        ass_library_done(library);
        free(data);
        return 1;
    }

    ass_set_frame_size(renderer, 1920, 1080);
    ass_set_storage_size(renderer, 1920, 1080);
    ass_set_fonts(renderer, NULL, "sans-serif",
                  ASS_FONTPROVIDER_AUTODETECT, NULL, 1);

    /*
     * Path A: ass_read_memory
     * 一次性解析完整 .ass 文件，覆盖 [Script Info] / [Styles] /
     * [Events] 三段式解析器，以及所有 Override tag 的解析路径。
     */
    ASS_Track *track_a = ass_read_memory(library, data, size, NULL);
    if (track_a) {
        render_track(renderer, track_a);
        ass_free_track(track_a);
    }

    /*
     * Path B: ass_process_data (流式输入)
     * 模拟 MKV/Matroska demuxer 分块送入字幕数据的场景，
     * 覆盖与 Path A 不同的增量解析代码路径。
     * 将输入切成三段模拟多次 packet 到达。
     */
    ASS_Track *track_b = ass_new_track(library);
    if (track_b) {
        long s1 = size / 3;
        long s2 = size / 3;
        long s3 = size - s1 - s2;
        ass_process_data(track_b, data,        (int)s1);
        ass_process_data(track_b, data + s1,   (int)s2);
        ass_process_data(track_b, data + s1 + s2, (int)s3);
        render_track(renderer, track_b);
        ass_free_track(track_b);
    }

    ass_renderer_done(renderer);
    ass_library_done(library);
    free(data);
    return 0;
}
