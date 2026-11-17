#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "libass/ass.h"

static ASS_Library  *library  = NULL;
static ASS_Renderer *renderer = NULL;

static void msg_cb(int level, const char *fmt, va_list va, void *data)
{
    (void)level; (void)fmt; (void)va; (void)data;
}

static int init_globals(void)
{
    if (library)
        return 1;

    library = ass_library_init();
    if (!library)
        return 0;

    ass_set_message_cb(library, msg_cb, NULL);

    renderer = ass_renderer_init(library);
    if (!renderer) {
        ass_library_done(library);
        library = NULL;
        return 0;
    }

    ass_set_frame_size(renderer, 1920, 1080);
    ass_set_storage_size(renderer, 1920, 1080);
    ass_set_fonts(renderer, NULL, "sans-serif",
                  ASS_FONTPROVIDER_AUTODETECT, NULL, 1);
    return 1;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 16 || size > (1 << 20))
        return 0;

    if (!init_globals())
        return 0;

    ASS_Track *track = ass_read_memory(library, (char *)data, size, NULL);
    if (!track)
        return 0;

    int changed;
    for (int i = 0; i < track->n_events; i++) {
        ASS_Event *ev = &track->events[i];
        ass_render_frame(renderer, track, ev->Start, &changed);
        if (ev->Duration > 1) {
            ass_render_frame(renderer, track,
                             ev->Start + ev->Duration / 2, &changed);
            ass_render_frame(renderer, track,
                             ev->Start + ev->Duration - 1, &changed);
        }
    }

    ass_render_frame(renderer, track,    0, &changed);
    ass_render_frame(renderer, track, 2000, &changed);
    ass_render_frame(renderer, track, 5000, &changed);

    ass_free_track(track);
    return 0;
}
