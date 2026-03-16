/*
 * dataloader.c — mmap-based binary data loader
 *
 * MIT License — see COPYING
 */

#define _POSIX_C_SOURCE 200809L

#include "dataloader.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int dataloader_init(DataLoader *dl, const char *filename,
                    int batch_size, int seq_len)
{
    memset(dl, 0, sizeof(*dl));
    dl->batch_size = batch_size;
    dl->seq_len    = seq_len;

    dl->filename = strdup(filename);
    dl->fd = open(filename, O_RDONLY);
    if (dl->fd < 0) { perror(filename); return -1; }

    struct stat st;
    if (fstat(dl->fd, &st) < 0) { perror("fstat"); close(dl->fd); return -1; }

    size_t file_size = (size_t)st.st_size;
    dl->n_tokens = file_size / sizeof(uint16_t);

    if (dl->n_tokens < (size_t)(batch_size * seq_len + 1)) {
        fprintf(stderr, "dataloader: file too small (%zu tokens, need %d)\n",
                dl->n_tokens, batch_size * seq_len + 1);
        close(dl->fd);
        return -1;
    }

    dl->data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, dl->fd, 0);
    if (dl->data == MAP_FAILED) {
        perror("mmap"); close(dl->fd); return -1;
    }

    /* Allocate batch buffers */
    dl->inputs  = malloc((size_t)batch_size * seq_len * sizeof(int));
    dl->targets = malloc((size_t)batch_size * seq_len * sizeof(int));
    if (!dl->inputs || !dl->targets) {
        fprintf(stderr, "dataloader: OOM\n");
        munmap(dl->data, file_size);
        close(dl->fd);
        return -1;
    }

    dl->pos = 0;
    return 0;
}

void dataloader_next_batch(DataLoader *dl)
{
    int B = dl->batch_size;
    int T = dl->seq_len;

    /* Fill B sequences of length T+1 (last token is target) */
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            size_t idx = (dl->pos + (size_t)(b * T + t)) % dl->n_tokens;
            dl->inputs[b * T + t] = (int)dl->data[idx];
        }
        for (int t = 0; t < T; t++) {
            size_t idx = (dl->pos + (size_t)(b * T + t + 1)) % dl->n_tokens;
            dl->targets[b * T + t] = (int)dl->data[idx];
        }
    }

    dl->pos += (size_t)(B * T);
    if (dl->pos + (size_t)(B * T + 1) > dl->n_tokens)
        dl->pos = 0;
}

void dataloader_reset(DataLoader *dl)
{
    dl->pos = 0;
}

void dataloader_free(DataLoader *dl)
{
    if (dl->data && dl->data != MAP_FAILED) {
        struct stat st;
        if (dl->fd >= 0 && fstat(dl->fd, &st) == 0)
            munmap(dl->data, (size_t)st.st_size);
    }
    if (dl->fd >= 0) close(dl->fd);
    free(dl->filename);
    free(dl->inputs);
    free(dl->targets);
    memset(dl, 0, sizeof(*dl));
}

size_t dataloader_num_batches(const DataLoader *dl)
{
    return dl->n_tokens / (size_t)(dl->batch_size * dl->seq_len);
}
