/*
 * dataloader.h — mmap-based binary data loader for cngpt
 *
 * Data file format (produced by prepare_data.py or similar):
 *   All tokens stored as uint16_t in row-major order, no header.
 *   A separate .bin file contains the token ids sequentially.
 *
 * MIT License — see COPYING
 */

#ifndef CNGPT_DATALOADER_H
#define CNGPT_DATALOADER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    /* Configuration */
    int batch_size;    /* B */
    int seq_len;       /* T */

    /* File state */
    char *filename;
    int   fd;
    uint16_t *data;    /* mmap'd token array */
    size_t    n_tokens;

    /* Batch state */
    size_t  pos;       /* current position in token stream */

    /* Batch buffers (caller-allocated or internally managed) */
    int *inputs;       /* [B, T]   — allocated internally */
    int *targets;      /* [B, T]   — allocated internally */
} DataLoader;

/* Open data file and mmap it. Returns 0 on success. */
int  dataloader_init(DataLoader *dl, const char *filename,
                     int batch_size, int seq_len);

/* Advance to next batch. Wraps around end of data.
 * Fills dl->inputs and dl->targets. */
void dataloader_next_batch(DataLoader *dl);

/* Reset position to start. */
void dataloader_reset(DataLoader *dl);

/* Release resources. */
void dataloader_free(DataLoader *dl);

/* Total number of batches in one epoch. */
size_t dataloader_num_batches(const DataLoader *dl);

#endif /* CNGPT_DATALOADER_H */
