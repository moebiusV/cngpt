/*
 * tokenizer.c — GPT-2 BPE decode (inference display only)
 *
 * When no vocab file is provided, falls back to printing token ids.
 * A proper vocab file can be exported with scripts/export_weights.py.
 *
 * MIT License — see COPYING
 */

#define _POSIX_C_SOURCE 200809L

#include "tokenizer.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GPT-2 byte-to-unicode mapping.
 * Characters 33–126 and 161–172 and 174–255 are printable ASCII/Latin-1.
 * The remaining 256 - 188 = 68 get mapped to extra Unicode code points. */

static void decode_gpt2_byte_token(const char *piece, char *out, size_t out_sz)
{
    /* GPT-2 encodes raw bytes as single-char tokens using a specific
     * byte → unicode mapping.  For terminal display we just print the
     * printable subset and replace the rest with '.'. */
    size_t i = 0, j = 0;
    while (piece[i] && j + 1 < out_sz) {
        unsigned char c = (unsigned char)piece[i];
        if (c >= 32 && c < 127)
            out[j++] = (char)c;
        else
            out[j++] = '.';
        i++;
    }
    out[j] = '\0';
}

int tokenizer_init(Tokenizer *tok, const char *vocab_file)
{
    memset(tok, 0, sizeof(*tok));
    tok->eot_token = 50256;   /* GPT-2 <|endoftext|> */

    if (!vocab_file) {
        /* Byte-level fallback: create 256 single-byte tokens */
        tok->vocab_size = 256;
        tok->vocab = calloc(256, sizeof(char *));
        if (!tok->vocab) return -1;
        for (int i = 0; i < 256; i++) {
            tok->vocab[i] = malloc(8);
            if (!tok->vocab[i]) return -1;
            snprintf(tok->vocab[i], 8, "<%d>", i);
        }
        return 0;
    }

    FILE *f = fopen(vocab_file, "r");
    if (!f) { perror(vocab_file); return -1; }

    /* Count lines */
    int lines = 0;
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) lines++;
    rewind(f);

    tok->vocab_size = lines;
    tok->vocab = calloc((size_t)lines, sizeof(char *));
    if (!tok->vocab) { fclose(f); return -1; }

    for (int i = 0; i < lines; i++) {
        if (!fgets(buf, sizeof(buf), f)) break;
        /* Strip trailing newline */
        size_t len = strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r'))
            buf[--len] = '\0';
        tok->vocab[i] = strdup(buf);
    }
    fclose(f);
    return 0;
}

void tokenizer_decode_token(const Tokenizer *tok, int token_id)
{
    if (!tok->vocab || token_id < 0 || token_id >= tok->vocab_size) {
        printf("[%d]", token_id);
        return;
    }
    if (token_id == tok->eot_token) {
        printf("<|endoftext|>");
        return;
    }
    const char *piece = tok->vocab[token_id];
    /* Handle Ġ (U+0120, GPT-2 space prefix) → space */
    if ((unsigned char)piece[0] == 0xC4 && (unsigned char)piece[1] == 0xA0) {
        printf(" %s", piece + 2);
    } else if (strncmp(piece, "Ċ", 2) == 0) {
        /* Ċ = U+010A = newline */
        printf("\n");
    } else {
        char tmp[512];
        decode_gpt2_byte_token(piece, tmp, sizeof(tmp));
        printf("%s", tmp[0] ? tmp : piece);
    }
}

char *tokenizer_decode(const Tokenizer *tok, const int *tokens, int n_tokens)
{
    /* Rough upper bound: 10 bytes per token */
    size_t buf_sz = (size_t)n_tokens * 16 + 1;
    char *out = malloc(buf_sz);
    if (!out) return NULL;
    size_t pos = 0;

    for (int i = 0; i < n_tokens; i++) {
        int id = tokens[i];
        if (!tok->vocab || id < 0 || id >= tok->vocab_size) {
            int written = snprintf(out + pos, buf_sz - pos, "[%d]", id);
            if (written > 0) pos += (size_t)written;
        } else {
            const char *piece = tok->vocab[id];
            size_t plen = strlen(piece);
            if (pos + plen + 1 >= buf_sz) {
                buf_sz = (pos + plen + 1) * 2;
                char *tmp = realloc(out, buf_sz);
                if (!tmp) { free(out); return NULL; }
                out = tmp;
            }
            memcpy(out + pos, piece, plen);
            pos += plen;
        }
    }
    out[pos] = '\0';
    return out;
}

void tokenizer_free(Tokenizer *tok)
{
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++)
            free(tok->vocab[i]);
        free(tok->vocab);
    }
    memset(tok, 0, sizeof(*tok));
}
