/*
 * tokenizer.h — GPT-2 BPE tokenizer (decode only, for inference display)
 *
 * Loads the GPT-2 tokenizer vocabulary from a simple text file.
 * Encoding is not implemented; the training pipeline uses pre-tokenized data.
 *
 * MIT License — see COPYING
 */

#ifndef CNGPT_TOKENIZER_H
#define CNGPT_TOKENIZER_H

typedef struct {
    char  **vocab;      /* vocab[i] = UTF-8 string for token i */
    int     vocab_size;
    int     eot_token;  /* end-of-text token id (50256 for GPT-2) */
} Tokenizer;

/* Load vocabulary from a simple text file (one token per line, hex-escaped).
 * Returns 0 on success. Pass NULL to use built-in byte-level fallback. */
int  tokenizer_init(Tokenizer *tok, const char *vocab_file);

/* Decode a single token to stdout (handles special tokens). */
void tokenizer_decode_token(const Tokenizer *tok, int token_id);

/* Decode a sequence of tokens into a malloc'd string.
 * Caller must free the result. */
char *tokenizer_decode(const Tokenizer *tok,
                       const int *tokens, int n_tokens);

void tokenizer_free(Tokenizer *tok);

#endif /* CNGPT_TOKENIZER_H */
