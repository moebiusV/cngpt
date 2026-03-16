## src/Makefile.am — build cngpt binary
##
## MIT License — see COPYING

AM_CFLAGS = -std=c11 $(WARN_FLAGS) $(BLAS_CFLAGS)

bin_PROGRAMS = cngpt

cngpt_SOURCES = \
    cngpt.c \
    model.c \
    model.h \
    ops.c \
    ops.h \
    dataloader.c \
    dataloader.h \
    tokenizer.c \
    tokenizer.h

cngpt_LDADD = $(BLAS_LIBS) -lm -lpthread
