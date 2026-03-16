## tests/Makefile.am — unit tests and benchmark harness
##
## MIT License — see COPYING

AM_CFLAGS = -std=c11 $(WARN_FLAGS) $(BLAS_CFLAGS)

check_PROGRAMS = check_ops check_grad

check_ops_SOURCES = \
    check_ops.c \
    ../src/ops.c \
    ../src/ops.h

check_ops_LDADD = $(BLAS_LIBS) -lm

check_grad_SOURCES = \
    check_grad.c \
    ../src/model.c \
    ../src/model.h \
    ../src/ops.c \
    ../src/ops.h \
    ../src/dataloader.c \
    ../src/dataloader.h \
    ../src/tokenizer.c \
    ../src/tokenizer.h

check_grad_LDADD = $(BLAS_LIBS) -lm -lpthread

TESTS = check_ops check_grad

## bench target: compiler-flag sweep (requires weights file)
## Usage: make bench WEIGHTS=gpt2.bin
.PHONY: bench
bench:
	@if [ -z "$(WEIGHTS)" ]; then \
	    echo "Usage: make bench WEIGHTS=<path/to/weights.bin>"; \
	    exit 1; \
	fi
	bash $(srcdir)/bench.sh --weights=$(WEIGHTS)

EXTRA_DIST = bench.sh check_grad.c
