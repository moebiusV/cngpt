#!/bin/bash
# bench.sh — compiler-flag sweep benchmark harness for cngpt
#
# Sweeps optimization flags and records tokens/sec for inference.
# Also sweeps OPENBLAS_NUM_THREADS.
#
# Usage: ./tests/bench.sh [--weights=<file>] [--iters=50] [--seq=128]
#
# MIT License — see COPYING

set -euo pipefail

# ---------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------
WEIGHTS=""
ITERS=50
SEQ=128

for arg in "$@"; do
    case "$arg" in
        --weights=*) WEIGHTS="${arg#*=}" ;;
        --iters=*)   ITERS="${arg#*=}" ;;
        --seq=*)     SEQ="${arg#*=}" ;;
    esac
done

if [ -z "$WEIGHTS" ]; then
    echo "Usage: $0 --weights=<file> [--iters=N] [--seq=T]" >&2
    exit 1
fi

if [ ! -f "$WEIGHTS" ]; then
    echo "Error: weights file not found: $WEIGHTS" >&2
    exit 1
fi

# ---------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------
SRC_DIR="$(cd "$(dirname "$0")/../src" && pwd)"
TMP_DIR=$(mktemp -d /tmp/cngpt_bench_XXXXXX)
trap 'rm -rf "$TMP_DIR"' EXIT

SRCS="$SRC_DIR/cngpt.c $SRC_DIR/model.c $SRC_DIR/ops.c \
      $SRC_DIR/dataloader.c $SRC_DIR/tokenizer.c"

BLAS_INC="-I/usr/include/x86_64-linux-gnu"
BLAS_LIB="-L/usr/lib/x86_64-linux-gnu -lopenblas"
COMMON_FLAGS="-std=c11 $BLAS_INC"
LINK_FLAGS="$BLAS_LIB -lm -lpthread"

# Compile with given flags, return binary path
build_binary() {
    local name="$1"
    local flags="$2"
    local bin="$TMP_DIR/cngpt_$name"

    if gcc $COMMON_FLAGS $flags $SRCS -o "$bin" $LINK_FLAGS 2>/dev/null; then
        echo "$bin"
    else
        # Retry with verbose to capture errors silently
        echo ""
    fi
}

# Run bench subcommand, return ms/token
run_bench() {
    local bin="$1"
    local threads="$2"
    if [ -z "$bin" ] || [ ! -x "$bin" ]; then
        echo "-1"
        return
    fi
    local output
    output=$(OPENBLAS_NUM_THREADS="$threads" \
             "$bin" bench --weights="$WEIGHTS" --iters="$ITERS" --seq="$SEQ" 2>/dev/null)
    # Parse "  tok/s      : 123.4"
    echo "$output" | grep "tok/s" | awk '{print $NF}'
}

# ---------------------------------------------------------------
# Flag configurations
# ---------------------------------------------------------------
declare -A FLAG_SETS
FLAG_SETS["BASE"]="-O2"
FLAG_SETS["OPT"]="-O3"
FLAG_SETS["FAST"]="-Ofast"
FLAG_SETS["NATIVE"]="-O3 -march=native"
FLAG_SETS["FMA"]="-O3 -march=native -mfma"
FLAG_SETS["MATH"]="-O3 -march=native -ffast-math"
FLAG_SETS["UNROLL"]="-O3 -march=native -ffast-math -funroll-loops"
FLAG_SETS["LTO"]="-O3 -march=native -ffast-math -flto"
FLAG_SETS["FULL"]="-O3 -march=native -ffast-math -funroll-loops -flto"

# Ordered list for display
FLAG_ORDER=(BASE OPT FAST NATIVE FMA MATH UNROLL LTO FULL)

# ---------------------------------------------------------------
# PGO build (special two-pass)
# ---------------------------------------------------------------
build_pgo() {
    local flags="-O3 -march=native -ffast-math -fprofile-generate"
    local bin_gen="$TMP_DIR/cngpt_pgo_gen"
    local bin_use="$TMP_DIR/cngpt_pgo_use"

    if ! gcc $COMMON_FLAGS $flags $SRCS -o "$bin_gen" $LINK_FLAGS 2>/dev/null; then
        echo ""
        return
    fi

    # Training run for profiling (short, single forward pass)
    OPENBLAS_NUM_THREADS=1 \
    "$bin_gen" bench --weights="$WEIGHTS" --iters=5 --seq="$SEQ" \
        > /dev/null 2>&1 || true

    local use_flags="-O3 -march=native -ffast-math -fprofile-use -fprofile-correction"
    if gcc $COMMON_FLAGS $use_flags $SRCS -o "$bin_use" $LINK_FLAGS 2>/dev/null; then
        echo "$bin_use"
    else
        echo ""
    fi
}

# ---------------------------------------------------------------
# Compile all flag sets
# ---------------------------------------------------------------
echo "Compiling binaries..."
declare -A BINS

for name in "${FLAG_ORDER[@]}"; do
    flags="${FLAG_SETS[$name]}"
    bin=$(build_binary "$name" "$flags")
    BINS["$name"]="$bin"
    if [ -n "$bin" ]; then
        printf "  %-8s OK\n" "$name"
    else
        printf "  %-8s FAILED\n" "$name"
    fi
done

echo -n "  PGO      "
PGO_BIN=$(build_pgo)
if [ -n "$PGO_BIN" ]; then
    echo "OK"
    BINS["PGO"]="$PGO_BIN"
else
    echo "FAILED"
fi

# ---------------------------------------------------------------
# Benchmark flag sweep (OPENBLAS_NUM_THREADS=1)
# ---------------------------------------------------------------
echo ""
echo "=== Compiler Flag Benchmark (threads=1, seq=$SEQ, iters=$ITERS) ==="
echo ""
printf "%-10s | %10s | %10s | %10s\n" "FLAG_SET" "tok/s" "ms/tok" "speedup"
echo "-----------|------------|------------|------------"

BASE_TOKS=""
ALL_RESULTS=()

measure_flags=(${FLAG_ORDER[@]} PGO)
for name in "${measure_flags[@]}"; do
    bin="${BINS[$name]:-}"
    if [ -z "$bin" ]; then
        ALL_RESULTS+=("$name -1")
        continue
    fi
    toks=$(run_bench "$bin" 1)
    if [ -z "$toks" ] || [ "$toks" = "-1" ]; then
        printf "%-10s | %10s | %10s | %10s\n" "$name" "ERR" "ERR" "ERR"
        continue
    fi

    ms_tok=$(echo "$toks" | awk '{printf "%.2f", 1000.0/$1}')

    if [ -z "$BASE_TOKS" ] && [ "$name" = "BASE" ]; then
        BASE_TOKS="$toks"
    fi

    if [ -n "$BASE_TOKS" ] && [ "$BASE_TOKS" != "0" ]; then
        speedup=$(echo "$toks $BASE_TOKS" | awk '{printf "%.2fx", $1/$2}')
    else
        speedup="—"
    fi

    printf "%-10s | %10.1f | %10s | %10s\n" "$name" "$toks" "${ms_tok}ms" "$speedup"
    ALL_RESULTS+=("$name $toks")
done

# ---------------------------------------------------------------
# Thread sweep (best flag set = FULL)
# ---------------------------------------------------------------
BEST_BIN="${BINS[FULL]:-${BINS[NATIVE]:-}}"
if [ -n "$BEST_BIN" ]; then
    echo ""
    echo "=== OpenBLAS Thread Count Sweep (FULL flags, seq=$SEQ) ==="
    echo ""
    printf "%-12s | %10s | %10s\n" "THREADS" "tok/s" "speedup"
    echo "-------------|------------|------------"

    BASE_THREAD_TOKS=""
    for t in 1 2 4 6 8; do
        toks=$(run_bench "$BEST_BIN" "$t")
        if [ -z "$toks" ] || [ "$toks" = "-1" ]; then continue; fi
        if [ -z "$BASE_THREAD_TOKS" ]; then BASE_THREAD_TOKS="$toks"; fi
        speedup=$(echo "$toks $BASE_THREAD_TOKS" | awk '{printf "%.2fx", $1/$2}')
        printf "%-12s | %10.1f | %10s\n" "$t" "$toks" "$speedup"
    done
fi

echo ""
echo "=== Summary ==="
echo "Weights: $WEIGHTS"
echo "Sequence length: $SEQ"
echo "Iters per run: $ITERS"
echo "System: $(uname -m), $(nproc) logical CPUs"
echo "GCC: $(gcc --version | head -1)"
echo "Done."
