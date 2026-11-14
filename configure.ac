dnl configure.ac — autoconf configuration for cngpt
dnl
dnl MIT License — see COPYING

AC_PREREQ([2.69])
AC_INIT([cngpt], [0.1.0], [https://github.com/davmul/cngpt/issues])
AM_INIT_AUTOMAKE([foreign -Wall -Werror subdir-objects])
AC_CONFIG_SRCDIR([src/cngpt.c])
AC_CONFIG_HEADERS([config.h])

dnl ---------------------------------------------------------------
dnl Compiler
dnl ---------------------------------------------------------------
AC_PROG_CC
AC_PROG_CC_C99

dnl OpenMP: threads the optimized forward pass (src/fast.c). Optional;
dnl without it everything still builds and runs single-threaded.
AC_OPENMP

dnl Enable common warnings. CNGPT_CHECK_CFLAG is defined here so the
dnl build does not depend on autoconf-archive being installed.
AC_DEFUN([CNGPT_CHECK_CFLAG], [
    AC_MSG_CHECKING([whether $CC accepts $1])
    cngpt_save_CFLAGS="$CFLAGS"
    CFLAGS="$CFLAGS $1 -Werror"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [])],
        [AC_MSG_RESULT([yes]); $2],
        [AC_MSG_RESULT([no]); $3])
    CFLAGS="$cngpt_save_CFLAGS"
])

CNGPT_CHECK_CFLAG([-Wall],     [WARN_FLAGS="$WARN_FLAGS -Wall"])
CNGPT_CHECK_CFLAG([-Wextra],   [WARN_FLAGS="$WARN_FLAGS -Wextra"])
CNGPT_CHECK_CFLAG([-Wpedantic],[WARN_FLAGS="$WARN_FLAGS -Wpedantic"])
CNGPT_CHECK_CFLAG([-Wno-unused-parameter],
    [WARN_FLAGS="$WARN_FLAGS -Wno-unused-parameter"])
AC_SUBST([WARN_FLAGS])

dnl ---------------------------------------------------------------
dnl Required headers
dnl ---------------------------------------------------------------
AC_CHECK_HEADERS([fcntl.h sys/mman.h sys/stat.h unistd.h],, [
    AC_MSG_ERROR([POSIX headers not found])
])

dnl ---------------------------------------------------------------
dnl Math library
dnl ---------------------------------------------------------------
AC_CHECK_LIB([m], [sqrtf],, [
    AC_MSG_ERROR([libm not found])
])

dnl ---------------------------------------------------------------
dnl pthreads (required by OpenBLAS)
dnl ---------------------------------------------------------------
AC_CHECK_LIB([pthread], [pthread_create],, [
    AC_MSG_ERROR([libpthread not found])
])
AC_CHECK_HEADERS([pthread.h],, [
    AC_MSG_ERROR([pthread.h not found])
])

dnl ---------------------------------------------------------------
dnl OpenBLAS / CBLAS
dnl ---------------------------------------------------------------
dnl First look for cblas.h in common distribution paths
BLAS_CFLAGS=""
BLAS_LIBS=""

dnl Debian/Ubuntu path
AC_CHECK_FILE([/usr/include/x86_64-linux-gnu/cblas.h], [
    BLAS_CFLAGS="-I/usr/include/x86_64-linux-gnu"
    BLAS_LIBS="-L/usr/lib/x86_64-linux-gnu -lopenblas"
], [
    dnl Generic path
    AC_CHECK_HEADERS([cblas.h],, [
        AC_MSG_ERROR([cblas.h not found. Install libopenblas-dev (Debian/Ubuntu).])
    ])
    BLAS_LIBS="-lopenblas"
])

dnl Verify the library links: compile and link a real cblas_sgemm call
dnl against the real prototype from cblas.h.
saved_CFLAGS="$CFLAGS"
saved_LIBS="$LIBS"
CFLAGS="$CFLAGS $BLAS_CFLAGS"
LIBS="$LIBS $BLAS_LIBS -lm -lpthread"
AC_MSG_CHECKING([for cblas_sgemm in OpenBLAS])
AC_LINK_IFELSE([AC_LANG_PROGRAM([[
    #include <cblas.h>
]], [[
    float a = 1.0f, b = 1.0f, c = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, 1, 1, 1.0f, &a, 1, &b, 1, 0.0f, &c, 1);
    return c > 0.5f ? 0 : 1;
]])], [
    AC_MSG_RESULT([yes])
], [
    AC_MSG_RESULT([no])
    AC_MSG_ERROR([Could not link against OpenBLAS. Check BLAS_LIBS or install libopenblas-dev.])
])
CFLAGS="$saved_CFLAGS"
LIBS="$saved_LIBS"

AC_SUBST([BLAS_CFLAGS])
AC_SUBST([BLAS_LIBS])

dnl ---------------------------------------------------------------
dnl Optional: check for Python 3 (for export_weights.py)
dnl ---------------------------------------------------------------
AC_PATH_PROG([PYTHON3], [python3], [not_found])
AS_IF([test "$PYTHON3" = "not_found"],
    [AC_MSG_WARN([python3 not found — export_weights.py will not be usable])])

dnl ---------------------------------------------------------------
dnl Output
dnl ---------------------------------------------------------------
AC_CONFIG_FILES([
    Makefile
    src/Makefile
    tests/Makefile
    man/Makefile
])
AC_OUTPUT

echo ""
echo "cngpt $PACKAGE_VERSION configuration:"
echo "  CC         : $CC"
echo "  CFLAGS     : $CFLAGS $WARN_FLAGS"
echo "  OpenMP     : ${OPENMP_CFLAGS:-none}"
echo "  BLAS_CFLAGS: $BLAS_CFLAGS"
echo "  BLAS_LIBS  : $BLAS_LIBS"
echo "  Python3    : $PYTHON3"
echo ""
echo "Now run: make && make check"
