## Makefile.am — top-level automake for cngpt
##
## MIT License — see COPYING

SUBDIRS = src tests man

EXTRA_DIST = \
    scripts/export_weights.py \
    README.md \
    AUTHORS \
    NEWS \
    ChangeLog \
    INSTALL

## Convenience aliases at top level
.PHONY: bench
bench:
	$(MAKE) -C tests bench
