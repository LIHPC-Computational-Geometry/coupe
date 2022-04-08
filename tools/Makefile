RUSTTARGET ?= ./target
PREFIX ?= /usr/local

CARGO ?= cargo
RM ?= rm
SCDOC ?= scdoc
RUSTFLAGS ?=
BINDIR ?= bin
LIBDIR ?= lib
MANDIR ?= share/man

tools = apply-part apply-weight mesh-refine mesh-part part-bench part-info weight-gen
toolsdoc = $(foreach tool,$(tools),doc/$(tool).1)
toolsbin = $(foreach tool,$(tools),$(RUSTTARGET)/release/$(tool))

all: tools mesh-io $(toolsdoc)

tools:
	$(CARGO) build --locked --release --bins

mesh-io:
	$(CARGO) build --locked --release -p mesh-io-ffi

doc/%.1: doc/%.1.scd
	$(SCDOC) <$^ >$@

clean:
	$(CARGO) clean

install:
	mkdir -p $(DESTDIR)$(PREFIX)/$(BINDIR)
	mkdir -p $(DESTDIR)$(PREFIX)/$(LIBDIR)
	mkdir -p $(DESTDIR)$(PREFIX)/$(MANDIR)/man1
	cp -f $(toolsbin) $(DESTDIR)$(PREFIX)/$(BINDIR)/
	cp -f $(toolsdoc) $(DESTDIR)$(PREFIX)/$(MANDIR)/man1/
	cp -f $(RUSTTARGET)/release/libmeshio.a $(DESTDIR)$(PREFIX)/$(LIBDIR)/
	cp -f $(RUSTTARGET)/release/libmeshio.so $(DESTDIR)$(PREFIX)/$(LIBDIR)/

.PHONY: tools mesh-io clean install
