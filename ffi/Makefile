RUSTTARGET ?= ./target
PREFIX ?= /usr/local

CARGO ?= cargo
SCDOC ?= scdoc
DOXYGEN ?= doxygen
LIBDIR ?= lib
INCDIR ?= include

all: coupe

coupe:
	$(CARGO) build --locked --release

doc:
	$(DOXYGEN) Doxyfile
	@echo "HTML docs generated at \e]8;;file://$(PWD)/build/html/index.html\e\\\\build/html\e]8;;\e\\"

clean:
	$(CARGO) clean

install:
	mkdir -p $(DESTDIR)$(PREFIX)/$(LIBDIR)
	mkdir -p $(DESTDIR)$(PREFIX)/$(INCDIR)
	cp -f include/coupe.h $(DESTDIR)$(PREFIX)/$(INCDIR)/
	cp -f $(RUSTTARGET)/release/libcoupe.a $(DESTDIR)$(PREFIX)/$(LIBDIR)/
	cp -f $(RUSTTARGET)/release/libcoupe.so $(DESTDIR)$(PREFIX)/$(LIBDIR)/

.PHONY: coupe clean doc install
