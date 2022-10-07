#!/bin/sh
set -e

mkdir -p target/doc/man
for f in tools/doc/*.scd
do
    html=$(basename "$f" .scd).html
    scdoc <"$f" |
        mandoc -Thtml -O style=man.css |
        sed 's|<b>\([a-z-]\+\)</b>(1)|<a href="\1.1.html"><b>\1</b>(1)</a>|g' >target/doc/man/"$html"
done

cp docs/index.html target/doc/
cp docs/man.css target/doc/man/
