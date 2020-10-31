
all: build

encode: build
	./bms -e

decode: build
	./bms -d

build:
	echo -e "#!/usr/bin/env python3.8\n" > bms
	cat main.py >> bms
	chmod +x bms

clean:
	rm -f bms xdolej08.tar.gz

pack:
	zip xdolej08.zip *.py Makefile
