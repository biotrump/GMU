# Accelerated k-means and mean-shift algorithms via OpenCL
# Authors: Martin Simon & Pavel Sirucek

SRC=src/
DOC=doc/
TARFILE=gmu-xsimon14-xsiruc01.tar.gz
DOCFILE=$(DOC)gmu.pdf

.PHONY: all src doc

all: src

src:
	make -C $(SRC)

doc:
	make -C $(DOC)


clean: clean-src# clean-doc

clean-src:
	make -C $(SRC) clean

clean-doc:
	make -C $(DOC) clean


pack: doc src clean
	rm -f $(TARFILE)
	tar -czf $(TARFILE) Makefile LICENSE README.md $(SRC)* $(DOCFILE)