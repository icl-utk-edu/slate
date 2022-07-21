docs = flops.pdf
base = ${basename ${docs}}
deps = ${addsuffix .d, ${base}}

FLAGS = -pdf -bibtex

all: ${docs}

%.tex: %.py
	./$< > $@

%.pdf: %.tex
	latexmk ${FLAGS} -M -MF $*.d $*

# -pdf avoids apparently spurious warning
clean:
	latexmk -C -pdf
	-rm -f ${deps} *.bbl *.loa

-include ${deps}
