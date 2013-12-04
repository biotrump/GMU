NAME=gmu
RM=rm -rf
VLNA=vlna

all: $(NAME).pdf

pdf: $(NAME).pdf

$(NAME).ps: $(NAME).dvi
	dvips $(NAME)

$(NAME).pdf: clean
	pdflatex $(NAME)
	pdflatex $(NAME)

$(NAME).dvi: $(NAME).tex titlepage.tex
	$(VLNA) $(NAME)
	latex $(NAME)
	latex $(NAME)

clean:
	$(RM) $(NAME).dvi $(NAME).aux $(NAME).toc $(NAME).log $(NAME).out
