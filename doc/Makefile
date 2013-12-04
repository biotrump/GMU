NAME=gmu
SLIDES=slides
RM=rm -rf
VLNA=vlna

all: pdf slides

pdf: $(NAME).pdf

slides: $(SLIDES).pdf

$(SLIDES).ps: $(SLIDES).dvi
	dvips $(SLIDES)

$(SLIDES).pdf: clean
	pdflatex $(SLIDES)

$(SLIDES).dvi: $(SLIDES).tex
	latex $(SLIDES)


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
	$(RM) $(NAME).dvi $(SLIDES).dvi $(NAME).aux $(SLIDES).aux $(NAME).toc $(SLIDES).toc $(NAME).log $(SLIDES).log $(NAME).out $(SLIDES).out
