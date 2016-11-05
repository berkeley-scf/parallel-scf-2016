all: session1.html session1_slides.html session2.html session2_slides.html

session%.html: session%.md
	pandoc -s -o $@ $<

session%_slides.html: session%.md
	pandoc -s --webtex -t slidy -o $@ $<

clean:
	rm -rf *html
