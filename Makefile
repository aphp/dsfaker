install:
	pip install -r requirements.txt
	pip install .

uninstall:
	pip uninstall -y dsfaker

reinstall: uninstall install

.ONESHELL:
doc:
	pip install -r requirements.doc.txt
	cd docs
	rm -r _build
	make html

open-doc-chrome: doc
	google-chrome-stable docs/_build/html/index.html

test:
	pip install -r requirements.test.txt


all: install
