PY ?= python3

doctor:
	$(PY) -m chonkforge doctor

train:
	$(PY) -m chonkforge train --steps 30

bench:
	$(PY) -m chonkforge bench

serve:
	$(PY) -m chonkforge serve --port 8000

quantize:
	$(PY) -m chonkforge quantize

test:
	$(PY) -m unittest discover -s tests -t .

compile:
	$(PY) -m compileall -q chonkforge

.PHONY: doctor train bench serve quantize test compile
