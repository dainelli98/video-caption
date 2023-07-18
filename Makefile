.PHONY: venv lint docs format

PYTHON=python3
PIP=pip3
POETRY=poetry

venv:
	$(PIP) install --no-cache-dir --upgrade pip wheel poetry==1.5.0
	$(POETRY) config virtualenvs.create false
	$(POETRY) lock --no-update
	$(POETRY) install --all-extras
	pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type pre-merge-commit

lint:
	ruff --fix vid_cap

docs:
	rm -f docs/source/vid_cap*.rst
	rm -f docs/source/modules.rst
	rm -rf docs/build
	rm -f docs/source/release_notes.md
	sphinx-apidoc -o docs/source vid_cap
	cp RELEASE_NOTES.md docs/source/release_notes.md
	sphinx-build -b html -c docs/source -W docs/source docs/build/html -D autodoc_member_order="bysource"

format:
	black -l 100 .
	ruff -s --fix --exit-zero .
	docformatter -r -i --wrap-summaries 100 --wrap-descriptions 90 .

inference:
	poetry run python -m vid_cap.__main__ inference --n-heads 4 --n-layers 2 --video-path $(VIDEO) --inference-model models/inference/model --inference-model-vocab models/inference/vocab.pkl 