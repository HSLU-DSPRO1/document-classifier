.PHONY: install
install:
	python -m pip install -r requirements.txt

.PHONY: notebook
notebook:
	jupyter lab

.PHONY: clean
clean:
	rm -rf data/processed/*.csv outputs/figures/*.png
