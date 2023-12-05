extract:
	python extract/scrape_allsides/scrape.py
	python extract/preprocessing/merge.py
	python extract/scrape_articles/scrape.py
