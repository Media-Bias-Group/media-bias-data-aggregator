import logging
import os
import random
import subprocess
import time
import uuid

import newspaper
import pandas as pd
from newspaper import ArticleException
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

TMP_PATH = "data/tmp"
OUTPUT_PATH = "data/output"


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.CRITICAL,
    format="\x1b[32;1m" + "%(message)s (%(filename)s:%(lineno)d)" + "\x1b[0m",
)
logging.getLogger("newspaper").setLevel(logging.CRITICAL)
MAX_RETRIES = 5
RETRY_DELAY = lambda: random.randint(1, 10)


def scrape_source_articles(outlet_link):
    try:
        source = newspaper.build(outlet_link, memoize_articles=False)
    except Exception as e:
        logging.ERROR(f"Can't build {outlet_link}. Not an URL.")
        return None

    rowlist = []
    for article in tqdm(source.articles):
        try:
            article.download()
            article.parse()
            rowlist.append(
                {
                    "text": article.text,
                    "title": article.title,
                },
            )
        except ArticleException as e:
            continue

    return pd.DataFrame(rowlist)


def split_into_sentences(article_text, title):
    """
    Splits an article text into sentences.

    Args:
        article_text (str): The article text to be split.

    Returns:
        list: The list of sentences.
    """
    sentences = [title] + sent_tokenize(article_text)
    sentences = [sentence for sentence in sentences if len(sentence) > 10]
    return sentences


def main():
    outlets_df = pd.read_parquet(f"{TMP_PATH}/outlets_merged.parquet")

    outlets_df["outlet_id"] = [
        str(uuid.uuid4()) for _ in range(len(outlets_df))
    ]

    # scrape articles
    for _, outlet_row in tqdm(outlets_df.iterrows(), total=len(outlets_df)):
        articles_df = scrape_source_articles(
            outlet_link=outlet_row["news_link"],
        )
        if articles_df is None:
            continue
        articles_df["outlet_id"] = [outlet_row["outlet_id"]] * len(articles_df)
        articles_df["article_id"] = [
            str(uuid.uuid4()) for _ in range(len(articles_df))
        ]
        articles_df.to_parquet(
            f"{TMP_PATH}/articles_{outlet_row['uni_source']}.parquet",
        )

    # merge articles
    article_dataframes = []
    for data in os.listdir(TMP_PATH):
        if data.split("_")[0] == "articles":
            article_dataframes.append(pd.read_parquet(f"{TMP_PATH}/{data}"))

    articles_df = pd.concat(article_dataframes)

    # split into sentences
    sentence_dataframes = []
    for _, article_row in tqdm(articles_df.iterrows()):
        sentences = split_into_sentences(
            article_row["text"],
            article_row["title"],
        )
        sentence_ids = [str(uuid.uuid4()) for _ in range(len(sentences))]
        sentence_dataframes.append(
            pd.DataFrame(
                {
                    "sentence": sentences,
                    "article_id": [article_row["article_id"]] * len(sentences),
                    "sentence_id": sentence_ids,
                },
            ),
        )

    sentences_df = pd.concat(sentence_dataframes)

    if not os.path.exists(OUTPUT_PATH):
        subprocess.run(["mkdir", OUTPUT_PATH])

    outlets_df.to_parquet(f"{OUTPUT_PATH}/outlets.parquet")
    articles_df.to_parquet(f"{OUTPUT_PATH}/articles.parquet")
    sentences_df.to_parquet(f"{OUTPUT_PATH}/sentences.parquet")

    subprocess.run(["rm", "-r", TMP_PATH])


if __name__ == "__main__":
    main()
