import logging
import os
import re
import subprocess

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="\x1b[32;1m" + "%(message)s (%(filename)s:%(lineno)d)" + "\x1b[0m",
)


def unify_site_name(site_name):
    """
    Unifies the site name by removing specifications in brackets, converting to lowercase,
    removing '.com', removing special characters, and replacing spaces with hyphens.

    Args:
        site_name (str): The site name to be cleaned.

    Returns:
        str: The cleaned site name.
    """
    # Remove the specification in brackets
    base_list = [
        "News",
        "Online News",
        "Online",
    ]  # List of specifications that refer to general news content
    specification = re.search(r"\((.*?)\)", site_name)

    # Remove the specification if it is in the base list
    if specification is not None:
        if specification.group(1) in base_list:
            site_name = re.sub(r"\((.*?)\)", "", site_name).rstrip(" ")

    site_name = site_name.lower()
    site_name = re.sub(".com", "", site_name)
    site_name = re.sub(r"[^\w\s]", "", site_name)
    site_name = site_name.replace(" ", "-")
    return site_name


def community_trust(fdbck):
    agrees, disagrees = fdbck.split("/")
    return int(agrees) / (int(agrees) + int(disagrees))


def community_volume(fdbck):
    agrees, disagrees = fdbck.split("/")
    return int(agrees) + int(disagrees)


def main():
    if not os.path.exists("data/tmp"):
        subprocess.run(["mkdir", "data/tmp"])

    d1 = pd.read_parquet("data/allsides_snapshot.parquet")
    d2 = pd.read_parquet("data/adfontes_snapshot.parquet")

    d1["uni_source"] = d1["news_source"].apply(unify_site_name)
    d2["uni_source"] = d2["source"].apply(unify_site_name)

    merged = d1.merge(d2, on="uni_source")
    merged = merged[~merged["news_link"].isna()]
    merged["community_trust"] = merged["community_feedback"].apply(
        community_trust,
    )
    merged["community_feedback"] = merged["community_feedback"].apply(
        community_volume,
    )
    scaler = MinMaxScaler()
    merged["reliability"] = scaler.fit_transform(merged[["reliability"]])

    merged = merged[merged.bias_rating != "Mixed"]

    neutral = merged[merged.bias_rating == "Center"].sort_values(
        "community_feedback",
        ascending=False,
    )[:30]
    biased = merged[merged.bias_rating != "Center"]

    merged = pd.concat([biased, neutral])

    logging.info("Saving...")
    merged.to_parquet("data/tmp/outlets_merged.parquet")


if __name__ == "__main__":
    main()
