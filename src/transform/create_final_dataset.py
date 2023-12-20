import pandas as pd
from src.utils_ import to_parquet

INPUT_PATH = "data/transform/tmp"
OUTPUT_PATH = "data/transform/output"

def _discretize(num):
    side = "Left" if num < 0 else "Right"
    val = abs(num)
    if val <= 5:
        return "Center"
    elif val <= 15:
        return f"Lean {side}"
    else:
        return side


def _balanced_sampling(group):
    min_count = group.groupby(["media_bias"]).size().min()

    def sample_subgroup(subgroup):
        return subgroup.sample(n=min_count)

    return (
        group.groupby(["media_bias"])
        .apply(sample_subgroup)
        .reset_index(drop=True)
    )

@to_parquet(f"{OUTPUT_PATH}/final_sentence_pool.parquet")
def main():
    articles = pd.read_parquet(f"{INPUT_PATH}/articles.parquet")
    outlets = pd.read_parquet("data/extract/output/outlets.parquet")
    sentences = pd.read_parquet(f"{INPUT_PATH}/sentences.parquet")

    df = articles.merge(sentences, on="article_id").merge(
        outlets, on="outlet_id"
    )
    df=df.reset_index(drop=True)

    df["uncertain"] = (
        (df.bias_estimate >= 0.25) & (df.bias_estimate <= 0.75)
    ).astype(int)
    df["media_bias"] = (df.bias_estimate >= 0.5).astype(int)
    df["bias_rating_adfontes"] = df.bias.apply(_discretize)
    df = df[df.bias_rating == df.bias_rating_adfontes]
    df["trustworthy"] = (df["reliability"] >= 0.5).astype(int)

    df = (
        df.groupby(["bias_rating"])
        .apply(_balanced_sampling)
        .reset_index(drop=True)
    )

    def sample_(group):
        size_ = min(len(group), 18000)
        return group.sample(n=size_)

    df = df.groupby("bias_rating").apply(sample_).reset_index(drop=True)
    df = df[
        [
            "sentence",
            "bias_rating",
            "uni_source",
            "media_bias",
            "uncertain",
            "topic",
            "sentence_id",
            "article_id",
        ]
    ]
    final = df.sample(frac=1.0, random_state=42)

    return final


if __name__ == "__main__":
    main()
