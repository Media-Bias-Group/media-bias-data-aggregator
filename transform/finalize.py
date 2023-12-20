import pandas as pd

def discretize(num):
    side = 'Left' if num < 0 else 'Right'
    val = abs(num)
    if val <=5:
        return 'Center'
    elif val <= 15:
        return f"Lean {side}"
    else: 
        return side
    
def balanced_sampling(group):
    min_count = group.groupby(['media_bias']).size().min()

    def sample_subgroup(subgroup):
        return subgroup.sample(n=min_count)

    return group.groupby(['media_bias']).apply(sample_subgroup).reset_index(drop=True)


def main():
    articles = pd.read_parquet('data/tmp/articles.parquet')
    outlets = pd.read_parquet('data/output/outlets.parquet')
    sentences = pd.read_parquet('data/tmp/sentences.parquet')

    df = articles.merge(sentences,on='article_id').merge(outlets,on='outlet_id')

    df['uncertain'] = ((df.bias_estimate >= 0.25) & (df.bias_estimate <= 0.75)).astype(int)
    df['media_bias'] = (df.bias_estimate >= 0.5).astype(int)
    df['bias_rating_adfontes'] = df.bias.apply(discretize)
    df = df[df.bias_rating == df.bias_rating_adfontes]
    df['trustworthy'] = (df['reliability'] >= 0.5).astype(int)


    df=df.groupby(['bias_rating']).apply(balanced_sampling).reset_index(drop=True)

    def sample_(group):
        size_ = min(len(group),18000)
        return group.sample(n=size_)

    df = df.groupby('bias_rating').apply(sample_).reset_index(drop=True)
    df=df[['sentence','bias_rating','uni_source','media_bias','uncertain','topic','sentence_id','article_id']]
    final = df.sample(frac=1.,random_state=42)

    final.to_csv('data/output/final_sentences.csv',index=False)


if __name__ == "__main__":
    main()