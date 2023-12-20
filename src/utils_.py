import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load your DataFrame
# If it's a CSV file, use: df = pd.read_csv('your_file.csv')
# Assuming df is your DataFrame

# Example DataFrame loading (remove this line in your actual script)
df = pd.read_csv('/home/tomas/Documents/MBG/projects/media-bias-data-aggregator/data/output/final_sentences.csv') # Replace with your DataFrame

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# Plot 1: Distribution of media_bias in bias_rating
sns.countplot(x='bias_rating', hue='media_bias', data=df, ax=axes[0])
axes[0].set_title('Distribution of Media Bias in Partisanship of the article')
axes[0].set_ylabel('Count')


top_topics = df['topic'].value_counts().head(5).index

# Filter the DataFrame to only include these topics
df_filtered = df[df['topic'].isin(top_topics)]

# Plot 3: Distribution of media_bias in topic
sns.countplot(x='topic', hue='media_bias', data=df_filtered, ax=axes[1])
axes[1].set_title('Distribution of Media Bias in Topic')
axes[1].set_ylabel('Count')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('./media_bias_distribution.png')

# Show the plot
plt.show()