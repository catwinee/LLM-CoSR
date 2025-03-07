import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

model_base_name = 'LLM-CoSR'
model_names = ["Random-MM", "Contras-MM"]
display_names = [f'{model_base_name}-R', model_base_name]

df = pd.read_csv("results.csv")

df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.replace('Random-MM', 'LLM-CoSR-R')
df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.replace('Contras-MM', 'LLM-CoSR')

dfs = []
for model in display_names:
    mask = df['name'].str.match(fr'^{model}-\d+$', flags=re.IGNORECASE)
    model_df = df[mask].copy()
    
    model_df['model_type'] = model
    model_df['model_num'] = model_df['name'].str.extract(r'(\d+)$').astype(int)
    dfs.append(model_df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df = combined_df.sort_values(['model_type', 'model_num'])

plt.figure(figsize=(12, 7))
sns.set_theme(
    style="whitegrid",
    palette=sns.color_palette("husl", len(model_names)),
    font_scale=1.1
)

first_metric = 'F1@5'
metrics = [first_metric]
for metric in metrics:
    sns.lineplot(
        data=combined_df,
        x='model_num',
        y=metric,
        hue='model_type',
        estimator=np.mean,
        errorbar=('sd', 1),
        err_style='band',
        linewidth=2.5,
        marker='o',
        markersize=8,
        alpha=0.9
    )

all_model_nums = combined_df['model_num']
min_num, max_num = all_model_nums.min(), all_model_nums.max()
plt.xlim(min_num - 2, max_num + 2)
plt.xticks(np.arange(min_num, max_num + 10, 10))

plt.title(f"{first_metric} Trend Across Edge Removal Ratio (avg±std)", fontsize=16, pad=20)
plt.xlabel("Remove Ratio", fontsize=14, labelpad=12)
plt.ylabel(first_metric, fontsize=14, labelpad=12)

plt.legend(
    title='Models',
    title_fontsize='13',
    fontsize='12',
    loc='best'
)
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'model_comparison{metrics[0]}.png')
plt.show()