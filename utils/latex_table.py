import pandas as pd

df = pd.read_csv('results.csv')
grouped = df.groupby('name').agg(['mean', 'std'])
model_names = [
    # "Contrastive-MM",
    'Contras-MM-80',
    # "MISR",
    "FISR",
    "MTFM",
    "LightGCL",
    "T2L2",
    "T2L2-W/O-propagation",
    "GSAT",
    "Random-MM-80"
]

multirow_metrics = ['Precision', 'Recall', 'NDCG', 'F1', 'PSP', 'PSDCG', 'MAP']
top_ks = [5, 10, 15, 20, 25]

latex_lines = []
latex_lines.append(r"\begin{tabular}{ll" + "l" * len(model_names) + r"}")
latex_lines.append(r"\hline")
latex_lines.append(r"Metrics & topk & " + " & ".join(model_names) + r" \\")
latex_lines.append(r"\hline")

for metric in multirow_metrics:
    for i, top_k in enumerate(top_ks):
        row = []
        if i == 0:
            row.append(r"\multirow{5}{*}{\textbf{" + metric + r"}}")
        else:
            row.append("")
        
        row.append(str(top_k))
        
        for model in model_names:
            key = f"{metric}@{top_k}"
            mean = grouped.loc[model, (key, 'mean')]
            std = grouped.loc[model, (key, 'std')]
            row.append(f"{mean:.3f}(±{std:.3f})")
        
        latex_lines.append(" & ".join(row) + r" \\")
        if i != len(top_ks)-1:
            latex_lines.append(r"\cmidrule{2-" + str(2 + len(model_names)) + r"}")
        else:
            latex_lines.append(r"\hline")

mrr_row = [r"\textbf{MRR}", ""]
for model in model_names:
    mean = grouped.loc[model, ('MRR', 'mean')]
    std = grouped.loc[model, ('MRR', 'std')]
    mrr_row.append(f"{mean:.3f}(±{std:.3f})")
latex_lines.append(" & ".join(mrr_row) + r" \\")
latex_lines.append(r"\hline")
latex_lines.append(r"\end{tabular}")

print("\n".join(latex_lines))

