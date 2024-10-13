***Context***: hired by Swiss government to advise on a large-scale "AI integration" initiative code names **Neutrality**

***Goal***: investigate which LMs might be best suited for education. 

We are given 3 LM performances on MMLU. Dataset is corrupted.

# Task 1: Inspecting the results and getting a first model ranking

# Task 2: Inspecting the underlying data used to generate the results for possible biases

```python 
# Distribution of correct answers (A, B, C, D) grouped by LM and dataset  
categories_mmlu = df_mmlu['result'].unique()  
categories_other = df_other['result'].unique()  
categories = sorted(set(categories_mmlu) | set(categories_other))  
  
fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)  
sns.barplot(data=df_mmlu, x="result", y="correct", ax=ax[0], estimator="mean", errorbar=('ci', 95), capsize=0.04, err_kws={'color': 'black'}, alpha=0.7, order=categories)  
sns.barplot(data=df_other, x="result", y="correct", ax=ax[1], estimator="mean", errorbar=('ci', 95), capsize=0.04, err_kws={'color': 'black'}, alpha=0.7, color="orange", order=categories)  
ax[0].set_title("MMLU Dataset")  
ax[1].set_title("Other Dataset")  
ax[0].set_xlabel("Answer")  
ax[1].set_xlabel("Answer")  
ax[0].set_ylabel("Mean Accuracy")  
ax[0].grid(True, alpha=0.5)  
ax[1].grid(True, alpha=0.5)  
plt.ylim(0.4, 0.9)  
plt.tight_layout()  
plt.show()
```

```python
from matplotlib.lines import Line2D  
  
categories_mmlu = df_mmlu['result'].unique()  
categories_other = df_other['result'].unique()  
categories = sorted(set(categories_mmlu) | set(categories_other))  
  
models_mmlu = df_mmlu['model_name'].unique()  
models_other = df_other['model_name'].unique()  # same as models_mmlu but for clarity  
models = sorted(set(models_mmlu) | set(models_other))  
  
fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)  
sns.barplot(  
    data=df_mmlu,  
    x='result',  
    y='correct',  
    hue='model_name',    
    ax=ax[0],  
    estimator='mean',  
    errorbar=('ci', 95),  
    capsize=0.04,  
    err_kws={'color': 'black'},  
    alpha=0.7,  
    order=categories,  
    palette='tab10'  )  
ax[0].set_title("MMLU Dataset")  
ax[0].set_xlabel("Answer")  
ax[0].set_ylabel("Mean Accuracy")  
ax[0].get_legend().remove()  # Strip legend from the left plot  
ax[0].grid(True, alpha=0.5)  
  
# Other Dataset  
sns.barplot(  
    data=df_other,  
    x='result',  
    y='correct',  
    hue='model_name',  
    ax=ax[1],  
    estimator='mean',  
    errorbar=('ci', 95),  
    capsize=0.04,  
    err_kws={'color': 'black'},  
    alpha=0.7,  
    order=categories,  
    palette='tab10'  
)  
  
# Compute the mean accuracy for each category  
mean_mmlu = df_mmlu.groupby('result')['correct'].mean().reindex(categories)  
mean_other = df_other.groupby('result')['correct'].mean().reindex(categories)  
positions = range(len(categories))  # Positions for the bars  
  
# Add horizontal lines for the mean accuracy over all models  
for i, ax_i in enumerate(ax):  
    if i == 0:  
        mean_values = mean_mmlu  
    else:  
        mean_values = mean_other  
          
    for pos, category in zip(positions, categories):  
        mean_value = mean_values[category]  
        if pd.notna(mean_value):  
            num_models = len(models)  
            bar_width = 0.8 / num_models  
            total_bar_width = bar_width * num_models  
            left = pos - total_bar_width / 2  
            right = pos + total_bar_width / 2  
            ax_i.hlines(  
                y=mean_value,  
                xmin=left,  
                xmax=right,  
                colors='red',  
                linestyles='--',  
                linewidth=2,   
            )  
  
# Add the mean line to the legend  
mean_line = Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Mean over Models')  
handles, labels = ax[1].get_legend_handles_labels()  
handles.append(mean_line)  
labels.append('Mean')  
ax[1].legend(  
    handles=handles,  
    labels=labels,  
    loc='upper left',  
    bbox_to_anchor=(1, 1),  
    title='Model',  
    title_fontsize='25'  
)  
  
ax[1].set_title("Other Dataset")  
ax[1].set_xlabel("Answer")  
ax[1].grid(True, alpha=0.5)  
plt.ylim(0,1.05)  
plt.tight_layout()  
plt.show()
```

# Task 3: Learning about tokens, final recommendation