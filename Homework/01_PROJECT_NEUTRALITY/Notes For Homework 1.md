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

### 3.2.b
**Note**: Here, we are conditioning on the correct answer to the questions rather than the one provided by the language model.

If we were to condition on the model's provided answer, the results would likely be more pronounced than what we observe here. For example, consider a case where we analyze the average frequencies of tokens "A," "B," "C," and "D" and the model's answer is "B." If we then observe a very high frequency of token "B," this could suggest a bias in the model's response.

In our case, by conditioning on the correct answer, we are primarily interested in the "natural distribution" of tokens within the data. However, this does not show a clear correlation with the responses generated by different language models.

**What we observe**: The frequencies of tokens "A," "B," "C," and "D" remain relatively constant regardless of the correct answer. In every instance, token "A" appears most frequently. This higher frequency of token "A" might be due to the fact that the questions are written in English, where "A" is commonly used as an indefinite article, whereas the other letters rarely appear on their own. From this observation alone, it is difficult to conclude whether the frequencies of tokens "A," "B," "C," and "D" influence the model's answers.

**Note on tokenization**: The tokenization process is actually more complex and may sometimes split words into multiple tokens or combine word fragments. Therefore, our analysis based solely on tokens "A," "B," "C," and "D" might not capture all the nuances present in the question-answer pairs.

### 3.4
What a long journey it has been! With all this experience, here are the valuable recommendations we can offer the Swiss government regarding the NEUTRALITY project.

1. When training an AI model, data quality is crucial. The data needs to be cleaned and preprocessed to prevent the model from learning from noise, which is especially important with text data since it tends to be noisy and unstructured. Thoroughly cleaning the data ensures the model learns from relevant, unbiased, and ethical information. However, we are unaware of the exact data used to train models X, Y, and Z, but based on the results, it appears that the data may not have been cleaned properly to answer MCQ with only one letter, as some outputs are incorrectly formatted.

2. After having rebalanced the data so each model answers the same number of questions per subject, it appeared that the accuracy remains almost unchanged. Consequently, we argue that the amount of data per subject might not be a critical factor for the models' performance in this case. So, don't put too much effort into balancing the data. Instead, focus on the quality of the data and the model's architecture.

3. We need to carefully evaluate the trade-off between context length and model performance. While a larger context can provide the model with more information for better predictions, it also increases computational costs and slows down both training and inference. As we observed, the turbo model with a 300-token context length was sufficient for all but 4 out of the 57 subjects. Therefore, we recommend sticking with the 300-token context length for most cases to optimize efficiency, and only consider increasing it for the few subjects if needed where additional context is necessary for better performance.

4. To summarize our analysis, we cannot recommend any of the models for the NEUTRALITY project over the others. Specifically, LM X shows a strong bias towards answer 'A', while LM Y is biased towards answer 'D'. Meanwhile, LM Z doesn't exhibit any noticeable bias but has overall disappointing accuracy. We recommend leveraging the average of models X and Y, as their strengths and weaknesses complement each other. Model Z is not advised due to its consistently poor performance across all subjects. 


Phrase finale drôle demandée à GPT: And as a final note: If neutrality is the goal, maybe our models are already living up to it—by not making any strong decisions either way!
