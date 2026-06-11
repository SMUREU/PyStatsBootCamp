# Topic Modeling for Text Data: Python Code

By: Dr. Monnie McGee

Adapted to Python for the SMUREU Data Science Bootcamp.

This notebook contains the Python code adapted from the original R Markdown lesson on Topic Modeling.

The goal is to preserve the same lesson structure while using Python tools.

## R Exercise: Tokenize Text


```python
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
 
```


```python
 
documents = pd.read_csv("AbstractExample.csv")

documents.head()
 
```

The `documents` data frame should contain columns:

- `id`
- `text`


```python
 
documents.columns
 
```

We will tokenize the text and remove English stopwords using `CountVectorizer`.


```python
 
vectorizer = CountVectorizer(
    stop_words="english"
)

dtm = vectorizer.fit_transform(documents["text"])

dtm
 
```

The document-term matrix stores word counts.

Rows represent documents.

Columns represent terms.


```python
 
terms = vectorizer.get_feature_names_out()
terms[:20]
 
```


```python
 
word_counts = pd.DataFrame(
    dtm.toarray(),
    columns=terms,
    index=documents["id"]
)

word_counts.head()
 
```

## Fit LDA in Python


```python
 
# Fit a topic model with 5 topics
lda_fit = LatentDirichletAllocation(
    n_components=5,
    random_state=1234
)

lda_fit.fit(dtm)
 
```

Display the 10 most probable words for each topic.


```python
 
def display_topics(model, feature_names, n_top_words=10):
    topic_words = {}

    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_terms = [feature_names[i] for i in top_indices]
        topic_words[f"Topic {topic_idx + 1}"] = top_terms

    return pd.DataFrame(topic_words)

display_topics(lda_fit, terms, 10)
 
```

## Visualize Top Words by Topic


```python
 
def topic_word_table(model, feature_names, n_top_words=10):
    rows = []

    for topic_idx, topic in enumerate(model.components_):
        topic_probabilities = topic / topic.sum()
        top_indices = topic_probabilities.argsort()[::-1][:n_top_words]

        for index in top_indices:
            rows.append({
                "topic": topic_idx + 1,
                "term": feature_names[index],
                "beta": topic_probabilities[index]
            })

    return pd.DataFrame(rows)

top_terms = topic_word_table(lda_fit, terms, 10)

top_terms
 
```


```python
 
for topic in sorted(top_terms["topic"].unique()):
    subset = top_terms[top_terms["topic"] == topic].sort_values("beta")

    plt.figure(figsize=(8, 4))
    plt.barh(subset["term"], subset["beta"])
    plt.title(f"Top Words for Topic {topic}")
    plt.xlabel("Beta")
    plt.ylabel("Term")
    plt.show()
 
```

What would you name the topics?

## Most Representative Documents


```python
 
# Document-topic probabilities
gamma = lda_fit.transform(dtm)

gamma_df = pd.DataFrame(
    gamma,
    columns=[f"Topic {i + 1}" for i in range(gamma.shape[1])]
)

gamma_df.insert(0, "id", documents["id"])

gamma_df
 
```

Most representative document for each topic:


```python
 
representative_docs = []

for topic_idx in range(gamma.shape[1]):
    doc_position = gamma[:, topic_idx].argmax()

    representative_docs.append({
        "topic": topic_idx + 1,
        "id": documents.iloc[doc_position]["id"],
        "gamma": gamma[doc_position, topic_idx],
        "text": documents.iloc[doc_position]["text"]
    })

representative_docs = pd.DataFrame(representative_docs)

representative_docs
 
```

## Choosing K for LDA

Compare several values of K.


```python
 
lda3 = LatentDirichletAllocation(
    n_components=3,
    random_state=1234
)

lda5 = LatentDirichletAllocation(
    n_components=5,
    random_state=1234
)

lda3.fit(dtm)
lda5.fit(dtm)
 
```


```python
 
display_topics(lda3, terms, 10)
 
```


```python
 
display_topics(lda5, terms, 10)
 
```

Or compare perplexity.

Lower is better.


```python
 
k_values = range(2, 6)

perplexities = []

for k in k_values:
    fit = LatentDirichletAllocation(
        n_components=k,
        random_state=1234
    )

    fit.fit(dtm)
    perplexities.append(fit.perplexity(dtm))

perplexity_df = pd.DataFrame({
    "k": list(k_values),
    "perplexity": perplexities
})

perplexity_df
 
```


```python
 
plt.figure(figsize=(6, 4))
plt.plot(perplexity_df["k"], perplexity_df["perplexity"], marker="o")
plt.xlabel("Number of Topics (K)")
plt.ylabel("Perplexity")
plt.title("LDA Perplexity by Number of Topics")
plt.show()
 
```

Perplexity measures how well the model predicts unseen words; lower values indicate better predictive performance, but not necessarily more interpretable topics. Perplexity plays a role similar to $R^2$ in regression. It measures how well the model explains or predicts the observed data. However, just as $R^2$ tends to increase when we add more predictors, perplexity tends to decrease when we add more topics.

## Fit an STM-Style Model in Python

LDA ignores metadata on the text, also called covariates.

Structural Topic Modeling is an advancement on LDA that is meant to remedy that problem.

Python does not have a direct mainstream equivalent to the R `stm` package.

For this Python version, we will preserve the same idea by:

1. Fitting an LDA topic model
2. Estimating document-topic probabilities
3. Comparing topic prevalence across document metadata groups

This gives us an STM-style workflow for instructional purposes, though it is not a full Structural Topic Model.


```python
 
abstracts = pd.read_csv("AbstractExample.csv")

abstracts.head()
 
```

Add document metadata.


```python
 
abstracts["field"] = [
    "policy", "health",
    "environment", "health",
    "environment", "policy",
    "statistics",
    "environment"
]

abstracts
 
```

Create a document-term matrix.


```python
 
stm_vectorizer = CountVectorizer(
    stop_words="english"
)

stm_dtm = stm_vectorizer.fit_transform(abstracts["text"])
stm_terms = stm_vectorizer.get_feature_names_out()
 
```

Fit a topic model with 5 topics.


```python
 
stm_like_fit = LatentDirichletAllocation(
    n_components=5,
    random_state=1234
)

stm_like_fit.fit(stm_dtm)
 
```

## Choosing K for STM

The R `stm` package includes tools such as `searchK()`.

`searchK()` does more than fit models. It also tries to compute diagnostics such as held-out likelihood. With only a small number of short documents, the held-out step can fail. Here is the analogous idea in Python using perplexity, but we won't rely on it for our toy example.


```python
 
# Compare several values of K using perplexity
k_values = [3, 5]

stm_like_perplexities = []

for k in k_values:
    fit = LatentDirichletAllocation(
        n_components=k,
        random_state=1234
    )

    fit.fit(stm_dtm)

    stm_like_perplexities.append({
        "k": k,
        "perplexity": fit.perplexity(stm_dtm)
    })

pd.DataFrame(stm_like_perplexities)
 
```

## Get Topics


```python
 
stm_like_topics = topic_word_table(
    stm_like_fit,
    stm_terms,
    n_top_words=10
)

stm_like_topics
 
```

Topic proportions:


```python
 
theta = stm_like_fit.transform(stm_dtm)

theta_df = pd.DataFrame(
    theta,
    columns=[f"Topic {i + 1}" for i in range(theta.shape[1])]
)

theta_df = pd.concat(
    [abstracts[["id", "field", "text"]].reset_index(drop=True), theta_df],
    axis=1
)

theta_df
 
```

Average topic proportions:


```python
 
topic_columns = [col for col in theta_df.columns if col.startswith("Topic")]

theta_df[topic_columns].mean()
 
```

Most representative documents for Topic 1:


```python
 
top_doc = theta_df.sort_values(
    by="Topic 1",
    ascending=False
).head(5)

top_doc[["id", "field", "Topic 1", "text"]]
 
```

What would you name the topics?

## Estimate Effects of Metadata

How do the topics change as a function of the model covariates?

STM allows a regression-like model where the response variable is topic prevalence and the predictor is document metadata.

In this Python version, we can approximate this idea by comparing estimated topic proportions across metadata groups.


```python
 
field_topic_means = theta_df.groupby("field")[topic_columns].mean()

field_topic_means
 
```


```python
 
field_topic_means.T.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.ylabel("Average Topic Proportion")
plt.xlabel("Topic")
plt.title("Average Topic Proportions by Field")
plt.legend(title="Field")
plt.show()
 
```

For a regression-style comparison, we can model a topic proportion as a function of document metadata.


```python
 
import statsmodels.formula.api as smf
 
```


```python
 
theta_model_data = theta_df.rename(columns={
    "Topic 1": "topic_1",
    "Topic 2": "topic_2",
    "Topic 3": "topic_3",
    "Topic 4": "topic_4",
    "Topic 5": "topic_5"
})

model = smf.ols(
    "topic_1 ~ C(field)",
    data=theta_model_data
).fit()

model.summary()
 
```

This is not identical to an STM model, but it demonstrates the same research question:

> Do topic proportions vary by document metadata?

## Discussion

- What would you name each topic?
- Which words define each topic?
- Which documents are most representative of each topic?
- Does the metadata appear related to topic prevalence?
- How might the result change if we chose a different value of K?
- What are the limitations of using a small toy dataset?
