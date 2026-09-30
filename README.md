# Multi-Class News Headline Classification
### A Comparative Analysis of ML and Deep Learning Models on AG News

A systematic benchmark of preprocessing strategies, word representations, and model
architectures for 4-class news headline classification (Business, Sci/Tech, Sports, World).
Built for CSE440 at BRAC University.

## Overview
Rather than chasing a single best score, this project studies **how design choices affect
performance**: 3 preprocessing pipelines x 2 representations x 8 models = **24 experimental
configurations**.

## Dataset
- **AG News** headlines with HTML markup (`<b>`, `<br>`, etc.)
- 88,106 training / 36,000 test samples
- Class imbalance (~3:1): Sci/Tech 33.2%, World 31.9%, Sports 23.0%, Business 11.8%
- 80/20 stratified train/validation split

## Approach

| Component | Options |
|---|---|
| Preprocessing | None (HTML strip only) / Extreme (lowercase, stopwords, Porter stemming) / Optimum (EDA-informed, keeps numbers, WordNet lemmatization) |
| Representation | TF-IDF (50k features, 1-2 grams) / Skip-gram Word2Vec (Gensim, 100-dim) |
| ML model | Logistic Regression |
| Deep models | Feedforward DNN, SimpleRNN, GRU, LSTM + Bidirectional variants |

## Results (Test Set)

| Model | Best Preprocessing | Accuracy | Macro F1 |
|---|---|---|---|
| Logistic Regression (TF-IDF) | Extreme | 0.914 | 0.913 |
| Deep NN (TF-IDF) | Optimum | 0.912 | 0.912 |
| GRU | None | 0.919 | 0.918 |
| LSTM | None | 0.910 | 0.910 |
| Bi-SimpleRNN | Optimum | 0.904 | 0.903 |
| **Bi-GRU** | **Optimum** | **0.919** | **0.919** |
| Bi-LSTM | Extreme | 0.919 | 0.919 |

- Logistic Regression + TF-IDF reached the highest **validation accuracy (94.99%)**.
- Bi-GRU with Optimum preprocessing achieved the best deep learning test result (~91.9%).

## Key Findings
1. TF-IDF models are robust to preprocessing choice (0.2 pt spread for Logistic Regression).
2. RNNs are more sensitive; bidirectional variants are the most stable.
3. A well-tuned linear model is competitive with deep models on short text.
4. RNN hidden size matters: 32 units caused convergence failures (stuck at 25% accuracy);
   64 units with 0.5 dropout fixed it.
5. Sports is easiest to classify; Business is hardest; Sci/Tech and World overlap most.

## Engineering Notes
- Implemented a sparse batch generator for the TF-IDF DNN, cutting peak memory from ~26 GB to <500 MB.
- Trained on Google Colab (GPU). Total compute for all experiments: ~3-4 hours.

## Tech Stack
Python, scikit-learn, TensorFlow/Keras, Gensim, NLTK, Matplotlib, Seaborn

## Repository Structure
```text
├── notebooks/        # experiments per preprocessing / model
├── report/           # CSE440_project_report.pdf
├── figures/          # confusion matrices, training curves
└── README.md
```

## Future Work
Pre-trained transformers (BERT, RoBERTa), attention-augmented RNNs, subword tokenization,
data augmentation for class imbalance, ensembles of TF-IDF and embedding models.

## Authors
- Nur Faisal Ahmed Sourove
- Shahriar Mohammad
- Md Enayet Ali Labib
