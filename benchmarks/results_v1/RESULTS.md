# Model comparison

Same domain-grouped splits for every model; threshold = best F1 on val; metrics on the held-out test split (234,683 URLs). OpenPhish / phishunt = recall on live feeds excluded from training. URLs/s = end-to-end scoring of 20k raw URLs on an M-series laptop (CPU for classic models, MPS for neural).

## Full training data (1,835,018 URLs)

| Model | Family | ROC AUC | PR AUC | F1 | Precision | Recall | FPR | Brand F1 | OpenPhish | phishunt | Train s | URLs/s | Size MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stacked_lgbm | stacked (text + features) | 0.9799 | 0.9823 | 0.9224 | 0.9366 | 0.9088 | 6.14% | 0.9664 | 0.947 | 0.935 | 107 | 14,182 | 12.1 |
| stacked_xgb | stacked (text + features) | 0.9799 | 0.9823 | 0.9222 | 0.9342 | 0.9105 | 6.39% | 0.9660 | 0.950 | 0.902 | 108 | 14,448 | 12.0 |
| stacked_hgb | stacked (text + features) | 0.9794 | 0.9818 | 0.9222 | 0.9378 | 0.9070 | 5.99% | 0.9658 | 0.947 | 0.898 | 123 | 13,801 | 12.0 |
| char_cnn | neural (char CNN) | 0.9743 | 0.9773 | 0.9120 | 0.9294 | 0.8953 | 6.77% | 0.9624 | 0.953 | 0.938 | 2,048 | 19,591 | 1.7 |
| svm_ngrams | text n-grams | 0.9697 | 0.9734 | 0.9081 | 0.9283 | 0.8888 | 6.84% | 0.9582 | 0.967 | 0.926 | 71 | 40,515 | 18.2 |
| lr_ngrams | text n-grams | 0.9649 | 0.9684 | 0.8979 | 0.9182 | 0.8785 | 7.80% | 0.9529 | 0.963 | 0.935 | 65 | 40,428 | 10.9 |
| lgbm_features | lexical features | 0.9616 | 0.9667 | 0.8857 | 0.8973 | 0.8744 | 9.97% | 0.9458 | 0.963 | 0.981 | 17 | 20,700 | 1.8 |
| hgb_features | lexical features | 0.9613 | 0.9663 | 0.8872 | 0.9068 | 0.8684 | 8.90% | 0.9459 | 0.960 | 0.983 | 50 | 19,670 | 2.3 |
| xgb_features | lexical features | 0.9613 | 0.9664 | 0.8872 | 0.9110 | 0.8645 | 8.41% | 0.9467 | 0.953 | 0.981 | 24 | 21,057 | 2.2 |
| rf_features | lexical features | 0.9518 | 0.9582 | 0.8765 | 0.9024 | 0.8519 | 9.18% | 0.9407 | 0.963 | 0.962 | 36 | 21,579 | 158.4 |
| nb_ngrams | text n-grams | 0.9430 | 0.9495 | 0.8581 | 0.8625 | 0.8537 | 13.56% | 0.9366 | 0.983 | 0.886 | 52 | 39,217 | 62.8 |
| lr_features | lexical features | 0.9344 | 0.9438 | 0.8500 | 0.8524 | 0.8477 | 14.63% | 0.9328 | 0.947 | 0.952 | 32 | 20,821 | 0.0 |

## 200k-URL training subsample (fair comparison incl. DistilBERT)

| Model | Family | ROC AUC | PR AUC | F1 | Precision | Recall | FPR | Brand F1 | OpenPhish | phishunt | Train s | URLs/s | Size MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| distilbert_200k | transformer (fine-tuned) | 0.9786 | 0.9809 | 0.9184 | 0.9333 | 0.9039 | 6.44% | 0.9629 | 0.970 | 0.951 | 2,227 | 431 | 267.8 |
| stacked_hgb_200k | stacked (text + features) | 0.9758 | 0.9785 | 0.9140 | 0.9321 | 0.8965 | 6.51% | 0.9622 | 0.947 | 0.958 | 26 | 13,429 | 10.1 |
| char_cnn_200k | neural (char CNN) | 0.9628 | 0.9673 | 0.8903 | 0.9057 | 0.8755 | 9.09% | 0.9510 | 0.950 | 0.922 | 361 | 18,854 | 1.7 |
| lgbm_features_200k | lexical features | 0.9591 | 0.9644 | 0.8843 | 0.9018 | 0.8674 | 9.41% | 0.9439 | 0.967 | 0.977 | 7 | 20,971 | 1.4 |
| hgb_features_200k | lexical features | 0.9589 | 0.9639 | 0.8843 | 0.9066 | 0.8631 | 8.86% | 0.9432 | 0.960 | 0.981 | 11 | 20,154 | 1.7 |
| lr_ngrams_200k | text n-grams | 0.9561 | 0.9607 | 0.8835 | 0.8936 | 0.8736 | 10.36% | 0.9477 | 0.960 | 0.927 | 14 | 41,234 | 8.8 |
