# Model comparison

Same domain-grouped splits for every model; threshold = best F1 on val; metrics on the held-out test split (234,683 URLs). OpenPhish / phishunt = recall on live feeds excluded from training. URLs/s = end-to-end scoring of 20k raw URLs on an M-series laptop (CPU for classic models, MPS for neural).

## Full training data (2,399,377 URLs)

| Model | Family | ROC AUC | PR AUC | F1 | Precision | Recall | FPR | Brand F1 | OpenPhish | phishunt | PhreshPhish AUC | JPxxx AUC | DeepURLBench AUC | Train s | URLs/s | Size MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stacked_xgb | stacked (text + features) | 0.9707 | 0.9751 | 0.9104 | 0.9125 | 0.9083 | 9.70% | 0.9593 | 0.986 | 0.989 | 0.9882 | 0.9395 | 0.9438 | 156 | 1,825 | 13.5 |
| stacked_hgb | stacked (text + features) | 0.9690 | 0.9731 | 0.9095 | 0.9117 | 0.9073 | 9.78% | 0.9601 | 0.990 | 0.990 | 0.9878 | 0.9378 | 0.9422 | 189 | 1,833 | 13.4 |
| stacked_lgbm_domw | stacked (text + features) | 0.9686 | 0.9730 | 0.9084 | 0.9074 | 0.9094 | 10.33% | 0.9585 | 0.986 | 0.989 | 0.9882 | 0.9420 | 0.9451 | 144 | 1,835 | 13.1 |
| stacked_lgbm | stacked (text + features) | 0.9677 | 0.9713 | 0.9084 | 0.9120 | 0.9049 | 9.72% | 0.9592 | 0.983 | 0.987 | 0.9880 | 0.9393 | 0.9444 | 144 | 1,826 | 13.1 |
| stacked_lgbm_norecent | stacked (text + features) | 0.9675 | 0.9720 | 0.9078 | 0.9093 | 0.9063 | 10.07% | 0.9590 | 0.986 | 0.985 | 0.9763 | 0.9417 | 0.9430 | 142 | 1,824 | 13.1 |
| char_cnn | neural (char CNN) | 0.9605 | 0.9645 | 0.9005 | 0.9015 | 0.8995 | 10.94% | 0.9005 | 0.983 | 0.969 | 0.9841 | 0.9469 | 0.9145 | 1,466 | 23,758 | 1.7 |
| svm_ngrams | text n-grams | 0.9586 | 0.9645 | 0.8971 | 0.8951 | 0.8991 | 11.74% | 0.9518 | 0.969 | 0.963 | 0.9824 | 0.9471 | 0.9106 | 148 | 37,226 | 19.3 |
| lr_ngrams | text n-grams | 0.9462 | 0.9511 | 0.8830 | 0.8780 | 0.8881 | 13.74% | 0.9430 | 0.980 | 0.971 | 0.9818 | 0.9441 | 0.9171 | 86 | 38,152 | 11.0 |
| xgb_features | lexical features | 0.9448 | 0.9544 | 0.8666 | 0.8808 | 0.8530 | 12.86% | 0.9320 | 0.959 | 0.989 | 0.9765 | 0.9172 | 0.9193 | 44 | 1,937 | 3.4 |
| lgbm_features | lexical features | 0.9429 | 0.9515 | 0.8670 | 0.8826 | 0.8520 | 12.63% | 0.9324 | 0.956 | 0.988 | 0.9762 | 0.9169 | 0.9190 | 24 | 1,921 | 2.4 |
| hgb_features | lexical features | 0.9410 | 0.9499 | 0.8646 | 0.8705 | 0.8587 | 14.23% | 0.9314 | 0.956 | 0.989 | 0.9758 | 0.9148 | 0.9172 | 60 | 1,931 | 2.0 |
| rf_features | lexical features | 0.9352 | 0.9445 | 0.8591 | 0.8648 | 0.8534 | 14.86% | 0.9292 | 0.949 | 0.986 | 0.9715 | 0.9077 | 0.9124 | 48 | 1,933 | 184.0 |
| nb_ngrams | text n-grams | 0.9342 | 0.9443 | 0.8608 | 0.8672 | 0.8545 | 14.57% | 0.9220 | 0.959 | 0.947 | 0.9710 | 0.9326 | 0.8801 | 75 | 36,421 | 64.4 |
| lr_features | lexical features | 0.8956 | 0.9041 | 0.8246 | 0.8077 | 0.8423 | 22.34% | 0.9058 | 0.922 | 0.980 | 0.9472 | 0.9043 | 0.8472 | 79 | 1,922 | 0.0 |

## 200k-URL training subsample (fair comparison incl. DistilBERT)

| Model | Family | ROC AUC | PR AUC | F1 | Precision | Recall | FPR | Brand F1 | OpenPhish | phishunt | PhreshPhish AUC | JPxxx AUC | DeepURLBench AUC | Train s | URLs/s | Size MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| distilbert_200k | transformer (fine-tuned) | 0.9699 | 0.9749 | 0.9068 | 0.9034 | 0.9103 | 10.87% | 0.9068 | 0.990 | 0.972 | 0.9855 | 0.9559 | 0.9292 | 1,742 | 496 | 267.8 |
| stacked_hgb_200k | stacked (text + features) | 0.9645 | 0.9693 | 0.9002 | 0.9048 | 0.8957 | 10.50% | 0.9555 | 0.976 | 0.982 | 0.9850 | 0.9349 | 0.9324 | 39 | 1,804 | 9.6 |
| stacked_lgbm_200k | stacked (text + features) | 0.9645 | 0.9691 | 0.9013 | 0.9028 | 0.8998 | 10.79% | 0.9565 | 0.980 | 0.986 | 0.9852 | 0.9370 | 0.9355 | 36 | 1,791 | 9.5 |
| char_cnn_200k | neural (char CNN) | 0.9427 | 0.9491 | 0.8747 | 0.8721 | 0.8774 | 14.34% | 0.8747 | 0.983 | 0.961 | 0.9729 | 0.9343 | 0.8893 | 306 | 23,319 | 1.7 |
