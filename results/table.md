# Sen1Floods11 subset results

Values are mean ± sample standard deviation across seeds. `undefined` means AUROC was not defined.

## In-domain test, both modalities

| Method | Condition | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|---|
| majority_class | full | 0.712 ± 0.000 | 0.416 ± 0.000 | 0.500 ± 0.000 |
| optical_indices | full | 0.934 ± 0.000 | 0.920 ± 0.000 | 0.960 ± 0.000 |
| optical_only | full | 0.939 ± 0.009 | 0.926 ± 0.010 | 0.952 ± 0.007 |
| sar_indices | full | 0.861 ± 0.000 | 0.835 ± 0.000 | 0.923 ± 0.000 |
| sar_only | full | 0.864 ± 0.006 | 0.830 ± 0.004 | 0.901 ± 0.005 |
| late_fusion | full | 0.928 ± 0.014 | 0.913 ± 0.015 | 0.956 ± 0.004 |
| late_fusion_moddrop | full | 0.928 ± 0.012 | 0.910 ± 0.014 | 0.952 ± 0.005 |
| cross_attention_fusion | full | 0.933 ± 0.010 | 0.919 ± 0.011 | 0.960 ± 0.005 |
| cross_attention_fusion_moddrop | full | 0.935 ± 0.009 | 0.921 ± 0.010 | 0.958 ± 0.003 |


## In-domain test, missing modality

| Method | Condition | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|---|
| late_fusion | full | 0.928 ± 0.014 | 0.913 ± 0.015 | 0.956 ± 0.004 |
| late_fusion | optical_missing | 0.288 ± 0.000 | 0.223 ± 0.000 | 0.851 ± 0.043 |
| late_fusion | sar_missing | 0.348 ± 0.135 | 0.296 ± 0.163 | 0.948 ± 0.006 |
| late_fusion_moddrop | full | 0.928 ± 0.012 | 0.910 ± 0.014 | 0.952 ± 0.005 |
| late_fusion_moddrop | optical_missing | 0.819 ± 0.039 | 0.770 ± 0.028 | 0.866 ± 0.007 |
| late_fusion_moddrop | sar_missing | 0.917 ± 0.028 | 0.901 ± 0.029 | 0.938 ± 0.004 |
| cross_attention_fusion | full | 0.933 ± 0.010 | 0.919 ± 0.011 | 0.960 ± 0.005 |
| cross_attention_fusion | optical_missing | 0.288 ± 0.000 | 0.223 ± 0.000 | 0.895 ± 0.005 |
| cross_attention_fusion | sar_missing | 0.370 ± 0.076 | 0.337 ± 0.104 | 0.943 ± 0.006 |
| cross_attention_fusion_moddrop | full | 0.935 ± 0.009 | 0.921 ± 0.010 | 0.958 ± 0.003 |
| cross_attention_fusion_moddrop | optical_missing | 0.843 ± 0.036 | 0.817 ± 0.031 | 0.897 ± 0.005 |
| cross_attention_fusion_moddrop | sar_missing | 0.898 ± 0.025 | 0.882 ± 0.026 | 0.938 ± 0.007 |


## Bolivia geographic hold-out, both modalities

| Method | Condition | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|---|
| majority_class | full | 0.668 ± 0.000 | 0.400 ± 0.000 | 0.500 ± 0.000 |
| optical_indices | full | 0.818 ± 0.000 | 0.812 ± 0.000 | 0.960 ± 0.000 |
| optical_only | full | 0.855 ± 0.018 | 0.846 ± 0.017 | 0.957 ± 0.007 |
| sar_indices | full | 0.913 ± 0.000 | 0.898 ± 0.000 | 0.979 ± 0.000 |
| sar_only | full | 0.874 ± 0.002 | 0.845 ± 0.004 | 0.969 ± 0.006 |
| late_fusion | full | 0.851 ± 0.018 | 0.842 ± 0.017 | 0.957 ± 0.002 |
| late_fusion_moddrop | full | 0.877 ± 0.018 | 0.862 ± 0.018 | 0.954 ± 0.009 |
| cross_attention_fusion | full | 0.860 ± 0.012 | 0.853 ± 0.011 | 0.976 ± 0.001 |
| cross_attention_fusion_moddrop | full | 0.916 ± 0.017 | 0.907 ± 0.017 | 0.976 ± 0.002 |


## Bolivia geographic hold-out, missing modality

| Method | Condition | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|---|
| late_fusion | full | 0.851 ± 0.018 | 0.842 ± 0.017 | 0.957 ± 0.002 |
| late_fusion | optical_missing | 0.332 ± 0.000 | 0.249 ± 0.000 | 0.922 ± 0.034 |
| late_fusion | sar_missing | 0.337 ± 0.011 | 0.257 ± 0.018 | 0.951 ± 0.005 |
| late_fusion_moddrop | full | 0.877 ± 0.018 | 0.862 ± 0.018 | 0.954 ± 0.009 |
| late_fusion_moddrop | optical_missing | 0.850 ± 0.030 | 0.808 ± 0.049 | 0.950 ± 0.014 |
| late_fusion_moddrop | sar_missing | 0.815 ± 0.068 | 0.808 ± 0.063 | 0.948 ± 0.013 |
| cross_attention_fusion | full | 0.860 ± 0.012 | 0.853 ± 0.011 | 0.976 ± 0.001 |
| cross_attention_fusion | optical_missing | 0.332 ± 0.000 | 0.249 ± 0.000 | 0.963 ± 0.005 |
| cross_attention_fusion | sar_missing | 0.332 ± 0.000 | 0.249 ± 0.000 | 0.946 ± 0.008 |
| cross_attention_fusion_moddrop | full | 0.916 ± 0.017 | 0.907 ± 0.017 | 0.976 ± 0.002 |
| cross_attention_fusion_moddrop | optical_missing | 0.915 ± 0.008 | 0.902 ± 0.012 | 0.965 ± 0.006 |
| cross_attention_fusion_moddrop | sar_missing | 0.766 ± 0.058 | 0.762 ± 0.055 | 0.940 ± 0.017 |

