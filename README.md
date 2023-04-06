# The Modality Focusing Hypothesis

[**The Modality Focusing Hypothesis: Towards Understanding Crossmodal Knowledge Distillation**](https://openreview.net/forum?id=w0QXrZ3N-s)                                     
Zihui Xue, Zhengqi Gao, Sucheng Ren, Hang Zhao            
ICLR, 2023 (notable top-5%)  
[arXiv](https://arxiv.org/abs/2206.06487) | [website](https://zihuixue.github.io/MFH/index.html)

## Synthetic Gaussian

(1) Use synthetic Gaussian data to gain intuition about MFH

[Figure 2 and 3 in the paper] Generate multimodal data (data generation inspired from [here](https://github.com/lopezpaz/distillation_privileged_information.git)), and apply cross-modal KD.


```shell
python gauss/main.py
```
Experiment 1: vary γ

<img src="gauss/figs/exp1.png" alt="image" width="70%">

Experiment 2: vary α

<img src="gauss/figs/exp2.png" alt="image" width="70%">

(2) Verify the implications of MFH
```shell
python gauss/main_modify_gamma.py
```

[Table 2 in the paper] Modify γ in data, and observe the performance differences of cross-modal KD.

<img src="gauss/figs/table2.png" alt="image" width="70%" style="text-align: left;">

Three modes: (a) baseline-randomly keep some feature channels in x<sub>1</sub>; 
(b) if the ground truth data generation way is known, only keep "modality-general decisive" features channels in x<sub>1</sub>;
(c) if the data generation process is unknown, use Algorithm 1 to rank features based on "modality-general decisive" information.

Modifying data results in different γ, we then re-apply cross-modal KD to observe the performance differences. 
For mode (b) and (c), we observe that: although teacher performance downgrades significantly, student performance does not get affected. 
This helps verify the MFH implication and our proposed Alg. 1 as well.


## Citing MFH
```
@inproceedings{xue2023modality,
      title={The Modality Focusing Hypothesis: Towards Understanding Crossmodal Knowledge Distillation},
      author={Xue, Zihui and Gao, Zhengqi and Ren, Sucheng and Zhao, Hang},
      booktitle={ICLR},
      year={2023}
}
```
