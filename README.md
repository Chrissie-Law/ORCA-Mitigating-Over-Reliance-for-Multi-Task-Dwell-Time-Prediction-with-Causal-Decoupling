# ORCA: Mitigating Over-Reliance for Multi-Task Dwell Time Prediction with Causal Decoupling
> Official implementation of ORCA (CIKM 2025, Short Paper)

This is the official implementation of **ORCA**, a causal-decoupling framework for multi-task recommendation that mitigates CTR-driven over-reliance in dwell time prediction.  
Our paper is accepted at **CIKM 2025 (Short Paper)**.  

- :page_facing_up: [Paper on arXiv (**recommended**)](https://arxiv.org/abs/2508.16573)  
- :globe_with_meridians: [Official CIKM 2025 Paper (to appear)](PROJECT_PAGE_PLACEHOLDER)  

The official citation will be updated here upon publication.

---

## Overview


We observe that multi-task learning (MTL) models often skew dwell-time (DT) predictions toward the shortest and longest bins, severely under-representing moderate durations (Figure 1).  


<div align="center">
      <img src="./fig/heatmap.png" width="90%" alt="CTR–DT heatmaps of clicked impressions for the ground truth and classical models, and causal graph of MTL.">
      <p><strong><figcaption>Figure 1:</strong> CTR–DT heatmaps of clicked impressions for the ground truth and classical models, and causal graph of MTL. <i>X</i>: input, <i>C</i>: clicks, <i>T</i>: dwell time, <i>U</i>: unobserved factors.</p>
</div>

We attribute this *moderate-duration bin under-representation* to over-reliance on the CTR-DT spurious correlation. This over-reliance leads the model to treat CTR as a shortcut for DT, losing the nuanced information needed for moderate-duration dwell time.

Therefore, we propose the **Over-Reliance-decoupled CAusal multi-task (ORCA)** framework to explicitly
model and subtract CTR’s negative transfer while preserving its positive transfer. ORCA integrates two key components: (i) feature-level counterfactual intervention, and (ii) a task-interaction module with instance inverse-weighting, weakening CTR-mediated effect and restoring direct DT semantics. 

<div align="center">
      <img src="./fig/model.png" width="90%" alt="Overview of ORCA. Negative Dependency Extractor explicitly models the model’s over-reliance on CTR.">
      <p><strong><figcaption>Figure 2:</strong> Overview of ORCA. Negative Dependency Extractor explicitly models the model’s over-reliance on CTR.</p>
</div>

## Repository structure

```text
.
├── LICENSE             # License file
├── README.md           # Project introduction and usage instructions
├── data\               # Raw and processed datasets
├── fig\                # Figures for the paper and README
├── main.py             # Entry point; parses arguments
├── main_orca.py        # Handles ORCA-specific training and testing logic
├── model
│   ├── layer.py        # Common model layer definitions
│   └── orca.py         # ORCA model architecture
├── preprocess
│   ├── industry.py     # Data preprocessing for industry datasets
│   └── tenrec.py       # Data preprocessing for Tenrec dataset
├── requirements.txt    # Python dependencies
└── result\             # Output results
```

## Datasets

We evaluate our model on two industrial datasets and the public [Tenrec](https://tenrec0.github.io) dataset.  

All datasets should be placed in the `data` directory. The preprocessing scripts in the `preprocess` directory will automatically process and export the preprocessed datasets when running the main program.

## Run

You can run this model through:

```bash
# Run directly with default parameters 
python main.py

# Run with specific parameters
python main.py --dataset_name tenrec
               --model_name orca
               --learning_rate 0.001
               --batch_size 512
```


## Related Repositories

We adapted some baseline code from [MTReclib](https://github.com/easezyc/Multitask-Recommendation-Library).

## License
This project is licensed under the MIT License.