[_Are Layout-Infused Language Models Robust to Layout Distribution Shifts? A Case Study with Scientific Documents_](https://aclanthology.org/2023.findings-acl.844.pdf), presented at ACL Findings 2023.

This repo is built on top of [VILA](https://github.com/allenai/VILA).

## Installation
```
git clone git@github.com:cchen23/layout_distribution_shift.git
cd layout_distribution_shift
conda create -n lds python=3.6
conda activate lds
pip install -r requirements.txt
```

## Getting Started

### Data download:
To use models that require document images, download the images [here](https://ai2-s2-research-public.s3.us-west-2.amazonaws.com/layout-distribution-shift-acl-2023/all_images_20230524.tar.gz) and place them in `layout_distribution_shift/data/grotoap2_images/`

### Performing experiments:
Scripts to perform the initial training phase and few-shot adaptation are in `run_scripts/`.

To perform the initial training phase:
```
bash train_publisher_splits.sh [test_publisher_name] [learning_rate] [model_name] [fewshot_lr] [random_seed]
```

After performing the initial training phase, to perform few-shot fine-tuning for a specific few-shot episode:
```
bash train_publisher_splits_fewshot_only.sh [test_publisher_name] [initial_training_learning_rate] [model_name] [fewshot_learning_rate] [fewshot_episode_num] [random_seed]
```

## References
```
@inproceedings{chen-layout:2023:ACL,
  author={Chen, Catherine and Shen, Zejiang and Klein, Dan and Stanovsky, Gabriel and Downey, Doug and Lo, Kyle},
  title={Are Layout-Infused Language Models Robust to Layout Distribution Shifts? A Case Study with Scientific Documents},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  year={2023}
}
```
