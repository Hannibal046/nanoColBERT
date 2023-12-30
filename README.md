# nanoColBERT

This repo provides a simple implementation of [ColBERT-v1](https://arxiv.org/abs/2004.12832) model.

The official github repo: [Link](https://github.com/stanford-futuredata/ColBERT) (v1 branch)

ColBERT is a powerful late-interaction model that could perform both retrieval and reranking.
![ColBERT](assets/ColBERT.png)
## Get Started
```
conda create -n nanoColBERT python=3.8 && conda activate nanoColBERT
## install torch and faiss according to your CUDA version
pip install -r requirements.txt 
```
Configure `wandb` and `accelerate`
```bash
wandb login
accelerate config
```
After everything setup, just launch the whole process with:
```bash
bash scripts/download.sh
bash scripts/run_colbert.sh
```
It would first download the data, preprocess the data, train the model, index with faiss, conduct retrieval and calculate the score.

## Results

This is our reproduced results:

|             | **MRR@10** | **Recall@50** | **Recall@200** | **Recall@1000** |
|:-----------:|:----------:|:-------------:|:--------------:|:---------------:|
|   [Reported](https://arxiv.org/pdf/2004.12832.pdf)  |    36.0    |      82.9     |      92.3      |       96.8      |
| nanoColBERT |    36.0    |      83.3     |      91.9      |       96.3      |

**Please be aware that this repository serves solely as a conceptual guide and has not been heavily optimized for efficiency**

The following reveals the duration of each step:
|    **Step**   | **Duration** | **Remark** |
|:-------------:|:------------:|----------|
|    tsv2mmap   |    3h5min    |            |
|     train     |    8h54min   |   400k steps on 1*A100   |
| doc2emebdding |     56min    |   8*A100   |
|  build_index  |     21min    |   30% training data with IVFPQ on 1*A100   |
|    retrieve   |     17min    |   6980 samples on 1*A100 |


## Pretrained Ckpt
We also provide our trained model on the Huggingface Space and you could simply use it with:
```python
from model import ColBERT
from transformers import BertTokenizer

pretrained_model = "nanoColBERT/ColBERTv1"
model = ColBERT.from_pretrained(pretrained_model)
tokenizer = BertTokenizer.from_pretrained(pretrained_model) 
```