# nanoColBERT
Simple implementation of ColBERT.

The official github repo: [Link](https://github.com/stanford-futuredata/ColBERT)

```bash
# accelerate launch --num_processes 4 --gpu_ids 0,1,2,3 train_colbert.py --per_device_train_batch_size 8
python train_colbert.py
```