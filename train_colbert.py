## built-in
import math,logging,functools,os
import types
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"]='*.safetensors' ## not upload ckpt to wandb cloud

## third-party
from accelerate import Accelerator
from accelerate.logging import get_logger
import transformers
from transformers import (
    BertTokenizer,
)
transformers.logging.set_verbosity_error()
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

## own
from model import ColBERT,ColBERTConfig

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def set_seed(seed: int = 19980406):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_yaml_file(file_path):
    import yaml  
    with open(file_path, "r") as file:  
        config = yaml.safe_load(file)  
    return config  

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    ## adding args here for more control from CLI is possible
    parser.add_argument("--config_file",default='config/train_colbert_msmarco.yaml')
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k:v for k,v in vars(args).items() if v is not None}
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    return args


class MSMarcoDataset(torch.utils.data.Dataset):
    def __init__(self,query_data_path,pos_doc_data_path,neg_doc_data_path,
                 query_max_len,doc_max_len,num_samples,
                 ):
        self.queries  = np.memmap(query_data_path,  dtype=np.int16, mode='r', shape=(num_samples,query_max_len))  
        self.pos_docs = np.memmap(pos_doc_data_path,dtype=np.int16, mode='r', shape=(num_samples,doc_max_len))
        self.neg_docs = np.memmap(neg_doc_data_path,dtype=np.int16, mode='r', shape=(num_samples,doc_max_len))
        self.num_samples = num_samples  
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        return (self.queries[idx],self.pos_docs[idx],self.neg_docs[idx])


    @staticmethod
    def collate_fn(samples,tokenizer):

        def trim_padding(input_ids,padding_id):
            ## because we padding it to make length in the preprocess script
            ## we need to trim the padded sequences in a 2-dimensional tensor to the length of the longest non-padded sequence
            non_pad_mask = input_ids != padding_id
            non_pad_lengths = non_pad_mask.sum(dim=1)
            max_length = non_pad_lengths.max().item()
            trimmed_tensor = input_ids[:,:max_length]
            return trimmed_tensor

        queries  = [x[0] for x in samples]
        pos_docs = [x[1] for x in samples]
        neg_docs = [x[2] for x in samples]

        query_input_ids = torch.from_numpy(np.stack(queries).astype(np.int32))
        query_attention_mask = (query_input_ids != tokenizer.mask_token_id).int() ## not pad token

        doc_input_ids = torch.from_numpy(np.stack(pos_docs+neg_docs).astype(np.int32))
        doc_input_ids = trim_padding(doc_input_ids,padding_id = tokenizer.pad_token_id)
        doc_attetion_mask = (doc_input_ids != tokenizer.pad_token_id).int()


        return {
            'query_input_ids':query_input_ids,
            'query_attention_mask':query_attention_mask,

            "doc_input_ids":doc_input_ids,
            "doc_attention_mask":doc_attetion_mask,
        }

def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb',
        mixed_precision='fp16' if args.fp16 else 'no',
    )

    accelerator.init_trackers(
        project_name="colbert", 
        config=args,
    )
    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = wandb_tracker.run.dir

    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    q_mark,d_mark = "[Q]","[D]"
    additional_special_tokens = [q_mark,d_mark]
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens":additional_special_tokens,
        }
    )
    colbert_config = ColBERTConfig(
        dim = args.dim,
        similarity_metric = args.similarity_metric,
        mask_punctuation = args.mask_punctuation, 
    )
    colbert = ColBERT.from_pretrained(
        args.base_model,
        config = colbert_config,
        )
    colbert.resize_token_embeddings(len(tokenizer))
    colbert.train()
    colbert = torch.compile(colbert)

    train_dataset = MSMarcoDataset(
        args.query_data_path,
        args.pos_doc_data_path,
        args.neg_doc_data_path,
        args.query_max_len,args.doc_max_len,args.num_samples
        )
    train_collate_fn = functools.partial(MSMarcoDataset.collate_fn,tokenizer=tokenizer,)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=args.shuffle_train_set,
        collate_fn=train_collate_fn,
        num_workers=4,pin_memory=True
        )
    
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in colbert.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in colbert.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    
    colbert, optimizer, train_dataloader = accelerator.prepare(
        colbert, optimizer, train_dataloader,
    )

    loss_fct = nn.CrossEntropyLoss()
    
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = args.max_train_steps
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval,int) else int(args.val_check_interval * NUM_UPDATES_PER_EPOCH)
    total_loss = 0.0
    progress_bar_postfix_dict = {}

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Num Updates Per Epoch = {NUM_UPDATES_PER_EPOCH}")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    completed_steps = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=100)

    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed+epoch)
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        for batch in train_dataloader:
            with accelerator.accumulate(colbert):
                with accelerator.autocast():
                    scores  = colbert(**batch).view(2,-1).permute(1,0) #[per_device_train_batch_size,2]
                    labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
                    loss = loss_fct(scores,labels)
                    total_loss += loss.item()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    accelerator.log({"batch_loss": loss}, step=completed_steps)
                    accelerator.log({"average_loss": total_loss/completed_steps}, step=completed_steps)
                    progress_bar_postfix_dict.update(dict(loss=f"{total_loss/completed_steps:.4f}"))
                    progress_bar.set_postfix(progress_bar_postfix_dict)
                    
                    if completed_steps % EVAL_STEPS == 0:
                        if accelerator.is_local_main_process:
                            unwrapped_model = accelerator.unwrap_model(colbert)
                            unwrapped_model.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}"))
                            tokenizer.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}"))
                        accelerator.wait_for_everyone()
                    
                    if completed_steps > MAX_TRAIN_STEPS: break
    
    if accelerator.is_local_main_process:wandb_tracker.finish()
    accelerator.end_training()

if __name__ == '__main__':
    main()