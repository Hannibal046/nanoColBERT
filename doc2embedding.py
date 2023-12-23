import csv
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    BertModel,
    AutoTokenizer,
    )
import torch
import numpy as np
from accelerate import PartialState
from model import ColBERT,ColBERTConfig

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_path",default="data/collection.tsv")
    parser.add_argument("--num_docs",type=int,default=8841823)
    parser.add_argument("--encoding_batch_size",type=int,default=1024)
    parser.add_argument("--mask_punctuation",type=bool,default=True)
    parser.add_argument("--dim",type=int,default=128)
    parser.add_argument("--max_doclen",type=int,default=180)
    parser.add_argument("--pretrained_model_path",required=True)
    parser.add_argument("--output_dir",required=True)
    args = parser.parse_args()

    distributed_state = PartialState()
    device = distributed_state.device

    config = ColBERTConfig(mask_punctuation=True,dim=128,similarity_metric='l2',vocab_size=30524)
    colbert = ColBERT.from_pretrained(
        args.pretrained_model_path,
        config = config,
        )
    colbert.eval()
    colbert.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path,use_fast=False)
    
    progress_bar = tqdm(total=args.num_docs, disable=not distributed_state.is_main_process,ncols=100,desc='loading collection...')
    collections = []
    with open(args.collection_path) as f:
        for line in f:
            line_parts = line.strip().split("\t") 
            pid, passage, *other = line_parts
            assert len(passage) >= 1

            if len(other) >= 1:
                title, *_ = other
                passage = title + " | " + passage
            
            collections.append(passage)
            progress_bar.update(1)

    with distributed_state.split_between_processes(collections) as sharded_collections:
        
        sharded_collections = [sharded_collections[idx:idx+args.encoding_batch_size] for idx in range(0,len(sharded_collections),args.encoding_batch_size)]
        encoding_progress_bar = tqdm(total=len(sharded_collections), disable=not distributed_state.is_main_process,ncols=100,desc='encoding collections...')
        doc_embeddings = []
        for docs in sharded_collections:
            model_input = tokenizer(docs,max_length=args.max_doclen,padding='max_length',return_tensors='pt',truncation=True).to(device)
            with torch.no_grad():
                output = colbert.get_doc_embedding(
                    input_ids = model_input.input_ids,
                    attention_mask = model_input.attention_mask
                    ).cpu().numpy()
            doc_embeddings.append(output)
            encoding_progress_bar.update(1)
        doc_embeddings = np.concatenate(doc_embeddings,axis=0)
        os.makedirs(args.output_dir,exist_ok=True)
        np.save(f'{args.output_dir}/collection_shard_{distributed_state.process_index}.npy',doc_embeddings)


