import csv
from tqdm import tqdm
import os
import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    BertTokenizer,
    BertModel,
    )
import torch
import numpy as np
from accelerate import PartialState

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia_path",default="downloads/data/wikipedia_split/psgs_w100.tsv")
    parser.add_argument("--num_docs",type=int,default=21015324)
    parser.add_argument("--encoding_batch_size",type=int,default=1024)
    parser.add_argument("--pretrained_model_path",required=True)
    parser.add_argument("--output_dir",required=True)
    args = parser.parse_args()

    distributed_state = PartialState()
    device = distributed_state.device

    ## load encoder
    if args.pretrained_model_path == 'facebook/dpr-ctx_encoder-single-nq-base':
        doc_encoder = DPRContextEncoder.from_pretrained(args.pretrained_model_path)
        tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.pretrained_model_path)
    else:
        doc_encoder = BertModel.from_pretrained(args.pretrained_model_path,add_pooling_layer=False)
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    doc_encoder.eval()
    doc_encoder.to(device)


    ## load wikipedia passages
    progress_bar = tqdm(total=args.num_docs, disable=not distributed_state.is_main_process,ncols=100,desc='loading wikipedia...')
    id_col,text_col,title_col=0,1,2
    wikipedia = []
    with open(args.wikipedia_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[id_col] == "id":continue
            wikipedia.append(
                [row[title_col],row[text_col].strip('"')]
            )
            progress_bar.update(1)

    with distributed_state.split_between_processes(wikipedia) as sharded_wikipedia:
        
        sharded_wikipedia = [sharded_wikipedia[idx:idx+args.encoding_batch_size] for idx in range(0,len(sharded_wikipedia),args.encoding_batch_size)]
        encoding_progress_bar = tqdm(total=len(sharded_wikipedia), disable=not distributed_state.is_main_process,ncols=100,desc='encoding wikipedia...')
        doc_embeddings = []
        for data in sharded_wikipedia:
            title = [x[0] for x in data]
            passage = [x[1] for x in data]
            model_input = tokenizer(title,passage,max_length=256,padding='max_length',return_tensors='pt',truncation=True).to(device)
            with torch.no_grad():
                if isinstance(doc_encoder,BertModel):
                    CLS_POS = 0
                    output = doc_encoder(**model_input).last_hidden_state[:,CLS_POS,:].cpu().numpy()
                else:
                    output = doc_encoder(**model_input).pooler_output.cpu().numpy()
            doc_embeddings.append(output)
            encoding_progress_bar.update(1)
        doc_embeddings = np.concatenate(doc_embeddings,axis=0)
        os.makedirs(args.output_dir,exist_ok=True)
        np.save(f'{args.output_dir}/wikipedia_shard_{distributed_state.process_index}.npy',doc_embeddings)


