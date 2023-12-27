import pickle
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import transformers
# transformers.logging.set_verbosity_error()
from transformers import BertTokenizer
import torch
from accelerate import PartialState
from model import ColBERT

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_path",default="data/collection.tsv")
    parser.add_argument("--encoding_batch_size",type=int,default=1024)
    parser.add_argument("--max_doclen",type=int,default=180)
    parser.add_argument("--pretrained_model_path",required=True)
    parser.add_argument("--output_dir",required=True)
    parser.add_argument("--max_embedding_num_per_shard",type=int,default=200_000)
    args = parser.parse_args()

    distributed_state = PartialState()
    device = distributed_state.device

    colbert = ColBERT.from_pretrained(args.pretrained_model_path,)
    colbert.eval()
    colbert.to(device)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path,use_fast=False)
    
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

    with distributed_state.split_between_processes(collections) as sharded_collections:
        
        sharded_collections = [sharded_collections[idx:idx+args.encoding_batch_size] for idx in range(0,len(sharded_collections),args.encoding_batch_size)]
        encoding_progress_bar = tqdm(total=len(sharded_collections), disable=not distributed_state.is_main_process,ncols=100,desc='encoding collections...')
        
        os.makedirs(args.output_dir,exist_ok=True)
        shard_id = 0
        doc_embeddings = []
        doc_embeddings_lengths = []
        
        for docs in sharded_collections:
            docs = ["[D] "+doc for doc in docs]
            model_input = tokenizer(docs,max_length=args.max_doclen,padding='max_length',return_tensors='pt',truncation=True).to(device)
            input_ids = model_input.input_ids
            attention_mask = model_input.attention_mask
            
            with torch.no_grad():
                doc_embedding = colbert.get_doc_embedding(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    return_list = True,
                )
            lengths = [doc.shape[0] for doc in doc_embedding]

            doc_embeddings.extend(doc_embedding)
            doc_embeddings_lengths.extend(lengths)
            encoding_progress_bar.update(1)

            if len(doc_embeddings) >= args.max_embedding_num_per_shard:
                doc_embeddings = torch.cat(doc_embeddings,dim=0)   
                torch.save(doc_embeddings,f'{args.output_dir}/collection_shard_{distributed_state.process_index}_{shard_id}.pt')
                pickle.dump(doc_embeddings_lengths,open(f"{args.output_dir}/length_shard_{distributed_state.process_index}_{shard_id}.pkl",'wb'))

                ## for new shard
                shard_id += 1
                doc_embeddings = []
                doc_embeddings_lengths = []

        if len(doc_embeddings) > 0:
            doc_embeddings = torch.cat(doc_embeddings,dim=0)
            torch.save(doc_embeddings,f'{args.output_dir}/collection_shard_{distributed_state.process_index}_{shard_id}.pt')
            pickle.dump(doc_embeddings_lengths,open(f"{args.output_dir}/length_shard_{distributed_state.process_index}_{shard_id}.pkl",'wb'))