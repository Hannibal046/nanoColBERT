from transformers import BertTokenizerFast
import numpy as np
from tqdm import tqdm
import os

def process_triplet(queries,poses,negs):

    ## get query_input_ids
    queries = [q_mark+" "+query for query in queries]
    query_input_ids = tokenizer(queries,padding='max_length',truncation=True,return_tensors='pt',max_length=query_max_len)['input_ids']
    query_input_ids[query_input_ids==tokenizer.pad_token_id] = tokenizer.mask_token_id ## called *query augmentation* in the paper

    ## get_doc_input_ids
    poses = [d_mark+" "+pos for pos in poses]
    pos_input_ids = tokenizer(poses,padding='max_length',truncation=True,return_tensors='pt',max_length=doc_max_len)['input_ids']
    
    negs = [d_mark+" "+neg for neg in negs]
    neg_input_ids = tokenizer(negs,padding='max_length',truncation=True,return_tensors='pt',max_length=doc_max_len)['input_ids']
    
    return query_input_ids,pos_input_ids,neg_input_ids



if __name__ == "__main__":

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    query_max_len = 32
    doc_max_len = 180
    triplet_path = "data/triples.train.small.tsv"
    batch_size = 10_000
    num_samples = 39780811

    q_mark,d_mark = "[Q]","[D]"
    additional_special_tokens = [q_mark,d_mark]
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens":additional_special_tokens,
        }
    )
    os.makedirs("data/processed",exist_ok=True)
    query_mmap = np.memmap('data/processed/queries.mmap', dtype='int16',mode='w+',shape=(num_samples,query_max_len))
    pos_mmap   = np.memmap("data/processed/pos_docs.mmap",dtype='int16',mode='w+',shape=(num_samples,doc_max_len))
    neg_mmap   = np.memmap("data/processed/neg_docs.mmap",dtype='int16',mode='w+',shape=(num_samples,doc_max_len))

    total = 0
    progress_bar = tqdm(range(num_samples),desc='processing triplet data...')
    with open(triplet_path) as f:
        queries,poses,negs = [],[],[]
        for line in f:
            query,pos,neg = line.strip().split("\t")
            queries.append(query)
            poses.append(pos)
            negs.append(neg)

            if len(queries) == batch_size:
                query_input_ids,pos_input_ids,neg_input_ids = process_triplet(queries,poses,negs)
                
                query_mmap[total:total+batch_size] = query_input_ids.numpy().astype(np.int16)  
                pos_mmap[  total:total+batch_size] = pos_input_ids.numpy().astype(np.int16)  
                neg_mmap[  total:total+batch_size] = neg_input_ids.numpy().astype(np.int16)  

                total += batch_size
                progress_bar.update(batch_size)
                queries,poses,negs = [],[],[]

        if len(queries) > 0:
            current_size = len(queries)
            query_input_ids,pos_input_ids,neg_input_ids = process_triplet(queries,poses,negs)
                    
            query_mmap[total:total+current_size] = query_input_ids.numpy().astype(np.int16)  
            pos_mmap[  total:total+current_size] = pos_input_ids.numpy().astype(np.int16)  
            neg_mmap[  total:total+current_size] = neg_input_ids.numpy().astype(np.int16)  

            assert current_size + total == num_samples

        query_mmap.flush()
        pos_mmap.flush()
        neg_mmap.flush()