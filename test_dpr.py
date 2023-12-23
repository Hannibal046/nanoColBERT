from utils import normalize_query
import csv
import faiss,pickle        
import numpy as np 
from tqdm import tqdm
from transformers import DPRQuestionEncoder,DPRQuestionEncoderTokenizer,BertModel,BertTokenizer
import torch
from utils.tokenizers import SimpleTokenizer
import unicodedata
import time
import transformers
transformers.logging.set_verbosity_error()

def normalize(text):
    return unicodedata.normalize("NFD", text)

def has_answer(answers,doc):
    tokenizer = SimpleTokenizer()
    doc = tokenizer.tokenize(normalize(doc)).words(uncased=True)
    for answer in answers:
        answer = tokenizer.tokenize(normalize(answer)).words(uncased=True)
        for i in range(0, len(doc) - len(answer) + 1):
                if answer == doc[i : i + len(answer)]:
                    return True
    return False

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia_path",default="downloads/data/wikipedia_split/psgs_w100.tsv")
    parser.add_argument("--nq_test_file",default="downloads/data/retriever/qas/nq-test.csv")
    parser.add_argument("--encoding_batch_size",type=int,default=32)
    parser.add_argument("--num_shards",type=int,default=8)
    parser.add_argument("--num_docs",type=int,default=21015324)
    parser.add_argument("--embedding_dir",required=True)
    parser.add_argument("--pretrained_model_path",required=True)
    args = parser.parse_args()

    ## load QA dataset
    query_col,answers_col=0,1
    queries,answers = [],[]
    with open(args.nq_test_file) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            queries.append(normalize_query(row[query_col]))
            answers.append(eval(row[answers_col]))
    queries = [queries[idx:idx+args.encoding_batch_size] for idx in range(0,len(queries),args.encoding_batch_size)]
    
    # make faiss index
    embedding_dimension = 768 
    index = faiss.IndexFlatIP(embedding_dimension)
    for idx in tqdm(range(args.num_shards),desc='building index from embedding...'):
        data = np.load(f"{args.embedding_dir}/wikipedia_shard_{idx}.npy")
        index.add(data)  

    ## load wikipedia passages
    id_col,text_col,title_col=0,1,2
    wiki_passages = []
    with open(args.wikipedia_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader,total=args.num_docs,desc="loading wikipedia passages..."):
            if row[id_col] == "id":continue
            wiki_passages.append(row[text_col].strip('"'))
    
    ## load query encoder
    if args.pretrained_model_path == 'facebook/dpr-question_encoder-single-nq-base':
        query_encoder = DPRQuestionEncoder.from_pretrained(args.pretrained_model_path)
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.pretrained_model_path)
    else:
        query_encoder = BertModel.from_pretrained(args.pretrained_model_path,add_pooling_layer=False)
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_encoder.to(device).eval()

    ## embed queries
    query_embeddings = []
    for query in tqdm(queries,desc='encoding queries...'):
        with torch.no_grad():
           query_embedding = query_encoder(**tokenizer(query,max_length=256,truncation=True,padding='max_length',return_tensors='pt').to(device))
        if isinstance(query_encoder,DPRQuestionEncoder):
            query_embedding = query_embedding.pooler_output
        else:
            query_embedding = query_embedding.last_hidden_state[:,0,:]
        query_embeddings.append(query_embedding.cpu().detach().numpy())
    query_embeddings = np.concatenate(query_embeddings,axis=0)

    ## retrieve top-k documents
    print("searching index ",end=' ')
    start_time = time.time()
    top_k = 100
    _,I = index.search(query_embeddings,top_k)
    print(f"takes {time.time()-start_time} s")

    hit_lists = []
    for answer_list,id_list in tqdm(zip(answers,I),total=len(answers),desc='calculating metrics...'):
        ## process single query
        hit_list = []
        for doc_id in id_list:
            doc = wiki_passages[doc_id]
            hit_list.append(has_answer(answer_list,doc))
        hit_lists.append(hit_list)

    top_k_hits = [0]*top_k
    best_hits = []
    for hit_list in hit_lists:
        best_hit = next((i for i, x in enumerate(hit_list) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
    
    top_k_ratio = [x/len(answers) for x in top_k_hits]
    
    for idx in range(top_k):
        if (idx+1) % 10 == 0:
            print(f"top-{idx+1} accuracy",top_k_ratio[idx])
