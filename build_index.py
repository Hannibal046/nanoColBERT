import faiss
import argparse
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir")
    parser.add_argument("--dim")
    parser.add_argument("--partitions")
    parser.add_argument("--sample_ratio")

    args = parser.parse_args()

    embedding_files = [os.path.join(args.embedding_dir,x) for x in os.listdir(args.embedding_dir)]
    embedding_files.sort(key=lambda x:os.path.basename(x).split(".")[0].split("_")[-1])

    embeddings = [np.load(x) for x in embedding_files]
    embeddings = np.concatenate(embeddings,axis=0)

    num_embeddings = embeddings.shape[0]

    ## build index
    quantizer = faiss.IndexFlatL2(args.dim)
    index = faiss.IndexIVFPQ(quantizer, args.dim, args.partitions, 16, 8)

    ## training
    gpu_resource = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = False
    co.indicesOptions = faiss.INDICES_CPU
    co.verbose = True
    co.shard = True

    gpu_quantizer = faiss.index_cpu_to_gpu(gpu_resource, 0, quantizer, co)
    gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, index, co)
    
    sampled_embeddings = embeddings[np.randint(0, high=num_embeddings, size=(int(num_embeddings * args.sample_ratio),))]
    gpu_index.train(sampled_embeddings)

    ## add
    batch_size = 10_000
    for idx in range(0,num_embeddings,batch_size):
        gpu_index.add(embeddings[idx:idx+batch_size,:])
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)

    ## save
    cpu_index.save(args.output_path)
