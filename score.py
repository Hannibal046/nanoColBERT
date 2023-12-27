from collections import defaultdict
import json
import argparse

def get_mrr(qid2positives,qid2ranking,cutoff_rank=10):
    """
    qid2positives: {1:[99,13]}
    qid2ranking: {1:[99,1,32]} (sorted)
    """
    
    qid2mrr = {}
    for qid in qid2positives:
        positives = qid2positives[qid]
        ranked_pids = qid2ranking[qid]

        for rank,pid in enumerate(ranked_pids,start=1):
            if pid in positives:
                if rank <= cutoff_rank:
                    qid2mrr[qid] = 1.0/rank
                break

    return {
        f"mrr@{cutoff_rank}":sum(qid2mrr.values())/len(qid2ranking.keys())
    }

def get_recall(qid2positives,qid2ranking,cutoff_ranks=[50,200,1000,5000,10000]):
    """
    qid2positives: {1:[99,13]}
    qid2ranking: {1:[99,1,32]} (sorted)
    """
    qid2recall = {cutoff_rank:{} for cutoff_rank in cutoff_ranks}
    num_samples = len(qid2ranking.keys())
    
    for qid in qid2positives:
        positives = qid2positives[qid]
        ranked_pids = qid2ranking[qid]
        for rank,pid in enumerate(ranked_pids,start=1):
            if pid in positives:
                for cutoff_rank in cutoff_ranks:
                    if rank <= cutoff_rank:
                        qid2recall[cutoff_rank][qid] = qid2recall[cutoff_rank].get(qid, 0) + 1.0 / len(positives)
    
    return {
        f"recall@{cutoff_rank}":sum(qid2recall[cutoff_rank].values()) / num_samples
        for cutoff_rank in cutoff_ranks
    }

if __name__ == '__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument("--qrel_path",default="data/qrels.dev.small.tsv")
    parser.add_argument("--ranking_path")
    args = parser.parse_args()

    qid2positives = defaultdict(list)
    with open(args.qrel_path) as f:
        for line in f:
            qid,_,pid,label = [int(x) for x in line.strip().split()]
            assert label == 1
            qid2positives[qid].append(pid)

    qid2ranking = defaultdict(list)
    with open(args.ranking_path) as f:
        for line in f:
            qid,pid,rank = [int(x) for x in line.strip().split("\t")]
            qid2ranking[qid].append(pid)
    
    results = get_mrr(qid2positives,qid2ranking)
    results.update(get_recall(qid2positives,qid2ranking))

    print(json.dumps(results,indent=4))