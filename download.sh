mkdir data
cd data

wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz

tar -xzvf triples.train.small.tar.gz
tar -xzvf collectionandqueries.tar.gz
rm *.gz