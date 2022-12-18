dictionary:
constructor:
load from dict file: 
	input:path
build from corpus:
__init__: 
__len__:  for build the embedding
padding_id: for build the embedding
encode sentence: convert sentence to id
word2id : convert token to id
id2word : convert id to token
word2count: filter the min frequency to build the dictionary
save dict: save the dictionary to text file


dataset
constructor:
__init__: dicitionary
__getitem__: giving an index, return a encoded sentence
__len__:
batch_by_size: parameters: sentences lengths,  max_sentences, max_token
collate_fn:


Translation Task

