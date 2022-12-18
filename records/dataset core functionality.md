*dataset core functionality：**

__getitem__: 

input: index

output: token index sequence and others


__len__:

output: the length of the dataset


collater:
input: samples
output: collated samples
description: given a list of items from __getitem__ batched by dataloader internally, collate them to specific output,  we integrate this function in this class


num_tokens
input: index
output: size

num_tokens_vec
desc: vectorize operation for num_tokens


size: we don't know this method now, repeat with num_tokens


prefetch:
input: indices
output: 


batch_by_size
input: indices,  max_tokens, max_sentence


filter_indices_by_size



