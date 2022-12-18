"""
Basically, no matter what the data format is, e.g. image, sequence or others, we should pack some data into a batch
based on some rules. specifically, in NLP, we pack sentences to batches based on max sentence number or max tokens
number. That means,  max sentence or max tokens decides the batch numbers and shed an influence on gradient. So,
when we save the data state, we must save the max sentence or max tokens, the epoch and updates number. if we just
record the epoch and updates number, that's nonsense.


"""