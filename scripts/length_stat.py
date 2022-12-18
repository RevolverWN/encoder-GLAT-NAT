
src_corpus = "../distilled_data/iwslt14de-en/train.de"
tgt_corpus = "../distilled_data/iwslt14de-en/train.en"

total_src_length = 0
total_tgt_length = 0
sentences_num = 0

with open(src_corpus, 'r', encoding="utf-8") as f_src:
    with open(tgt_corpus, 'r', encoding="utf-8") as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            total_src_length += len(src_line.strip().split())
            total_tgt_length += len(tgt_line.strip().split())
            sentences_num += 1

diff = total_tgt_length - total_src_length
ave = diff / sentences_num
print("total target length - total source length = {}".format(diff))
print("average sentences length difference is {}".format(ave))