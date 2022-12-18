src_corpus = "../distilled_data/iwslt14de-en/test.de"
tgt_corpus = "../distilled_data/iwslt14de-en/test.en"

diff = -1
band_size = 3
pred_lengths_beam = [[] for i in range(-band_size, band_size + 1)]
band_diff = [i + diff for i in list(range(-band_size, band_size + 1))]
ground_truth = []
acc = []

with open(src_corpus, 'r', encoding="utf-8") as f_src:
    with open(tgt_corpus, 'r', encoding="utf-8") as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_length = len(src_line.strip().split())
            ground_truth.append(len(tgt_line.strip().split()))
            for i, delta in enumerate(band_diff):
                pred_lengths_beam[i].append(src_length + delta)

for pred_length in pred_lengths_beam:
    count = 0
    for i, j in zip(pred_length, ground_truth):
        if i == j:
            count += 1
    acc.append(count / len(ground_truth))

print(acc)
