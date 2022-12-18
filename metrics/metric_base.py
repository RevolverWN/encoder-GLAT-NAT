from typing import List, Union

import sacrebleu
import torch
import torch.distributed as dist
from torch import Tensor

from dictionary import Dictionary
from metrics import register_metric
import metrics
from sacrebleu.metrics import BLEUScore
from tasks.trans_utils import remove_invalid_token, remove_bpe, remove_repeat_token
from utils import tensors_all_reduce


class MetricList(object):
    """
    we can design custom behavior on every stage of train, dev, test
    """

    def __init__(self):
        self.metrics_dict = {}

    def update(self, ground_truth, prediction):
        for metric in self.metrics_dict.values():
            metric.update(ground_truth, prediction)

    def result(self):
        for metric in self.metrics_dict.values():
            metric.result()

    def add_metric(self, name, metric_instance):
        self.metrics_dict[name] = metric_instance

    def items(self):
        return self.metrics_dict.items()

    def reset(self):
        for metric in self.metrics_dict.values():
            metric.reset()


class MetricBase(object):

    def update(self, ground_truth, prediction):
        pass

    def result(self):
        pass


@register_metric("BLEU")
class BLEU(MetricBase):
    def __init__(self, beam_size: int, tgt_dict: Dictionary, device):
        self.beam_size = beam_size
        self.tgt_dict = tgt_dict
        self.device = device
        self.bleu_list = [[] for i in range(self.beam_size)]

    def update(self, ground_truth: Tensor, prediction: Tensor) -> None:
        """

        :param ground_truth: [batch_size, max_seq_len]
        :param prediction: [batch_size * beam_size, max_seq_len]
        """
        hypothesis = prediction.cpu().numpy().tolist()
        hypothesis = remove_invalid_token(hypothesis,
                                          [self.tgt_dict.padding_id, self.tgt_dict.bos_id,
                                           self.tgt_dict.eos_id])
        hypothesis = list(map(self.tgt_dict.decode_sentence, hypothesis))
        hypothesis = [remove_bpe(sent) for sent in hypothesis]
        # hypothesis = [remove_repeat_token(sent) for sent in hypothesis]

        targets = ground_truth.cpu().numpy().tolist()
        targets = remove_invalid_token(targets, [self.tgt_dict.padding_id, self.tgt_dict.bos_id,
                                                 self.tgt_dict.eos_id])
        targets = list(map(self.tgt_dict.decode_sentence, targets))
        targets = [remove_bpe(sent) for sent in targets]

        for length_idx in range(self.beam_size):
            length_idx_batch_hypo = hypothesis[length_idx::self.beam_size]
            bleu: BLEUScore = sacrebleu.corpus_bleu(length_idx_batch_hypo, [targets], force=True)
            self.bleu_list[length_idx].append(bleu)

    def result(self):
        all_length_bleu = []

        for length_bleu_collection in self.bleu_list:
            correct, total = [], []
            sys_len = torch.tensor(0, dtype=torch.int64, device=self.device)
            ref_len = torch.tensor(0, dtype=torch.int64, device=self.device)
            for bleu in length_bleu_collection:
                if bleu:
                    correct.append(bleu.counts)
                    total.append(bleu.totals)
                    sys_len += bleu.sys_len
                    ref_len += bleu.ref_len

            correct = torch.tensor(correct, device=self.device).sum(dim=0)
            total = torch.tensor(total, device=self.device).sum(dim=0)

            tensors_all_reduce(correct)
            tensors_all_reduce(total)
            dist.all_reduce(sys_len)
            dist.all_reduce(ref_len)

            score = sacrebleu.BLEU.compute_bleu(correct.tolist(), total.tolist(), sys_len.item(), ref_len.item(),
                                                smooth_method='exp')
            all_length_bleu.append(score.score)

        max_bleu = max(all_length_bleu)
        return max_bleu

    def reset(self):
        self.bleu_list = [[] for i in range(self.beam_size)]


@register_metric("length_accuracy")
class LengthAccuracy(MetricBase):
    def __init__(self, beam_size: int, tgt_dict: Dictionary, device: int):
        self.beam_size = beam_size
        self.tgt_dict = tgt_dict
        self.device = device

        self.total_samples = 0
        self.length_acc_num = [[] for i in range(self.beam_size)]

    def update(self, ground_truth: Tensor, prediction: Tensor) -> None:
        """
        :param ground_truth: [batch_size, max_seq_len]
        :param prediction: [batch_size * beam_size, max_seq_len]
        """
        self.total_samples += len(ground_truth)

        hypothesis = prediction.cpu().numpy().tolist()
        hypothesis = remove_invalid_token(hypothesis, [self.tgt_dict.padding_id])
        hypo_length = [len(sample) for sample in hypothesis]

        targets = ground_truth.cpu().numpy().tolist()
        targets = remove_invalid_token(targets, [self.tgt_dict.padding_id, self.tgt_dict.bos_id,
                                                 self.tgt_dict.eos_id])
        tgt_length = [len(sample) for sample in targets]

        for i in range(self.beam_size):
            beam_idx_pred = hypo_length[i::self.beam_size]
            count = self.compute_acc_num(tgt_length, beam_idx_pred)
            self.length_acc_num[i].append(count)

    def result(self):
        """
        we first merge all the batch accuracy for every length_beam_size in every GPU,
        then we merge length_beam_size on every GPU
        """
        length_acc_num = torch.tensor(self.length_acc_num, dtype=torch.float64, device=self.device)
        total_samples = torch.tensor(self.total_samples, dtype=torch.float64, device=self.device)

        length_acc_beam = length_acc_num.sum(dim=1)
        dist.all_reduce(length_acc_beam)
        dist.all_reduce(total_samples)

        acc = length_acc_beam / total_samples
        max_acc = max(acc)
        return max_acc

    @staticmethod
    def compute_acc_num(ground_truth: List, prediction: List):
        """
        :param ground_truth: [batch_size]
        :param prediction: [batch_size]
        """
        count = 0
        for gt, pred in zip(ground_truth, prediction):
            if gt == pred:
                count += 1
        return count

    def reset(self):
        self.total_samples = 0
        self.length_acc_num = [[] for i in range(self.beam_size)]
