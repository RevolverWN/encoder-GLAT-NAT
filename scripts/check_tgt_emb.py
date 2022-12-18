import argparse
import collections
import os
import re

import torch
from fairseq.file_io import PathManager


def check_targt_embedding(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    tgt_emb = []
    params_num = []
    for fpath in inputs:
        with PathManager.open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )

        model_params = state["model"]
        tgt_emb.append(model_params['decoder.token_emb.weight'])
        params_num.append(model_params['decoder.token_emb.weight'].numel())

    num = (tgt_emb[0] == tgt_emb[1]).sum()
    print(num)
    print(params_num[0])


def main():
    parser = argparse.ArgumentParser(
        description="Tool to average the params of input checkpoints to "
        "produce a new checkpoint",
    )
    # fmt: off
    parser.add_argument("--ckpt_dir", type=str, help="checkpoint save path")
    parser.add_argument('--ckpt_name', required=True, nargs='+',
                        help='Input checkpoint file paths.')

    # fmt: on
    args = parser.parse_args()
    print(args)

    inputs = [os.path.join(args.ckpt_dir, path) for path in args.ckpt_name]

    check_targt_embedding(inputs)


if __name__ == "__main__":
    main()