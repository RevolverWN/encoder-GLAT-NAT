import argparse
import logging
import os
import sys

import options
from dictionary import Dictionary
from dataset.dataset import IndexDataset

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger("preprocess")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    options.get_preprocess_options(parser)
    args = parser.parse_args()

    os.makedirs(args.destdir, exist_ok=True)

    if not args.src_dict and not args.tgt_dict:
        if args.joined_dictionary:
            if args.tgt_min_frequency:
                assert args.tgt_min_frequency == args.src_min_frequency, \
                    """joined dictionary need src_min_frequency only, if tgt_min_frequency assigned, please assure they 
                    have same value"""
            tgt_dict = src_dict = Dictionary.build_dictionary([args.source_corpus, args.target_corpus],
                                                              args.src_min_frequency)
        else:
            src_dict = Dictionary.build_dictionary([os.path.join(args.corpus, "train." + args.src_lang)],
                                                   args.src_min_frequency)
            tgt_dict = Dictionary.build_dictionary([os.path.join(args.corpus, "train." + args.tgt_lang)],
                                                   args.tgt_min_frequency)

        src_dict.save_dict(os.path.join(args.destdir, "dict." + args.src_lang + ".txt"))
        tgt_dict.save_dict(os.path.join(args.destdir, "dict." + args.tgt_lang + ".txt"))
    else:
        src_dict = Dictionary.load_dictionary_from_file(args.src_dict)
        tgt_dict = Dictionary.load_dictionary_from_file(args.tgt_dict)

    # create binary files
    for split in [args.train_prefix, args.valid_prefix, args.test_prefix]:
        src_corpus_path = os.path.join(args.corpus, split + "." + args.src_lang)
        tgt_corpus_path = os.path.join(args.corpus, split + "." + args.tgt_lang)
        src_index_path = os.path.join(args.destdir, split + "." + args.src_lang)
        tgt_index_path = os.path.join(args.destdir, split + "." + args.tgt_lang)

        IndexDataset.create_index_files(src_index_path, src_corpus_path, src_dict)
        IndexDataset.create_index_files(tgt_index_path, tgt_corpus_path, tgt_dict)
        logger.info("write {} corpus to {} as binary files".format(src_corpus_path, src_index_path))
        logger.info("write {} corpus to {} as binary files".format(tgt_corpus_path, tgt_index_path))


if __name__ == '__main__':
    main()
