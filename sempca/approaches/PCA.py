import argparse
import time

import numpy as np

from sempca.models import PCA
from sempca.preprocessing import cut_by_613, Preprocessor
from sempca.representations import FeatureExtractor


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset", default="HDFS", type=str, help="Target dataset. Default HDFS"
    )
    argparser.add_argument(
        "--parser",
        default="Drain",
        type=str,
        help="Select parser, please see parser list for detail. Default Drain.",
    )
    argparser.add_argument(
        "--n_components",
        type=int,
        default=21,
        help="Number of component after PCA, dynamic ratio if less than one.",
    )
    argparser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Anomaly Threshold parameter in PCA.",
    )

    args, extra_args = argparser.parse_known_args()
    dataset = args.dataset
    parser = args.parser
    n_components = args.n_components if args.n_components else 0.95
    anomaly_threshold = args.threshold

    processor = Preprocessor()
    train, _, test = processor.process_and_split(
        dataset=dataset, parsing=parser, template_encoding=None, cut_func=cut_by_613
    )
    train_inputs = []
    train_labels = np.zeros(len(train))
    for idx, inst in enumerate(train):
        train_inputs.append([int(x) for x in inst.sequence])
        label = int(processor.label2id[inst.label])
        train_labels[idx] = label

    test_inputs = []
    test_labels = np.zeros(len(test))
    for idx, inst in enumerate(test):
        test_inputs.append([int(x) for x in inst.sequence])
        label = int(processor.label2id[inst.label])
        test_labels[idx] = label

    feature_representor = FeatureExtractor()
    train_inputs = feature_representor.fit_transform(
        np.asarray(train_inputs), term_weighting="tf-idf"
    )
    test_inputs = feature_representor.transform(np.asarray(test_inputs))

    model = PCA(n_components=n_components)
    train_start = time.time()
    model.fit(train_inputs)
    train_end = time.time()
    model.logger.info("Training time: %.2f" % (train_end - train_start))
    test_start = time.time()
    _metrics = model.evaluate(
        test_inputs, test_labels, fixed_threshold=anomaly_threshold
    )
    time_end = time.time()

    model.logger.info("Prediction time: {:2f}s", time_end - test_start)


if __name__ == "__main__":
    main()
