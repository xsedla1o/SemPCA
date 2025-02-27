import argparse

from sempca.models import PCAPlusPlus
from sempca.preprocessing import Preprocessor, cut_by_613
from sempca.representations import SequentialAdd, TemplateTfIdf


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
        default=20,
        help="Number of component after PCA, dynamic ratio if less than one.",
    )
    argparser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Anomaly Threshold parameter in PCA.",
    )
    argparser.add_argument(
        "--c_alpha",
        type=float,
        default=3.2905,
        help="Anomaly Threshold parameter in PCA.",
    )

    args, extra_args = argparser.parse_known_args()
    dataset = args.dataset
    parser = args.parser
    n_components = args.n_components if args.n_components else 0.95
    anomaly_threshold = args.threshold
    c_alpha = args.c_alpha

    template_encoder = TemplateTfIdf()

    preprocessor = Preprocessor()
    train, _, test = preprocessor.process(
        dataset=dataset,
        parsing=parser,
        template_encoding=template_encoder.present,
        cut_func=cut_by_613,
    )

    test_labels = [int(preprocessor.label2id[inst.label]) for inst in test]

    sequential_encoder = SequentialAdd(preprocessor.embedding)
    train_inputs = sequential_encoder.transform(train)
    test_inputs = sequential_encoder.transform(test)

    model = PCAPlusPlus(n_components=n_components, c_alpha=c_alpha)
    model.fit(train_inputs)
    model.evaluate(test_inputs, test_labels, fixed_threshold=anomaly_threshold)

    model.logger.info("All done.")


if __name__ == "__main__":
    main()
