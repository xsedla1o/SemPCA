import os
import sys
from collections import Counter
from typing import Type, Optional

from sempca.entities.instances import Instance
from sempca.preprocessing import BGLLoader, HDFSLoader, SpiritLoader
from sempca.preprocessing.loader.basic import DataPaths, BasicDataLoader
from sempca.utils import tqdm, get_logger

set2dataloader = {
    "HDFS": HDFSLoader,
    "BGL": BGLLoader,
    "BGLSample": BGLLoader,
    "Spirit": SpiritLoader,
}


class Preprocessor:
    def __init__(self):
        self.logger = get_logger("Preprocessor")
        self.train_event2idx = {}
        self.test_event2idx = {}
        self.id2label = {}
        self.label2id = {}
        self.templates = []
        self.embedding = None
        self.dataset = None
        self.parsing = None
        self.tag2id = {"Normal": 0, "Anomalous": 1}
        self.id2tag = {0: "Normal", 1: "Anomalous"}

    def process_and_split(self, dataset, parsing, template_encoding, cut_func):
        """
        Preprocess approach, log loading, parsing and cutting.
        Please be noted that if you want to add more datasets or parsers, you should modify here.
        :param dataset: Specified dataset
        :param parsing: Specified log parser, Drain now supported.
        :param template_encoding: Semantic representation functio for log templates.
        :param cut_func: Cutting function for all instances.
        :return: Train, Dev and Test data in list of instances.
        """
        self.paths = DataPaths(dataset_name=dataset, parser_name=parsing)
        self.base = self.paths.processed_out_dir

        dataloader = self.get_dataloader(dataset)(
            self.paths, semantic_repr_func=template_encoding
        )

        dataloader.parse(parsing)
        instances = self.generate_instances(dataloader)
        return self._cut_instances(instances, cut_func=cut_func)

    @staticmethod
    def get_dataloader(dataset: str) -> Type[BasicDataLoader]:
        if dataset not in set2dataloader.keys():
            print("Dataset %s not supported." % dataset, file=sys.stderr)
            raise NotImplementedError
        return set2dataloader[dataset]

    def _cut_instances(self, instances, cut_func=None):
        os.makedirs(self.base, exist_ok=True)

        train_file = self.base / "train"
        dev_file = self.base / "dev"
        test_file = self.base / "test"

        train, dev, test = cut_func(instances)

        self.label_distribution(train, dev, test)
        self.record_files(train, train_file, dev, dev_file, test, test_file)
        self.update_event2idx_mapping(train, test)

        return train, dev, test

    def generate_instances(self, dataloader, drop_ids: Optional[set] = None):
        """
        Generate instances from DataLoader object.

        :param dataloader: Initialized DataLoader object that has parsed the log data.
        :return: list of Instances
        """
        if drop_ids is None:
            drop_ids = set()
        instances = []
        self.logger.info("Start generating instances.")
        # Prepare semantic embedding sequences for instances.
        for block in tqdm(dataloader.blocks):
            if (
                block in dataloader.block2eventseq.keys()
                and block in dataloader.block2label.keys()
            ):
                id = block
                label = dataloader.block2label[id]
                eventseq = [x for x in dataloader.block2eventseq[id] if x not in drop_ids]
                inst = Instance(id, eventseq, label)
                instances.append(inst)
            else:
                self.logger.error("Found mismatch block: %s. Please check." % block)

        self.id2label = dataloader.id2label
        self.label2id = dataloader.label2id
        self.templates = dataloader.templates
        self.embedding = dataloader.id2embed

        return instances

    def record_files(
        self, train, train_file, dev, dev_file, test, test_file, pretrain_source=None
    ):
        with open(train_file, "w", encoding="utf-8") as writer:
            for instance in train:
                writer.write(str(instance) + "\n")
        if dev:
            with open(dev_file, "w", encoding="utf-8") as writer:
                for instance in dev:
                    writer.write(str(instance) + "\n")
        with open(test_file, "w", encoding="utf-8") as writer:
            for instance in test:
                writer.write(str(instance) + "\n")
        if pretrain_source:
            with open(pretrain_source, "w", encoding="utf-8") as writer:
                for inst in train:
                    writer.write(" ".join([str(x) for x in inst.sequence]) + "\n")

    def label_distribution(self, train, dev, test):
        train_label_counter = Counter([inst.label for inst in train])
        if dev:
            dev_label_counter = Counter([inst.label for inst in dev])
            self.logger.info(
                "Dev: %d Normal, %d Anomalous instances.",
                dev_label_counter["Normal"],
                dev_label_counter["Anomalous"],
            )
        test_label_counter = Counter([inst.label for inst in test])
        self.logger.info(
            "Train: %d Normal, %d Anomalous instances.",
            train_label_counter["Normal"],
            train_label_counter["Anomalous"],
        )
        self.logger.info(
            "Test: %d Normal, %d Anomalous instances.",
            test_label_counter["Normal"],
            test_label_counter["Anomalous"],
        )

    def update_event2idx_mapping(self, pre, post):
        """
        Calculate unique events in pre & post for event count vector calculation.
        :param pre: pre data, including training set and validation set(if has)
        :param post: post data, mostly testing set
        :return: update mappings in self
        """
        self.logger.info("Update train instances' event-idx mapping.")
        pre_ordered_events = self._count_events(pre)
        embed_size = len(pre_ordered_events)
        self.logger.info("Embed size: %d in pre dataset." % embed_size)
        for idx, event in enumerate(pre_ordered_events):
            self.train_event2idx[event] = idx
        self.logger.info("Update test instances' event-idx mapping.")
        post_ordered_events = self._count_events(post)
        base = len(pre_ordered_events)
        increment = 0
        for event in post_ordered_events:
            if event not in pre_ordered_events:
                pre_ordered_events.append(event)
                self.test_event2idx[event] = base + increment
                increment += 1
            else:
                self.test_event2idx[event] = self.train_event2idx[event]
        embed_size = len(pre_ordered_events)
        self.logger.info("Embed size: %d in pre+post dataset." % embed_size)

    def _count_events(self, sequence):
        events = set()
        for inst in sequence:
            for event in inst.sequence:
                events.add(int(event))
        ordered_events = sorted(list(events))
        return ordered_events


if __name__ == "__main__":
    from sempca.representations import TemplateTfIdf
    from sempca.preprocessing import cut_by_613

    processor = Preprocessor()
    template_encoder = TemplateTfIdf()
    processor.process_and_split(
        dataset="Spirit",
        parsing="Drain",
        template_encoding=template_encoder.present,
        cut_func=cut_by_613,
    )
