import os
import re

from sempca.preprocessing.loader import BasicDataLoader
from sempca.preprocessing.loader.basic import DataPaths
from sempca.preprocessing.loader.templates import hdfs_templates
from sempca.utils import tqdm


class HDFSLoader(BasicDataLoader):
    def __init__(self, paths: DataPaths, semantic_repr_func=None):
        super(HDFSLoader, self).__init__(paths, semantic_repr_func)
        self.blk_rex = re.compile(r"blk_[-]{0,1}[0-9]+")
        if not os.path.exists(self.paths.in_file):
            self.logger.error("Input file not found, please check.")
            exit(1)
        self.remove_cols = [0, 1, 2, 3, 4]
        self._load_raw_log_seqs()
        self._load_hdfs_labels()

    def parse_by_Official(self):
        self._restore()
        templates = hdfs_templates

        os.makedirs(self.paths.official_dir, exist_ok=True)
        templates_file = self.paths.templates_file
        log2temp_file = self.paths.log2temp_file
        logseq_file = self.paths.logseq_file
        if all(os.path.exists(f) for f in [templates_file, log2temp_file, logseq_file]):
            self.logger.info(
                "Found parsing result, please note that this does not guarantee a smooth execution."
            )
            with open(templates_file, "r", encoding="utf-8") as reader:
                self._load_templates(reader)

            with open(log2temp_file, "r", encoding="utf-8") as reader:
                self._load_log2temp(reader)

            with open(logseq_file, "r", encoding="utf-8") as reader:
                self._load_log_event_seqs(reader)
        else:
            self.logger.info("Parsing result not found, start a new one.")
            for id, template in enumerate(templates):
                self.templates[id] = template
            with open(self.paths.in_file, "r", encoding="utf-8") as reader:
                log_id = 0
                for line in tqdm(reader.readlines()):
                    line = line.strip()
                    if self.remove_cols:
                        processed_line = self._pre_process(line)
                    for index, template in self.templates.items():
                        if re.compile(template).match(processed_line) is not None:
                            self.log2temp[log_id] = index
                            break
                    if log_id not in self.log2temp.keys():
                        self.logger.warning(
                            "Mismatched log message : %s, try using original line."
                            % processed_line
                        )
                        for index, template in self.templates.items():
                            if re.compile(template).match(line) is not None:
                                self.log2temp[log_id] = index
                                break
                        if log_id not in self.log2temp.keys():
                            self.logger.error("Failed to parse line %s" % line)
                            exit(2)
                    log_id += 1

            for block, seq in self.block2seqs.items():
                self.block2eventseq[block] = []
                for log_id in seq:
                    self.block2eventseq[block].append(self.log2temp[log_id])

            with open(templates_file, "w", encoding="utf-8") as writer:
                for id, template in self.templates.items():
                    writer.write(",".join([str(id), template]) + "\n")
            with open(log2temp_file, "w", encoding="utf-8") as writer:
                for logid, tempid in self.log2temp.items():
                    writer.write(",".join([str(logid), str(tempid)]) + "\n")
            with open(logseq_file, "w", encoding="utf-8") as writer:
                self._save_log_event_seqs(writer)
        self._prepare_semantic_embed(self.paths.semantic_vector_file)

    def _pre_process(self, line):
        tokens = line.strip().split()
        after_process = []
        for idx, token in enumerate(tokens):
            if idx not in self.remove_cols:
                after_process.append(token)
        return " ".join(after_process)

    def _load_raw_log_seqs(self):
        """
        Load log sequences from raw HDFS log file.
        :return: Update related attributes in current instance.
        """
        sequence_file = self.paths.sequence_file
        if not os.path.exists(sequence_file):
            self.logger.info("Start extract log sequences from HDFS raw log file.")
            with open(self.paths.in_file, "r", encoding="utf-8") as reader:
                log_id = 0
                for line in tqdm(reader.readlines()):
                    processed_line = self._pre_process(line)
                    block_ids = set(re.findall(self.blk_rex, processed_line))
                    if len(block_ids) == 0:
                        self.logger.warning(
                            "Failed to parse line: %s . Try with raw log message."
                            % line
                        )
                        block_ids = set(re.findall(self.blk_rex, line))
                        if len(block_ids) == 0:
                            self.logger.error("Failed, please check the raw log file.")
                        else:
                            self.logger.info(
                                "Succeed. %d block ids are found." % len(block_ids)
                            )

                    for block_id in block_ids:
                        if block_id not in self.block2seqs.keys():
                            self.blocks.append(block_id)
                            self.block2seqs[block_id] = []
                        self.block2seqs[block_id].append(log_id)

                    log_id += 1
            with open(sequence_file, "w", encoding="utf-8") as writer:
                for block in self.blocks:
                    writer.write(
                        ":".join(
                            [block, " ".join([str(x) for x in self.block2seqs[block]])]
                        )
                        + "\n"
                    )
        else:
            self.logger.info(
                "Start load from previous extraction. File path %s" % sequence_file
            )
            with open(sequence_file, "r", encoding="utf-8") as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split(":")
                    block = tokens[0]
                    seq = tokens[1].split()
                    if block not in self.block2seqs.keys():
                        self.block2seqs[block] = []
                        self.blocks.append(block)
                    self.block2seqs[block] = [int(x) for x in seq]

        self.logger.info("Extraction finished successfully.")

    def _load_hdfs_labels(self):
        with open(self.paths.label_file, "r", encoding="utf-8") as reader:
            for line in reader.readlines():
                token = line.strip().split(",")
                block = token[0]
                label = self.id2label[int(token[1])]
                self.block2label[block] = label


if __name__ == "__main__":
    from sempca.representations import TemplateTfIdf

    semantic_encoder = TemplateTfIdf()
    paths = DataPaths(
        dataset_name="HDFS",
        in_file="datasets/temp_HDFS/HDFS.log",
        dataset_dir="datasets/temp_HDFS",
        persistence_dir="datasets/temp_HDFS/persistences",
        drain_config="conf/HDFS.ini",
    )
    loader = HDFSLoader(
        paths=paths,
        semantic_repr_func=semantic_encoder.present,
    )
    loader.parse_by_drain()
