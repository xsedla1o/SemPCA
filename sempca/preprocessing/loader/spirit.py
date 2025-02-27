import datetime
import os
import re
from collections import OrderedDict

from sempca.const import PROJECT_ROOT
from sempca.preprocessing.loader import BasicDataLoader, DataPaths
from sempca.utils import tqdm


class SpiritLoader(BasicDataLoader):
    def __init__(
        self,
        paths: DataPaths,
        window_size=120,
        semantic_repr_func=None,
    ):
        super(SpiritLoader, self).__init__(paths, semantic_repr_func)
        self.remove_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.regs = {
            "reg": [
                "<.*>",
                "((25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))\.){3}(25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))",
                "\/([\w\.]+\/?)*",
            ],
            "replace": ["[ID]", "[IP]", "[PATH]"],
        }
        self.node_idx = 7
        self.window_size = window_size
        self.file_for_parsing = os.path.join(
            self.paths.dataset_dir, "Cleaned_Spirit.log"
        )
        if not os.path.exists(self.paths.in_file):
            self.logger.error("File %s not found, please check." % self.paths.in_file)
            raise FileNotFoundError
        self._load_raw_log_seqs()

    def _pre_process(self, line):
        tokens = line.strip().split()
        after_process = []
        for id, token in enumerate(tokens):
            if id not in self.remove_cols:
                after_process.append(token)
        processed_line = " ".join(after_process)
        for regex, replace in zip(self.regs["reg"], self.regs["replace"]):
            processed_line = re.sub(regex, replace, processed_line)
        return processed_line

    def _load_raw_log_seqs(self):
        sequence_file = self.paths.sequence_file
        label_file = self.paths.label_file
        if os.path.exists(sequence_file) and os.path.exists(label_file):
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
            with open(label_file, "r", encoding="utf-8") as reader:
                for line in reader.readlines():
                    block_id, label = line.strip().split(":")
                    self.block2label[block_id] = label

        else:
            self._load_by_fixed_window(sequence_file, label_file)
            # self._load_by_time_window(sequence_file, label_file)

    def parse_by_official(self):
        self.logger.error("Parsing by official templates not implemented.")
        raise NotImplementedError

    def parse_by_drain(self, core_jobs=5, encode="iso8859-1"):
        return super(SpiritLoader, self).parse_by_drain(core_jobs, encode)

    def _load_by_node_time_window(self, sequence_file, label_file):
        self.logger.info("Start loading Spirit log sequences.")
        nodes = OrderedDict()
        # Prepare a clean log file for parsing
        writer = open(self.file_for_parsing, "w", encoding="utf-8")

        with open(self.paths.in_file, "r", encoding="iso-8859-1") as reader:
            line_num = 0
            for line in reader.readlines():
                line = line.strip()
                if line == "":
                    line_num += 1
                    writer.write(line + "\n")
                    continue
                tokens = line.split()
                prefix = tokens[0]
                node = tokens[3]
                datetime_str = tokens[2] + " " + tokens[6]
                line = self._pre_process(line)
                writer.write(line + "\n")
                dt = datetime.datetime.strptime(datetime_str, "%Y.%m.%d %H:%M:%S")
                if node not in nodes.keys():
                    nodes[node] = []
                nodes[node].append([line_num, prefix, dt])
                line_num += 1
        writer.close()

        # construct 1 hour' seq
        start_time = datetime.datetime.strptime(
            "2005.01.01 00:00:00", "%Y.%m.%d %H:%M:%S"
        )
        end_time = start_time + datetime.timedelta(minutes=1)
        block_id = 0
        key = str(block_id)
        self.blocks.append(key)
        self.block2seqs[key] = []
        self.block2label[key] = "Normal"
        while True:
            try:
                node, seq = nodes.popitem(last=False)
                for msg in seq:
                    line_num, prefix, dt = msg
                    if dt <= end_time:
                        self.block2seqs[key].append(line_num)
                        if prefix != "-":
                            self.block2label[key] = "Anomalous"
                    else:
                        block_id += 1
                        key = str(block_id)
                        self.blocks.append(key)
                        self.block2seqs[key] = [line_num]
                        self.block2label[key] = (
                            "Normal" if prefix == "-" else "Anomalous"
                        )
                        start_time = end_time
                        end_time = start_time + datetime.timedelta(hours=1)
            except KeyError:
                break

        with open(sequence_file, "w", encoding="utf-8") as writer:
            for block in self.blocks:
                writer.write(
                    ":".join(
                        [block, " ".join([str(x) for x in self.block2seqs[block]])]
                    )
                    + "\n"
                )
        with open(label_file, "w", encoding="utf-8") as writer:
            for block in self.block2label.keys():
                writer.write(":".join([block, self.block2label[block]]) + "\n")

        self.logger.info("Extraction finished successfully.")

    def _load_by_time_window(self, sequence_file, label_file):
        self.logger.info("Start loading Spirit log sequences.")
        summarized_lines = []
        # Prepare a clean log file for parsing
        writer = open(self.file_for_parsing, "w", encoding="utf-8")

        with open(self.paths.in_file, "r", encoding="iso-8859-1") as reader:
            line_num = 0
            for line in reader.readlines():
                line = line.strip()
                if line == "":
                    line_num += 1
                    writer.write(line + "\n")
                    continue
                tokens = line.split()
                prefix = tokens[0]
                datetime_str = tokens[2] + " " + tokens[6]
                line = self._pre_process(line)
                writer.write(line + "\n")
                dt = datetime.datetime.strptime(datetime_str, "%Y.%m.%d %H:%M:%S")
                summarized_lines.append([line_num, prefix, dt])
                line_num += 1
        writer.close()

        # construct 1 hour' seq
        start_time = datetime.datetime.strptime(
            "2005.01.01 00:00:00", "%Y.%m.%d %H:%M:%S"
        )
        end_time = start_time + datetime.timedelta(minutes=1)
        block_id = 0
        key = str(block_id)
        self.blocks.append(key)
        self.block2seqs[key] = []
        self.block2label[key] = "Normal"
        for msg in summarized_lines:
            line_num, prefix, dt = msg
            if dt <= end_time:
                self.block2seqs[key].append(line_num)
                if prefix != "-":
                    self.block2label[key] = "Anomalous"
            else:
                block_id += 1
                key = str(block_id)
                self.blocks.append(key)
                self.block2seqs[key] = [line_num]
                self.block2label[key] = "Normal" if prefix == "-" else "Anomalous"
                start_time = end_time
                end_time = start_time + datetime.timedelta(hours=1)

        with open(sequence_file, "w", encoding="utf-8") as writer:
            for block in self.blocks:
                writer.write(
                    ":".join(
                        [block, " ".join([str(x) for x in self.block2seqs[block]])]
                    )
                    + "\n"
                )
        with open(label_file, "w", encoding="utf-8") as writer:
            for block in self.block2label.keys():
                writer.write(":".join([block, self.block2label[block]]) + "\n")

        self.logger.info("Extraction finished successfully.")

    def _load_by_fixed_window(self, sequence_file, label_file):
        self.logger.info("Start loading Spirit log sequences.")
        summarized_lines = []
        # Prepare a clean log file for parsing
        writer = open(self.file_for_parsing, "w", encoding="utf-8")

        with open(self.paths.in_file, "r", encoding="iso-8859-1") as reader:
            line_num = 0
            for line in reader.readlines():
                line = line.strip()
                if line == "":
                    line_num += 1
                    writer.write(line + "\n")
                    continue
                tokens = line.split()
                prefix = tokens[0]
                datetime_str = tokens[2] + " " + tokens[6]
                line = self._pre_process(line)
                writer.write(line + "\n")
                dt = datetime.datetime.strptime(datetime_str, "%Y.%m.%d %H:%M:%S")
                summarized_lines.append([line_num, prefix, dt])
                # if dt > datetime.datetime.strptime('2005.03.01 00:00:00', '%Y.%m.%d %H:%M:%S'):
                #     break
                line_num += 1
        writer.close()

        # construct 1 minutes' seq
        block_id = 0
        key = "0"
        self.blocks.append(key)
        self.block2seqs[key] = []
        self.block2label[key] = "Normal"
        slen = 0
        for line_num, prefix, dt in summarized_lines:
            key = str(block_id)
            if slen < self.window_size:
                self.block2seqs[key].append(line_num)
                if prefix != "-":
                    self.block2label[key] = "Anomalous"
                slen += 1
                continue
            else:
                slen = 1
                block_id += 1
                key = str(block_id)
                self.blocks.append(key)
                self.block2seqs[key] = [line_num]
                self.block2label[key] = "Normal"
        with open(sequence_file, "w", encoding="utf-8") as writer:
            for block in self.blocks:
                writer.write(
                    ":".join(
                        [block, " ".join([str(x) for x in self.block2seqs[block]])]
                    )
                    + "\n"
                )
        with open(label_file, "w", encoding="utf-8") as writer:
            for block in self.block2label.keys():
                writer.write(":".join([block, self.block2label[block]]) + "\n")

        self.logger.info("Extraction finished successfully.")


if __name__ == "__main__":
    log2event = {}
    writer = open(
        os.path.join(PROJECT_ROOT, "datasets/Spirit/Spirit.log"),
        "w",
        encoding="iso8859-1",
    )
    with open(
        os.path.join(PROJECT_ROOT, "datasets/Spirit/Spirit_Full.log"),
        "r",
        encoding="iso8859-1",
    ) as reader:
        total_lines = 0
        num_alerts = 0
        line = reader.readline()
        while True:
            if total_lines > 1200000:
                break
            total_lines += 1
            if not line.startswith("-"):
                num_alerts += 1
            line = reader.readline()
            if total_lines % 1000000 == 0:
                print("Processed %d lines" % total_lines)
    writer.close()
    print(total_lines, num_alerts)
