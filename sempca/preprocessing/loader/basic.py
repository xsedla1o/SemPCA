import abc
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Callable

import numpy as np
from tqdm import tqdm as tqdm_original

from sempca.const import PROJECT_ROOT
from sempca.parser import Drain3Parser
from sempca.utils import tqdm, get_logger


@dataclass
class DataPaths:
    """
    DataPaths is a dataclass that contains all the paths to the files and directories
    """

    dataset_name: str  # e.g., "HDFS", "BGL"
    parser_name: str = "Drain"  # e.g., "Drain", "Official"
    in_file: Union[Path, str] = None
    project_root: Union[Path, str] = None
    label_file: Union[Path, str] = None
    drain_config: Union[Path, str] = None
    persistence_dir: Union[Path, str] = None
    datasets_dir: Union[Path, str] = None
    dataset_dir: Union[Path, str] = None
    processed_out_dir: Union[Path, str] = None

    word2vec_file: Path = field(init=False)
    # Dataset-specific paths (computed from dataset_name)
    official_dir: Path = field(init=False)
    templates_file: Path = field(init=False)
    log2temp_file: Path = field(init=False)
    logseq_file: Path = field(init=False)
    sequence_file: Path = field(init=False)
    semantic_vector_file: Path = field(init=False)

    @staticmethod
    def to_path(path: Union[str, Path, None]) -> Union[Path, None]:
        if path is not None:
            return Path(path)
        return None

    def __post_init__(self):
        self.project_root = self.to_path(self.project_root)
        if self.project_root is None:
            self.project_root = Path(PROJECT_ROOT)

        self.datasets_dir = self.to_path(self.datasets_dir)
        if self.datasets_dir is None:
            self.datasets_dir = self.project_root / "datasets"

        # Set default word2vec file
        self.word2vec_file = self.datasets_dir / "glove.6B.300d.txt"

        # Compute dataset-specific paths using the dataset_name
        self.dataset_dir = self.to_path(self.dataset_dir)
        if self.dataset_dir is None:
            self.dataset_dir = self.datasets_dir / self.dataset_name

        self.in_file = self.to_path(self.in_file)
        if self.in_file is None:
            self.in_file = self.dataset_dir / f"{self.dataset_name}.log"

        self.label_file = self.to_path(self.label_file)
        if self.label_file is None:
            self.label_file = self.dataset_dir / "label.txt"

        self.sequence_file = self.dataset_dir / "raw_log_seqs.txt"

        self.processed_out_dir = self.to_path(self.processed_out_dir)
        if self.processed_out_dir is None:
            self.processed_out_dir = self.dataset_dir / "inputs" / self.parser_name

        self.persistence_dir = self.to_path(self.persistence_dir)
        if self.persistence_dir is None:
            self.persistence_dir = self.dataset_dir / "persistences"

        self.official_dir = self.persistence_dir / "official"
        self.templates_file = self.official_dir / f"{self.dataset_name}_templates.txt"
        self.log2temp_file = self.official_dir / "log2temp.txt"
        self.logseq_file = self.official_dir / "event_seqs.txt"
        self.semantic_vector_file = self.official_dir / "event2semantic.vec"

        # Set Drain parsing paths with defaults if not provided.
        self.drain_config = self.to_path(self.drain_config)
        if self.drain_config is None:
            self.drain_config = self.project_root / "conf" / f"{self.dataset_name}.ini"


if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
    tqdm_write = tqdm_original.write
else:
    tqdm_write = print


def worker_parse(parser_instance, pre_process_func, log_id_line_tuple):
    log_id, line_content = log_id_line_tuple
    try:
        processed_line = pre_process_func(line_content)
        cluster = parser_instance.match(processed_line)
        if cluster:
            return log_id, cluster.cluster_id
        else:
            tqdm_write(
                f"Warning: No match for log ID {log_id}, line: {line_content[:150]}..."
            )
    except Exception as e:
        tqdm_write(f"ERROR parsing line {log_id}, {line_content}: {e}")
    return None


def read_nonempty_lines(filepath, encoding):
    with open(filepath, "r", encoding=encoding, errors="ignore") as reader:
        tqdm_reader = tqdm(reader, desc="Reading and Preprocessing")
        for log_id, line in enumerate(tqdm_reader):
            line = line.strip()
            if line == "":
                continue
            yield log_id, line


class BasicDataLoader:
    def __init__(self, paths: DataPaths, semantic_repr_func=None):
        self.logger = get_logger(self.__class__.__name__)
        self.paths = paths

        self.blocks = []  # list of block / sequence IDs
        self.templates = {}  # dict of template ID to template string
        self.log2temp = {}  # dict of log ID (line no.) to template ID
        self.remove_cols = []  # list of column indices to remove

        self.id2label = {0: "Normal", 1: "Anomalous"}
        self.label2id = {"Normal": 0, "Anomalous": 1}

        self.block2seqs = {}  # dict of block ID to list of log IDs (line nums.)
        self.block2label = {}  # dict of block ID to label "Normal" or "Anomalous"
        self.block2eventseq = {}  # dict of block ID to list of event IDs
        self.id2embed = {}  # dict of template ID to semantic embedding

        self.semantic_repr_func = semantic_repr_func  # callable to get embeddings

        # Only used by Spirit subclass, ignore for now.
        self.file_for_parsing = None  # path to a pre-processed file for parsing

    @abc.abstractmethod
    def _load_raw_log_seqs(self):
        return

    @abc.abstractmethod
    def get_preprocessor(self) -> Callable[[str], str]:
        """
        :return: A callable function to preprocess log lines.
        """

    def parse(self, parsing_method: str):
        """
        :parsing_method: Specify the parsing method, either "Drain" or "Official".
        :return: Update templates, log2temp attributes in self.
        """
        if parsing_method == "Drain":
            self.parse_by_drain()
        elif parsing_method == "Official":
            self.parse_by_official()
        else:
            self.logger.error("Parsing method %s not implemented yet.")
            raise NotImplementedError

    @abc.abstractmethod
    def parse_by_official(self):
        """
        Load parsing results by official templates.
        :return: Update templates, log2temp attributes in self.
        """

    def parse_by_drain(self, core_jobs=5, encode="utf-8"):
        """
        Load parsing results by Drain
        :return: Update templates, log2temp attributes in self.
        """
        self.logger.info("Start parsing by Drain.")
        self._restore()
        if not os.path.exists(self.paths.drain_config):
            self.logger.error(
                "Drain config file %s not found.", self.paths.drain_config
            )
            exit(1)
        parser = Drain3Parser(self.paths.drain_config, self.paths.persistence_dir)
        persistence_folder = parser.persistence_folder

        # Specify persistence files.
        log_template_mapping_file = os.path.join(
            persistence_folder, "log_event_mapping.dict"
        )
        templates_embedding_file = os.path.join(
            parser.persistence_folder, "templates.vec"
        )
        start_time = time.time()
        if self.file_for_parsing:
            # If prepared an cleaned file for parsing, no need to remove columns.
            in_file = self.file_for_parsing
            remove_columns = []
        else:
            in_file = self.paths.in_file
            remove_columns = self.remove_cols
        if parser.to_update:
            self.logger.info("No trained parser found, start training.")
            parser.parse_file(in_file, remove_cols=remove_columns, encode=encode)
            self.logger.info(
                "Get total %d templates." % len(parser.parser.drain.clusters)
            )
        # Load templates from trained parser.
        for cluster_inst in parser.parser.drain.clusters:
            self.templates[int(cluster_inst.cluster_id)] = cluster_inst.get_template()

        # check parsing resutls such as log2event dict and template embeddings.
        if self._check_parsing_persistences(log_template_mapping_file):
            self.load_parsing_results(log_template_mapping_file)

        else:
            self.logger.info(
                "Missing persistence file(s), start with a full parsing process."
            )
            parse_time = time.time()
            self.logger.warning(
                "If you don't want this to happen, please copy persistence files from somewhere else and put it in %s"
                % persistence_folder
            )
            if core_jobs and core_jobs > 1:
                pool = Pool(core_jobs)

                pre_process_func = self.get_preprocessor()
                worker_parse_init = partial(worker_parse, parser, pre_process_func)
                lines = read_nonempty_lines(self.paths.in_file, encode)

                chunk_size = 8192
                results_iterator = pool.imap_unordered(
                    worker_parse_init, lines, chunksize=chunk_size
                )
                self.log2temp = {}
                for result in results_iterator:
                    if result is not None:
                        log_id, template_id = result
                        self.log2temp[log_id] = template_id
                pool.close()
                pool.join()
            else:
                self.log2temp = {}
                preprocessor = self.get_preprocessor()
                for log_id, line in read_nonempty_lines(self.paths.in_file, encode):
                    line = line.strip()
                    processed_line = preprocessor(line)
                    cluster = parser.match(processed_line)
                    if not cluster:
                        tqdm_write(
                            "Warning: no match, try increasing max clusters:", line
                        )
                        continue
                    self.log2temp[log_id] = cluster.cluster_id

            # Record block id and log event sequences.
            self._record_parsing_results(log_template_mapping_file)

            self.logger.info("Finished parsing in %.2f" % (time.time() - parse_time))

        # Transform original log sequences with log ids(line number) to log event sequence.
        for block, seq in self.block2seqs.items():
            self.block2eventseq[block] = []
            for log_id in seq:
                self.block2eventseq[block].append(self.log2temp[log_id])

        # Prepare semantic embeddings.
        self._prepare_semantic_embed(templates_embedding_file)
        self.logger.info(
            "All data preparation finished in %.2f" % (time.time() - start_time)
        )

    def load_parsing_results(self, log_template_mapping_file):
        self.logger.info("Start loading previous parsing results.")
        start = time.time()
        log_template_mapping_reader = open(
            log_template_mapping_file, "r", encoding="utf-8"
        )
        # event_seq_reader = open(event_seq_file, 'r', encoding='utf-8')
        self._load_log2temp(log_template_mapping_reader)
        # self._load_log_event_seqs(event_seq_reader)
        log_template_mapping_reader.close()
        # event_seq_reader.close()
        self.logger.info("Finished in %.2f" % (time.time() - start))

    def _restore(self):
        self.block2emb = {}
        self.templates = {}
        self.log2temp = {}

    def _save_log_event_seqs(self, writer):
        self.logger.info("Start saving log event sequences.")
        for block, event_seq in self.block2eventseq.items():
            event_seq = map(lambda x: str(x), event_seq)
            seq_str = " ".join(event_seq)
            writer.write(str(block) + ":" + seq_str + "\n")
        self.logger.info("Log event sequences saved.")

    def _load_log_event_seqs(self, reader):
        for line in reader:
            tokens = line.strip().split(":")
            block = tokens[0]
            seq = tokens[1].split()
            self.block2eventseq[block] = [int(x) for x in seq]
        self.logger.info("Loaded %d blocks" % len(self.block2eventseq))

    def _prepare_semantic_embed(self, semantic_emb_file):
        if self.semantic_repr_func:
            self.id2embed = self.semantic_repr_func(self.templates)
            with open(semantic_emb_file, "w", encoding="utf-8") as writer:
                for id, embed in self.id2embed.items():
                    writer.write(str(id) + " ")
                    writer.write(" ".join([str(x) for x in embed.tolist()]) + "\n")
            self.logger.info(
                "Finish calculating semantic representations, please found the vector file at %s"
                % semantic_emb_file
            )
        else:
            self.logger.warning(
                "No template encoder. Please be NOTED that this may lead to duplicate full parsing process."
            )

    def _check_parsing_persistences(self, log_template_mapping_file):
        flag = self._check_file_existence_and_contents(log_template_mapping_file)
        return flag

    def _check_file_existence_and_contents(self, file):
        flag = os.path.exists(file) and os.path.getsize(file) != 0
        self.logger.info("checking file %s ... %s" % (file, str(flag)))
        return flag

    def _record_parsing_results(self, log_template_mapping_file):
        # Recording parsing result.log_event_mapping
        start_time = time.time()
        log_template_mapping_writer = open(
            log_template_mapping_file, "w", encoding="utf-8"
        )
        # event_seq_writer = open(evet_seq_file, 'w', encoding='utf-8')
        self._save_log2temp(log_template_mapping_writer)
        # self._save_log_event_seqs(event_seq_writer)
        log_template_mapping_writer.close()
        # event_seq_writer.close()
        self.logger.info("Done in %.2f" % (time.time() - start_time))

    def _load_templates(self, reader):
        for line in reader:
            tokens = line.strip().split(",")
            id = tokens[0]
            template = ",".join(tokens[1:])
            self.templates[int(id)] = template
        self.logger.info("Loaded %d templates" % len(self.templates))

    def _save_templates(self, writer):
        for id, template in self.templates.items():
            writer.write(",".join([str(id), template]) + "\n")
        self.logger.info("Templates saved.")

    def _load_log2temp(self, reader):
        for line in reader:
            logid, tempid = line.strip().split(",")
            self.log2temp[int(logid)] = int(tempid)
        self.logger.info(
            "Loaded %d log sequences and their mappings." % len(self.log2temp)
        )

    def _save_log2temp(self, writer):
        for log_id, temp_id in self.log2temp.items():
            writer.write(str(log_id) + "," + str(temp_id) + "\n")
        self.logger.info("Log2Temp saved.")

    def _load_semantic_embed(self, reader):
        for line in reader:
            token = line.split()
            template_id = int(token[0])
            embed = np.asarray(token[1:], dtype=float)
            self.id2embed[template_id] = embed
        self.logger.info(
            "Load %d templates with embedding size %d"
            % (len(self.id2embed), self.id2embed[1].shape[0])
        )

    def _split(self, X, copies=5):
        quota = int(len(X) / copies) + 1
        res = []
        for i in range(copies):
            res.append(X[i * quota : (i + 1) * quota])
        return res
