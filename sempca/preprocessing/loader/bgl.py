import os
import re
from abc import ABC
from collections import OrderedDict
from functools import partial
from typing import Tuple, List, Dict, Callable

import numpy as np
import pandas as pd

from sempca.preprocessing.loader import BasicDataLoader, DataPaths
from sempca.preprocessing.loader.templates import bgl_templates
from sempca.utils import tqdm, get_logger

time_lengths = [
    0,
    1,
    2,
    5,
    10,
    20,
    30,
    60,
    120,
    300,
    600,
    1200,
    1800,
    3600,
    3600 * 24,
    3600 * 24 * 7,
    3600 * 24 * 30,
]
line_lengths = [
    0,
    1,
    2,
    5,
    10,
    50,
    100,
    200,
    500,
    1000,
    5000,
    10000,
    15000,
    20000,
    50000,
    100000,
]


def _pre_process(line, remove_cols):
    tokens = line.strip().split()
    after_process = []
    for id, token in enumerate(tokens):
        if id not in remove_cols:
            after_process.append(token)
    return " ".join(after_process)


class SuperComputerLoader(BasicDataLoader, ABC):
    ds = "SuperComputer"

    def __init__(
        self,
        paths: DataPaths,
        semantic_repr_func=None,
        group_component: bool = False,
        win_secs: int = None,
        win_lines: int = 20,
        win_kind: str = "tumbling",
        win_step: int = 1,
        encoding: str = "utf-8",
    ):
        """
        Initialize SuperComputerLoader.

        Parameters
        ----------
        paths: dataset and persistence paths configuration
        group_component: whether to group logs by components
        win_secs: max window size in seconds
        win_lines: max window size in lines
        semantic_repr_func: semantic representation function
        """
        super().__init__(paths, semantic_repr_func)
        self._dataset_lines = None
        self.file_encoding = encoding

        assert isinstance(win_secs, int) or isinstance(win_lines, int), (
            "At least one of win_secs and win_lines should be an integer."
        )
        assert win_secs is None or (win_secs > 0), (
            "Window size must be a positive integer."
        )
        assert win_lines is None or (win_lines > 0), (
            "Window size must be a positive integer."
        )

        win_kinds = ["tumbling", "sliding"]
        assert win_kind in win_kinds, f"win_kind, must be one of {win_kinds}"

        if not os.path.exists(self.paths.in_file):
            self.logger.error("Input file not found, please check.")
            exit(1)
        self.remove_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.group_component = group_component
        self.win_secs = win_secs
        self.win_lines = win_lines

        self.win_kind = win_kind
        self.win_step = win_step

        self._pre_process = self.get_preprocessor()
        self._load_raw_log_seqs()

    def get_preprocessor(self) -> Callable[[str], str]:
        return partial(_pre_process, remove_cols=self.remove_cols)

    def _load_raw_log_seqs(self):
        """
        Load raw log sequences in log data.
        Returns
        -------
        None
        """
        sequence_file = self.paths.sequence_file
        label_file = self.paths.label_file
        if os.path.exists(sequence_file) and os.path.exists(label_file):
            self.logger.info(
                "Start load from previous extraction. File path %s" % sequence_file
            )
            with open(sequence_file, "r", encoding="utf-8") as reader:
                for line in tqdm(reader):
                    tokens = line.strip().rsplit(":", maxsplit=1)
                    block = tokens[0]
                    seq = tokens[1].split()
                    if block not in self.block2seqs.keys():
                        self.block2seqs[block] = []
                        self.blocks.append(block)
                    self.block2seqs[block] = [int(x) for x in seq]
            with open(label_file, "r", encoding="utf-8") as reader:
                for line in reader:
                    block_id, label = line.strip().rsplit(":", maxsplit=1)
                    self.block2label[block_id] = label

        else:
            self.logger.info("Start loading %s log sequences.", self.ds)
            with open(self.paths.in_file, "r", encoding=self.file_encoding) as reader:
                nodes = self._component_grouping(reader)

                if self.win_kind == "tumbling":
                    self._group_by_tumbling_window(nodes)
                elif self.win_kind == "sliding":
                    self._group_by_sliding_window(nodes)
                else:
                    raise ValueError(f"Unknown window type {self.win_kind}")

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

    def _component_grouping(self, reader):
        nodes = OrderedDict()
        if self.group_component:
            for idx, line in enumerate(reader):
                tokens = line.strip().split()
                node = str(tokens[3])
                if node not in nodes.keys():
                    nodes[node] = []
                nodes[node].append((idx, line.strip()))
            self._dataset_lines = nodes[node][-1][0] + 1
        else:
            nodes[""] = ((idx, line.strip()) for idx, line in enumerate(reader))
        return nodes

    @staticmethod
    def _get_window_end(ts_i, win_secs):
        return ts_i + win_secs if win_secs is not None else None

    @staticmethod
    def _get_end_condition(win_secs, win_lines):
        if win_secs is not None and win_lines is not None:

            def end_condition(seq_, ts_i_, window_end_):
                return window_end_ is not None and (
                    ts_i_ > window_end_ or len(seq_) + 1 > win_lines
                )

            return end_condition
        elif win_secs is not None:

            def end_condition(_, ts_i_, window_end_):
                return window_end_ is not None and ts_i_ > window_end_

            return end_condition
        elif win_lines is not None:

            def end_condition(seq_, _, _a):
                return seq_ is not None and len(seq_) + 1 > win_lines

            return end_condition
        else:
            raise ValueError("Either win_secs or win_lines must be set")

    @staticmethod
    def find_hist_bin(value, bins):
        for k in bins:
            if value <= k:
                return k
        return k

    def find_2d_hist_bin(self, x_val, y_val, hist):
        x = self.find_hist_bin(x_val, hist.index)
        y = self.find_hist_bin(y_val, hist.columns)
        return x, y

    def _group_by_tumbling_window(self, nodes: Dict[str, List[Tuple[int, str]]]):
        win_secs = self.win_secs
        win_lines = self.win_lines

        get_window_end = partial(self._get_window_end, win_secs=win_secs)
        end_condition = self._get_end_condition(win_secs, win_lines)

        self.logger.info(
            f"Max window: {(win_secs or 'unlimited')} seconds, "
            f"{(win_lines or 'unlimited')} lines"
        )
        pbar = tqdm(total=self._dataset_lines, unit="lines")

        hist2d = pd.DataFrame(index=time_lengths, columns=line_lengths).fillna(0)
        find_bin = partial(self.find_2d_hist_bin, hist=hist2d)

        hist2d_anomalies = hist2d.copy()

        real_lengths = []
        ts_first = 0
        ts_last = 0
        for node, seq in nodes.items():
            b_id = None
            window_end = None
            label = "Normal"
            for i, line in seq:
                line_label, ts, _ = line.split(" ", maxsplit=2)
                ts_i = int(ts)

                if b_id is not None and end_condition(
                    self.block2seqs[b_id], ts_i, window_end
                ):
                    # Close the current block
                    self.block2label[b_id] = label
                    pbar.update(len(self.block2seqs[b_id]))
                    real_lengths.append(ts_last - ts_first)

                    # Find the right bin for the histogram
                    (x, y) = find_bin(ts_last - ts_first, len(self.block2seqs[b_id]))
                    hist2d.loc[x, y] += 1
                    if label == "Anomalous":
                        hist2d_anomalies.loc[x, y] += 1

                    # Reset the block
                    b_id = None
                    window_end = None

                if b_id is None:
                    # Start a new block
                    b_id = f"{node}:{i}"
                    ts_first = ts_i
                    window_end = get_window_end(ts_i)
                    self.blocks.append(b_id)
                    self.block2seqs[b_id] = []
                    label = "Normal"

                if not line_label.startswith("-"):
                    label = "Anomalous"
                self.block2seqs[b_id].append(i)
                ts_last = ts_i

            else:
                # Close the last block in sequence
                self.block2label[b_id] = label
                pbar.update(len(self.block2seqs[b_id]))
                real_lengths.append(ts_last - ts_first)

                # Find the right bin for the histogram
                (x, y) = find_bin(ts_last - ts_first, len(self.block2seqs[b_id]))
                hist2d.loc[x, y] += 1
        pbar.close()

        self._log_debug_stats(real_lengths)
        self._log_debug_histogram(hist2d, hist2d_anomalies)

    def _log_debug_stats(self, real_lengths):
        self.logger.debug(
            f"blocks {len(self.blocks)}, seqs {len(self.block2seqs)}, "
            f"lines {sum(len(seq) for seq in self.block2seqs.values())}"
        )
        self.logger.debug(
            f"labels {len(self.block2label)}, anomalous "
            f"{sum(1 for label in self.block2label.values() if label == 'Anomalous')}"
        )

        # Make histogram of sequence lengths
        seq_lengths = [len(seq) for seq in self.block2seqs.values()]
        self.logger.debug(
            f"Sequence lengths: min {min(seq_lengths)}, max {max(seq_lengths)}, "
            f"avg {np.mean(seq_lengths):.2f}, median {np.median(seq_lengths)}"
        )

        # Make time histogram
        self.logger.debug(
            f"Time lengths: min {min(real_lengths)}, max {max(real_lengths)}, "
            f"avg {np.mean(real_lengths):.2f}, median {np.median(real_lengths)}"
        )

    def _log_debug_histogram(self, hist2d, hist2d_anomalies):
        """Make 2D histogram"""
        self.logger.debug("2D histogram:")
        self.logger.debug(hist2d)
        self.logger.debug(
            f"rows:\n{hist2d.sum(axis=1)}\n"
            f"cols:\n{hist2d.sum(axis=0)}\n"
            f"total: {hist2d.sum(axis=0).sum()}"
        )

        self.logger.debug("2D histogram (anomalies):")
        self.logger.debug(hist2d_anomalies)

        self.logger.debug("2D histogram (normal):")
        self.logger.debug(hist2d - hist2d_anomalies)

    def _group_by_sliding_window(self, nodes: Dict[str, List[Tuple[int, str]]]):
        win_secs = self.win_secs
        win_lines = self.win_lines

        step = self.win_step
        get_window_end = partial(self._get_window_end, win_secs=win_secs)
        end_condition = self._get_end_condition(win_secs, win_lines)

        self.logger.info(
            f"Max window: {(win_secs or 'unlimited')} seconds, "
            f"{(win_lines or 'unlimited')} lines"
        )
        pbar = tqdm(total=self._dataset_lines, unit="lines")

        hist2d = pd.DataFrame(index=time_lengths, columns=line_lengths).fillna(0)
        find_bins = partial(self.find_2d_hist_bin, hist=hist2d)
        hist2d_anomalies = hist2d.copy()

        real_lengths = []

        for node, seq in nodes.items():
            b_id = None
            bi = 0

            sliding_buffer = []
            sliding_ts_buffer = []
            sliding_anomalous = []

            ts_first = 0
            ts_last = 0

            window_end = None

            label = "Normal"
            for i, line in seq:
                line_label, ts, _ = line.split(" ", maxsplit=2)
                ts_i = int(ts)

                while b_id is not None and end_condition(
                    sliding_buffer, ts_i, window_end
                ):
                    # Close the current block
                    self.block2label[b_id] = label
                    self.block2seqs[b_id] = sliding_buffer.copy()
                    real_lengths.append(ts_last - ts_first)

                    # Find the right bin for the histogram
                    (x, y) = find_bins(ts_last - ts_first, len(self.block2seqs[b_id]))

                    hist2d.loc[x, y] += 1
                    if label == "Anomalous":
                        hist2d_anomalies.loc[x, y] += 1

                    if len(sliding_buffer) > step:
                        # Restart the block moving by step
                        sliding_buffer = sliding_buffer[step:]
                        sliding_ts_buffer = sliding_ts_buffer[step:]
                        sliding_anomalous = sliding_anomalous[step:]

                        b_id = f"{node}:{bi}"
                        bi += 1
                        ts_first = sliding_ts_buffer[0]
                        window_end = get_window_end(ts_first)
                        self.blocks.append(b_id)
                        label = "Normal" if not any(sliding_anomalous) else "Anomalous"
                    else:
                        # Reset the block
                        b_id = None
                        window_end = None
                        sliding_buffer.clear()
                        sliding_ts_buffer.clear()
                        sliding_anomalous.clear()

                if b_id is None:
                    # Start a new block
                    b_id = f"{node}:{bi}"
                    bi += 1
                    ts_first = ts_i
                    window_end = get_window_end(ts_i)
                    self.blocks.append(b_id)
                    label = "Normal"

                if not line_label.startswith("-"):
                    label = "Anomalous"
                    sliding_anomalous.append(True)
                else:
                    sliding_anomalous.append(False)
                sliding_buffer.append(i)
                sliding_ts_buffer.append(ts_i)
                ts_last = ts_i

                # Include growing buffer in output
                self.block2label[b_id] = label
                self.block2seqs[b_id] = sliding_buffer.copy()
                real_lengths.append(ts_last - ts_first)
                b_id = f"{node}:{bi}"
                bi += 1

                pbar.update(1)

            else:
                # Close the last block in sequence
                self.block2label[b_id] = label
                self.block2seqs[b_id] = sliding_buffer.copy()
                real_lengths.append(ts_last - ts_first)

                # Find the right bin for the histogram
                (x, y) = find_bins(ts_last - ts_first, len(self.block2seqs[b_id]))
                hist2d.loc[x, y] += 1
        pbar.close()

        self._log_debug_stats(real_lengths)
        self._log_debug_histogram(hist2d, hist2d_anomalies)


class BGLLoader(SuperComputerLoader):
    logger = get_logger("BGLLoader")
    ds = "BGL"

    def __init__(
        self,
        paths: DataPaths,
        semantic_repr_func=None,
        group_component: bool = False,
        win_secs: int = None,
        win_lines: int = 20,
        win_kind: str = "tumbling",
        win_step: int = 1,
    ):
        """
        Initialize BGLLoader.

        paths: dataset and persistence paths configuration

        See Also: SuperComputerLoader.__init__ for more parameters.
        """
        super(BGLLoader, self).__init__(
            paths,
            semantic_repr_func,
            group_component,
            win_secs,
            win_lines,
            win_kind,
            win_step,
            encoding="utf-8",
        )

    def parse_by_official(self):
        self.logger.info("Start parsing by Official.")
        self._restore()
        # Define official templates
        templates = bgl_templates

        os.makedirs(self.paths.official_dir, exist_ok=True)
        templates_file = self.paths.templates_file
        log2temp_file = self.paths.log2temp_file
        if os.path.exists(templates_file) and os.path.exists(log2temp_file):
            self.logger.info(
                "Found parsing result, please note that this does not guarantee a smooth execution."
            )
            with open(templates_file, "r", encoding="utf-8") as reader:
                for line in tqdm(reader):
                    tokens = line.strip().split(",")
                    id = int(tokens[0])
                    template = ",".join(tokens[1:])
                    self.templates[id] = template

            with open(log2temp_file, "r", encoding="utf-8") as reader:
                for line in reader:
                    logid, tempid = line.strip().split(",")
                    self.log2temp[int(logid)] = int(tempid)

        else:
            for id, template in enumerate(templates):
                self.templates[id] = template
            with open(self.paths.in_file, "r", encoding="utf-8") as reader:
                log_id = 0
                for line in tqdm(reader):
                    line = line.strip()
                    if self.remove_cols:
                        processed_line = self._pre_process(line)
                    for index, template in self.templates.items():
                        if re.compile(template).match(processed_line) is not None:
                            self.log2temp[log_id] = index
                            break
                    if log_id not in self.log2temp.keys():
                        # if processed_line == '':
                        #     self.log2temp[log_id] = -1
                        self.logger.warning(
                            "Mismatched log message: %s" % processed_line
                        )
                        for index, template in self.templates.items():
                            if re.compile(template).match(line) is not None:
                                self.log2temp[log_id] = index
                                break
                        if log_id not in self.log2temp.keys():
                            self.logger.error("Failed to parse line %s" % line)
                            exit(2)
                    log_id += 1

            with open(templates_file, "w", encoding="utf-8") as writer:
                for id, template in self.templates.items():
                    writer.write(",".join([str(id), template]) + "\n")
            with open(log2temp_file, "w", encoding="utf-8") as writer:
                for logid, tempid in self.log2temp.items():
                    writer.write(",".join([str(logid), str(tempid)]) + "\n")
            # with open(logseq_file, 'w', encoding='utf-8') as writer:
            #     self._save_log_event_seqs(writer)
        self._prepare_semantic_embed(self.paths.semantic_vector_file)
        # Summarize log event sequences.
        for block, seq in self.block2seqs.items():
            self.block2eventseq[block] = []
            for log_id in seq:
                self.block2eventseq[block].append(self.log2temp[log_id])


class TBirdLoader(SuperComputerLoader):
    logger = get_logger("TBirdLoader")
    ds = "TBird"

    def __init__(
        self,
        paths: DataPaths,
        semantic_repr_func=None,
        group_component: bool = False,
        win_secs: int = None,
        win_lines: int = 20,
        win_kind: str = "tumbling",
        win_step: int = 1,
    ):
        """
        Initialize TBirdLoader.

        paths: dataset and persistence paths configuration

        See Also: SuperComputerLoader.__init__ for more parameters.
        """
        super(TBirdLoader, self).__init__(
            paths,
            semantic_repr_func,
            group_component,
            win_secs,
            win_lines,
            win_kind,
            win_step,
            encoding="latin-1",
        )

    def parse_by_official(self):
        raise NotImplementedError()
