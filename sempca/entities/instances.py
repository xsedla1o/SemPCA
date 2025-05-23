import hashlib
from collections import Counter
from typing import List


class Instance:
    def __init__(self, block_id: str, log_sequence: List[int], label: str):
        self.id: str = block_id
        self.sequence: List[int] = log_sequence
        self.label: str = label
        self.repr = None
        self.predicted = ""
        self.confidence = 0
        self.semantic_emb_seq = []
        self.context_emb_seq = []
        self.semantic_emb = None
        self.encode = None
        self.semantic_repr = []
        self.context_repr = []

    def __str__(self):
        sequence_str = " ".join([str(x) for x in self.sequence])
        if self.predicted == "":
            return sequence_str + "\n" + str(self.id) + "," + self.label + "\n"
        else:
            return (
                sequence_str
                + "\n"
                + str(self.id)
                + ","
                + self.label
                + ","
                + self.predicted
                + ","
                + str(self.confidence)
                + "\n"
            )

    def __hash__(self):
        return hashlib.md5(str(self).encode("utf-8")).hexdigest()

    @property
    def seq_hash(self):
        return hash(" ".join([str(x) for x in self.sequence]))

    @property
    def event_count(self):
        return Counter(self.sequence)


class SubSequenceInstance:
    def __init__(self, sequential, label):
        self.sequential = sequential
        self.quantity = None
        self.label = label
        self.predictions = None
        self.belongs_to = None
        self._hash_key = None

    def __hash__(self):
        if self._hash_key is None:
            self._hash_key = hash(" ".join([str(x) for x in self.sequential.tolist()]))
        return self._hash_key


class LogWithDatetime:
    def __init__(self, idx, label, datetime, message):
        self.id = idx
        self.label = label
        self.time = datetime
        self.message = message


class LogTimeStep:
    def __init__(self, logs):
        self.logs = logs
        self.sequence = [log.id for log in self.logs]
        self.label = "Normal"
        for log in self.logs:
            if log.label == "Anomalous":
                self.label = "Anomalous"
                break
