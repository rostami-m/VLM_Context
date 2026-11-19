"""Deterministic prospective Gate and triangular accuracy-matrix metrics."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib,json,math
KS=(0,1,2,4,8,16)

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: expected integer >= {minimum}")


def finite(value, name, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name}: expected finite number >= {minimum}")


@dataclass
class CopyGate:
    """Consumes sequential probe counts only; no accuracy or query gold input.

    Equal-to-threshold counts are admissible. First strict crossing terminates;
    never evaluate a later K to rescue a failed earlier prefix. K=0 has no rate.
    """
    threshold: float
    probe_count: int = 40
    next_index: int = 1
    selected_k: int = 0
    stopped: bool = False
    observations: list = field(default_factory=list)

    def __post_init__(self):
        finite(self.threshold, "threshold")
        integer(self.probe_count, "probe_count", 1)
        if self.threshold > 1:
            raise ValueError("threshold > 1")

    def observe(self, k, copied, total):
        integer(k, "k", 1)
        integer(copied, "copied")
        integer(total, "total", 1)
        if self.stopped or self.next_index >= len(KS) or k != KS[self.next_index]:
            raise ValueError("out-of-order or post-stop observation")
        if total != self.probe_count or copied > total:
            raise ValueError("incorrect probe denominator/count")
        rate = copied / total
        rejected = rate > self.threshold
        self.observations.append({"k": k, "copied": copied, "total": total,
                                  "rate": rate, "rejected": rejected})
        if rejected:
            self.stopped = True
        else:
            self.selected_k = k
            self.next_index += 1
            self.stopped = self.next_index == len(KS)
        return {"selected_k": self.selected_k, "stopped": self.stopped,
                "next_k": None if self.stopped else KS[self.next_index]}


def accuracy_matrix_metrics(matrix):
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError("nonempty square matrix required")
    for t, row in enumerate(matrix):
        for j, value in enumerate(row):
            if j > t:
                if value is not None:
                    raise ValueError("unseen task cell must be null")
            else:
                finite(value, "accuracy")
                if value > 1:
                    raise ValueError("accuracy > 1")
    return {"AA": sum(matrix[-1]) / n,
            "BWT": None if n == 1 else sum(matrix[-1][j] - matrix[j][j] for j in range(n - 1)) / (n - 1)}
