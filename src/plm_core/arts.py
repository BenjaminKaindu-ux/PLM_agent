"""ARTS-style adaptive sequencing (Kellman: accuracy + response time, not accuracy alone).

Deterministic, external to any LLM. The PLM agent (or the drill UI) reads/writes this
state on every trial; sequencing must be fast and consistent, never subject to
reasoning variance.
"""

from dataclasses import dataclass, field
import random
import statistics


@dataclass
class CategoryState:
    name: str
    rt_threshold_s: float = 8.0
    retire_streak: int = 4
    streak: int = 0
    attempts: int = 0
    errors: int = 0
    rts: list = field(default_factory=list)
    retired: bool = False
    last_seen: int = -10  # trial index when last presented

    @property
    def accuracy(self) -> float:
        return 1.0 - (self.errors / self.attempts) if self.attempts else 0.0

    @property
    def median_rt(self) -> float:
        return statistics.median(self.rts) if self.rts else 0.0


class ArtsTracker:
    """Priority rises with error rate and slow-but-correct responses; recent
    presentation suppresses priority (enforces spacing); retirement at
    `retire_streak` consecutive correct under the category RT threshold.
    Retired categories reappear sparsely as maintenance checks."""

    MAINTENANCE_P = 0.08  # chance a retired category is re-probed

    def __init__(self, categories: list[CategoryState], seed: int | None = None):
        self.cats = {c.name: c for c in categories}
        self.trial = 0
        self.rng = random.Random(seed)

    def all_retired(self) -> bool:
        return all(c.retired for c in self.cats.values())

    def _priority(self, c: CategoryState) -> float:
        err_rate = (c.errors / c.attempts) if c.attempts else 0.5  # unseen = medium priority
        recent = c.rts[-3:]
        slow = 0.0
        if recent:
            mean_rt = sum(recent) / len(recent)
            slow = max(0.0, (mean_rt - c.rt_threshold_s) / c.rt_threshold_s)
        recency_suppression = max(0, 3 - (self.trial - c.last_seen)) * 0.8
        jitter = self.rng.uniform(0, 0.4)  # enforces interleaving
        return 1.0 + 2.0 * err_rate + 1.0 * slow - recency_suppression + jitter

    def next_category(self) -> str:
        active = [c for c in self.cats.values() if not c.retired]
        retired = [c for c in self.cats.values() if c.retired]
        if not active:
            pool = retired
        elif retired and self.rng.random() < self.MAINTENANCE_P:
            pool = retired
        else:
            pool = active
        return max(pool, key=self._priority).name

    def record(self, category: str, correct: bool, rt_s: float) -> None:
        c = self.cats[category]
        c.attempts += 1
        c.rts.append(rt_s)
        if not correct:
            c.errors += 1
        fast_and_right = correct and rt_s < c.rt_threshold_s
        c.streak = c.streak + 1 if fast_and_right else 0
        if c.streak >= c.retire_streak:
            c.retired = True
        c.last_seen = self.trial
        self.trial += 1

    def summary(self) -> list[dict]:
        return [
            {
                "category": c.name,
                "trials": c.attempts,
                "accuracy": f"{c.accuracy:.0%}" if c.attempts else "—",
                "median RT (s)": f"{c.median_rt:.1f}" if c.rts else "—",
                "streak": c.streak,
                "status": "RETIRED ✅" if c.retired else "training",
            }
            for c in self.cats.values()
        ]
