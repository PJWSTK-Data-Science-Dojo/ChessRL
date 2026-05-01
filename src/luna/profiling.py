"""Training / pipeline profiling helpers (wall-clock phases + optional PyTorch traces).

Typical ``[self_play detail]`` breakdown (small net, batched self-play): ``expand_backup``
often ~45-55% of MCTS interior (CPU child wiring after each recurrent batch),
``recurrent_inf`` and ``initial_inf`` ~20-30% each, ``selection`` ~0-5%, ``encode`` small.
Use this to choose optimizations: top-K policy H2D helps recurrent + expand; Numba helps
if ``selection`` grows with wide trees.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SelfPlayMCTSTimings:
    """Mutable accumulators for :class:`~luna.mcts.BatchedMCTS.search_batch` (one training iter)."""

    encode_s: float = 0.0
    initial_inf_s: float = 0.0
    selection_s: float = 0.0
    recurrent_inf_s: float = 0.0
    expand_backup_s: float = 0.0
    finalize_s: float = 0.0
    search_batch_calls: int = 0

    def mcts_sum_s(self) -> float:
        return (
            self.encode_s
            + self.initial_inf_s
            + self.selection_s
            + self.recurrent_inf_s
            + self.expand_backup_s
            + self.finalize_s
        )


@dataclass
class IterProfileStats:
    """Wall-clock seconds for one outer training iteration."""

    iter_index: int
    self_play_s: float = 0.0
    replay_save_s: float = 0.0
    checkpoint_io_s: float = 0.0
    train_s: float = 0.0
    arena_s: float = 0.0
    accept_s: float = 0.0
    total_s: float = 0.0
    # Self-play interior (filled when ``TrainingRunConfig.profile`` is true).
    self_play_env_s: float = 0.0
    self_play_mcts_encode_s: float = 0.0
    self_play_mcts_initial_inf_s: float = 0.0
    self_play_mcts_selection_s: float = 0.0
    self_play_mcts_recurrent_inf_s: float = 0.0
    self_play_mcts_expand_backup_s: float = 0.0
    self_play_mcts_finalize_s: float = 0.0
    self_play_search_batch_calls: int = 0

    def to_log_lines(self) -> str:
        if self.total_s <= 0:
            return f"[profile] iter {self.iter_index}: total_s=0 (skipped or not measured)"
        lines = [
            "[profile] iter {} — wall clock (total {:.1f}s):",
            "  self_play {:.1f}s ({:.0f}%)",
            "  replay_save {:.1f}s ({:.0f}%)",
            "  checkpoint_io {:.1f}s ({:.0f}%)",
            "  train {:.1f}s ({:.0f}%)",
            "  arena {:.1f}s ({:.0f}%)",
            "  accept_checkpoints {:.1f}s ({:.0f}%)",
        ]

        def pct(x: float) -> float:
            return 100.0 * x / self.total_s

        out = [
            lines[0].format(self.iter_index, self.total_s),
            lines[1].format(self.self_play_s, pct(self.self_play_s)),
            lines[2].format(self.replay_save_s, pct(self.replay_save_s)),
            lines[3].format(self.checkpoint_io_s, pct(self.checkpoint_io_s)),
            lines[4].format(self.train_s, pct(self.train_s)),
            lines[5].format(self.arena_s, pct(self.arena_s)),
            lines[6].format(self.accept_s, pct(self.accept_s)),
        ]
        if self.self_play_search_batch_calls > 0 and self.self_play_s > 0:
            sp = self.self_play_s
            mcts_interior = (
                self.self_play_mcts_encode_s
                + self.self_play_mcts_initial_inf_s
                + self.self_play_mcts_selection_s
                + self.self_play_mcts_recurrent_inf_s
                + self.self_play_mcts_expand_backup_s
                + self.self_play_mcts_finalize_s
            )
            out.append(
                f"  [self_play detail] env {self.self_play_env_s:.3f}s ({100.0 * self.self_play_env_s / sp:.0f}% of self_play) | "
                f"MCTS interior {mcts_interior:.3f}s | search_batch calls {self.self_play_search_batch_calls}"
            )
            out.append(
                f"    encode {self.self_play_mcts_encode_s:.3f}s | initial_inf {self.self_play_mcts_initial_inf_s:.3f}s | select {self.self_play_mcts_selection_s:.3f}s | "
                f"recurrent_inf {self.self_play_mcts_recurrent_inf_s:.3f}s | expand_backup {self.self_play_mcts_expand_backup_s:.3f}s | finalize {self.self_play_mcts_finalize_s:.3f}s"
            )
            if mcts_interior > 0:
                r = 100.0 / mcts_interior
                out.append(
                    f"    (share of MCTS) initial {self.self_play_mcts_initial_inf_s * r:.0f}% | recurrent {self.self_play_mcts_recurrent_inf_s * r:.0f}% | "
                    f"select {self.self_play_mcts_selection_s * r:.0f}% | expand {self.self_play_mcts_expand_backup_s * r:.0f}% | encode {self.self_play_mcts_encode_s * r:.0f}% | finalize {self.self_play_mcts_finalize_s * r:.0f}%"
                )
        return "\n".join(out)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_iter_summaries_json(path: str, rows: list[IterProfileStats]) -> None:
    """Writes all iterations to one JSON file."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = [r.to_dict() for r in rows]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
