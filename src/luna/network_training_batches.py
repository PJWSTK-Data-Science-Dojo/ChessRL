"""Ordered replay prefetch for the learner loop."""

from __future__ import annotations

from concurrent.futures import Future

from luna.config import MCTSParams
from luna.expert_anchor import ExpertAnchorBatchSource
from luna.network_training_types import TrainingSettings
from luna.network_types import NetworkRuntime, PreparedBatch
from luna.replay_buffer import PrioritizedReplayBuffer


class TrainingBatchSource:
    def __init__(
        self,
        network: NetworkRuntime,
        replay: PrioritizedReplayBuffer,
        settings: TrainingSettings,
        mcts_params: MCTSParams | None,
        expert_anchor: ExpertAnchorBatchSource | None,
    ) -> None:
        self._network = network
        self._replay = replay
        self._settings = settings
        self._mcts_params = mcts_params
        self._expert_anchor = expert_anchor
        self._asynchronous = network._async_batch_prefetch(settings.steps)
        self._future: Future[PreparedBatch] | None = None
        self._future_step: int | None = None

    def start(self) -> None:
        if self._asynchronous:
            self._submit(self._network._global_step + 1)

    def get(self, training_step: int, retry: PreparedBatch | None) -> PreparedBatch:
        if retry is not None:
            return retry
        if not self._asynchronous or self._future is None:
            return self._prepare(training_step)
        if self._future_step != training_step:
            raise RuntimeError("Asynchronous replay prefetch is out of sequence")
        prepared = self._future.result()
        self._future = None
        self._future_step = None
        return prepared

    def schedule_next(self, training_step: int, step_in_call: int) -> None:
        if self._asynchronous and self._future is None and step_in_call < self._settings.steps:
            self._submit(training_step + 1)

    def _submit(self, training_step: int) -> None:
        executor = self._network._prefetch_executor
        if executor is None:
            raise RuntimeError("Asynchronous replay prefetch has no executor")
        self._future_step = training_step
        self._future = executor.submit(self._prepare, training_step)

    def _prepare(self, training_step: int) -> PreparedBatch:
        prepared = self._network._prepare_batch(
            self._replay,
            self._settings.batch_size,
            self._settings.unroll,
            self._network._learner.td_steps,
            self._settings.discount,
            training_step,
            self._mcts_params,
        )
        if self._expert_anchor is None:
            return prepared
        return prepared._replace(expert_anchor=self._expert_anchor.next_batch())
