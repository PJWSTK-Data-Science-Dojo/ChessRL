"""Persistent parent-side pool for isolated self-play actors."""

from __future__ import annotations

import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, NoReturn, Self, TypeVar

import torch
from loguru import logger

from luna.config import TrainingRunConfig
from luna.replay_buffer import Trajectory
from luna.self_play_worker import (
    _actor_entry,
    _ActorCollectionDone,
    _ActorFailure,
    _ActorReady,
    _ActorRequest,
    _ActorResponse,
    _ActorTrajectory,
    _CollectRequest,
    _StopRequest,
)
from luna.self_play_worker import (
    _actor_learner_config as _actor_learner_config,
)
from luna.self_play_worker import (
    derive_actor_seed as derive_actor_seed,
)
from luna.self_play_worker import (
    partition_episode_counts as partition_episode_counts,
)

if TYPE_CHECKING:
    from luna.network import LunaNetwork


class SelfPlayActorError(RuntimeError):
    """An actor failed or violated the parent/actor protocol."""


_FutureResult = TypeVar("_FutureResult")


class SelfPlayActorPool:
    """Persistent spawned actors synchronized from one parent learner."""

    def __init__(
        self,
        network: LunaNetwork,
        run: TrainingRunConfig,
        *,
        worker_count: int,
        base_seed: int,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        self._network = network
        self._base_seed = base_seed
        self._timeout_s = run.self_play_actor_timeout_s
        self._closed = False
        self._context: SpawnContext = torch.multiprocessing.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[BaseProcess] = []
        self._cache_root = tempfile.TemporaryDirectory(prefix="luna-inductor-actors-")

        initialized = False
        try:
            for actor_id in range(worker_count):
                parent_connection, child_connection = self._context.Pipe(duplex=True)
                process = self._context.Process(
                    target=_actor_entry,
                    args=(
                        actor_id,
                        child_connection,
                        run,
                        network._learner,
                        base_seed,
                        str(Path(self._cache_root.name) / f"actor-{actor_id}"),
                    ),
                    name=f"luna-self-play-{actor_id}",
                )
                process.start()
                child_connection.close()
                self._connections.append(parent_connection)
                self._processes.append(process)
            self._wait_until_ready()
            initialized = True
        finally:
            if not initialized:
                self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, exc_traceback
        self.close()

    def collect(self, num_episodes: int, *, generation: int) -> list[Trajectory]:
        """Synchronize current weights and collect exactly ``num_episodes`` games."""
        if self._closed:
            raise RuntimeError("Self-play actor pool is closed")
        if generation < 0:
            raise ValueError("generation must be non-negative")
        deadline = time.monotonic() + self._timeout_s
        counts = partition_episode_counts(num_episodes, len(self._processes))
        self._raise_for_dead_workers(set(range(len(counts))))
        snapshot = self._shared_cpu_snapshot(deadline)
        requests: dict[int, _CollectRequest] = {}
        for actor_id, episode_count in enumerate(counts):
            requests[actor_id] = _CollectRequest(
                generation=generation,
                seed=derive_actor_seed(self._base_seed, actor_id, generation),
                episode_count=episode_count,
                state_dict=snapshot,
            )

        executor = ThreadPoolExecutor(max_workers=len(counts), thread_name_prefix="luna-actor-io")
        completed = False
        try:
            send_futures = {
                executor.submit(self._send_blocking, actor_id, request): actor_id
                for actor_id, request in requests.items()
            }
            self._await_futures(send_futures, deadline, phase="sending work to")
            del requests, snapshot

            receive_futures = {
                executor.submit(
                    self._receive_collection_blocking,
                    actor_id,
                    counts[actor_id],
                    generation,
                ): actor_id
                for actor_id in range(len(counts))
            }
            results = self._await_futures(receive_futures, deadline, phase="waiting for")
            completed = True
        finally:
            if not completed:
                self._shutdown(graceful=False)
            executor.shutdown(wait=True, cancel_futures=True)

        trajectories = [trajectory for actor_id in sorted(results) for trajectory in results[actor_id]]
        logger.info(
            "Collected {} self-play games from {} isolated actors (generation {})",
            len(trajectories),
            len(counts),
            generation,
        )
        return trajectories

    def close(self) -> None:
        """Ask actors to stop, then terminate only those that do not respond."""
        self._shutdown(graceful=True)

    def _shutdown(self, *, graceful: bool) -> None:
        if self._closed:
            return
        self._closed = True
        if graceful:
            for actor_id, process in enumerate(self._processes):
                if process.is_alive():
                    with suppress(BrokenPipeError, EOFError, OSError):
                        self._connections[actor_id].send(_StopRequest())
            deadline = time.monotonic() + min(self._timeout_s, 10.0)
            for process in self._processes:
                process.join(timeout=max(0.0, deadline - time.monotonic()))
        else:
            for process in self._processes:
                if process.is_alive():
                    process.terminate()
            for connection in self._connections:
                connection.close()
            for process in self._processes:
                process.join(timeout=1.0)

        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=5.0)
        for process in self._processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        if graceful:
            for connection in self._connections:
                connection.close()
        self._cache_root.cleanup()

    def _shared_cpu_snapshot(self, deadline: float) -> dict[str, torch.Tensor]:
        snapshot: dict[str, torch.Tensor] = {}
        for name, tensor in self._network.nnet.state_dict().items():
            snapshot[name] = tensor.detach().to(device="cpu", copy=True).contiguous().share_memory_()
            self._raise_if_timed_out(deadline, phase="preparing the model snapshot")
        return snapshot

    def _await_futures(
        self,
        futures: dict[Future[_FutureResult], int],
        deadline: float,
        *,
        phase: str,
    ) -> dict[int, _FutureResult]:
        pending = set(futures)
        results: dict[int, _FutureResult] = {}
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                waiting_for = ", ".join(str(futures[future]) for future in pending)
                self._abort_timeout(phase, waiting_for)
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            if not done:
                waiting_for = ", ".join(str(futures[future]) for future in pending)
                self._abort_timeout(phase, waiting_for)
            for future in done:
                actor_id = futures[future]
                try:
                    results[actor_id] = future.result()
                except SelfPlayActorError as exc:
                    self._abort(str(exc))
            self._raise_if_timed_out(deadline, phase=f"{phase} self-play actors")
        return results

    def _raise_if_timed_out(self, deadline: float, *, phase: str) -> None:
        if time.monotonic() >= deadline:
            self._abort_timeout(phase)

    def _abort_timeout(self, phase: str, waiting_for: str | None = None) -> NoReturn:
        suffix = f": {waiting_for}" if waiting_for else ""
        self._abort(f"Timed out after {self._timeout_s:g}s {phase}{suffix}")

    def _send_blocking(self, actor_id: int, request: _ActorRequest) -> None:
        connection = self._connections[actor_id]
        try:
            connection.send(request)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise SelfPlayActorError(f"Could not send work to self-play actor {actor_id}: {exc}") from exc

    def _receive_blocking(self, actor_id: int) -> _ActorResponse:
        connection = self._connections[actor_id]
        try:
            response = connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise SelfPlayActorError(f"Lost connection to self-play actor {actor_id}: {exc}") from exc
        if not isinstance(response, _ActorReady | _ActorTrajectory | _ActorCollectionDone | _ActorFailure):
            raise SelfPlayActorError(f"Actor {actor_id} sent an unsupported response: {type(response).__name__}")
        return response

    def _receive_collection_blocking(
        self,
        actor_id: int,
        episode_count: int,
        generation: int,
    ) -> list[Trajectory]:
        trajectories: list[Trajectory | None] = [None] * episode_count
        while True:
            response = self._receive_blocking(actor_id)
            if isinstance(response, _ActorFailure):
                raise SelfPlayActorError(self._failure_message(response))
            if isinstance(response, _ActorTrajectory):
                if response.actor_id != actor_id or response.generation != generation:
                    raise SelfPlayActorError(
                        f"Actor protocol mismatch: expected actor={actor_id}, generation={generation}; "
                        f"got actor={response.actor_id}, generation={response.generation}"
                    )
                episode_index = response.episode_index
                if not 0 <= episode_index < episode_count:
                    raise SelfPlayActorError(
                        f"Actor {actor_id} returned trajectory index {episode_index}; "
                        f"expected an index in [0, {episode_count})"
                    )
                if trajectories[episode_index] is not None:
                    raise SelfPlayActorError(f"Actor {actor_id} returned duplicate trajectory index {episode_index}")
                trajectories[episode_index] = response.trajectory
                continue
            if not isinstance(response, _ActorCollectionDone):
                raise SelfPlayActorError(f"Actor {actor_id} sent an unexpected {type(response).__name__}")
            if response.actor_id != actor_id or response.generation != generation:
                raise SelfPlayActorError(
                    f"Actor protocol mismatch: expected actor={actor_id}, generation={generation}; "
                    f"got actor={response.actor_id}, generation={response.generation}"
                )
            if response.episode_count != episode_count:
                raise SelfPlayActorError(
                    f"Actor {actor_id} completed {response.episode_count} trajectories; expected {episode_count}"
                )
            missing = [index for index, trajectory in enumerate(trajectories) if trajectory is None]
            if missing:
                raise SelfPlayActorError(f"Actor {actor_id} completed with missing trajectory indices: {missing}")
            return [trajectory for trajectory in trajectories if trajectory is not None]

    def _wait_until_ready(self) -> None:
        pending = set(range(len(self._processes)))
        deadline = time.monotonic() + self._timeout_s
        while pending:
            made_progress = False
            for actor_id in tuple(pending):
                response = self._receive_if_available(actor_id)
                if response is None:
                    continue
                made_progress = True
                if isinstance(response, _ActorFailure):
                    self._fail(response)
                if not isinstance(response, _ActorReady) or response.actor_id != actor_id:
                    self._abort(f"Actor {actor_id} sent an invalid startup response")
                pending.remove(actor_id)
            self._raise_for_dead_workers(pending)
            if time.monotonic() >= deadline:
                waiting_for = ", ".join(str(actor_id) for actor_id in sorted(pending))
                self._abort(f"Timed out waiting {self._timeout_s:g}s for self-play actors to start: {waiting_for}")
            if not made_progress:
                time.sleep(0.01)

    def _receive_if_available(self, actor_id: int, *, timeout_s: float = 0.0) -> _ActorResponse | None:
        connection = self._connections[actor_id]
        try:
            if not connection.poll(timeout_s):
                return None
            response = connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._abort(f"Lost connection to self-play actor {actor_id}: {exc}")
        if not isinstance(response, _ActorReady | _ActorTrajectory | _ActorCollectionDone | _ActorFailure):
            self._abort(f"Actor {actor_id} sent an unsupported response: {type(response).__name__}")
        return response

    def _raise_for_dead_workers(self, pending: set[int]) -> None:
        for actor_id in pending:
            process = self._processes[actor_id]
            if not process.is_alive():
                # A child sends its structured failure before exiting. Give the
                # pipe a brief final drain so the actionable traceback wins the
                # race against observing the process exit.
                response = self._receive_if_available(actor_id, timeout_s=0.1)
                if isinstance(response, _ActorFailure):
                    self._fail(response)
                self._abort(f"Self-play actor {actor_id} exited unexpectedly with code {process.exitcode}")

    def _fail(self, failure: _ActorFailure) -> NoReturn:
        self._abort(self._failure_message(failure))

    @staticmethod
    def _failure_message(failure: _ActorFailure) -> str:
        generation = "startup" if failure.generation is None else str(failure.generation)
        return (
            f"Self-play actor {failure.actor_id} failed during generation {generation}: "
            f"{failure.exception_type}: {failure.message}\n{failure.traceback_text}"
        )

    def _abort(self, message: str) -> NoReturn:
        self._shutdown(graceful=False)
        raise SelfPlayActorError(message)
