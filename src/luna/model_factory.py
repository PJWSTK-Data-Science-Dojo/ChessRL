"""Factory for versioned Luna model architectures."""

from collections.abc import Callable

from luna.balanced_networks import BalancedNetworks
from luna.config import MODEL_NAMES, EzV2LearnerConfig, ModelName
from luna.ezv2_networks import EZV2Networks
from luna.game.chess_game import ChessGame

ModelBuilder = Callable[[ChessGame, EzV2LearnerConfig], EZV2Networks]

_MODEL_BUILDERS: dict[ModelName, ModelBuilder] = {
    "baseline": EZV2Networks,
    "balanced": BalancedNetworks,
}


def available_models() -> tuple[ModelName, ...]:
    """Return stable configuration keys accepted by the model factory."""
    return MODEL_NAMES


def build_model(game: ChessGame, config: EzV2LearnerConfig) -> EZV2Networks:
    """Construct the architecture selected in the learner configuration."""
    try:
        builder = _MODEL_BUILDERS[config.model_name]
    except KeyError as exc:
        available = ", ".join(available_models())
        raise ValueError(f"Unknown model {config.model_name!r}; available models: {available}") from exc
    return builder(game, config)
