"""Forward-hook helpers for collecting per-linear input second moments."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from ..core.ldlq import ActivationCovarianceAccumulator


class LinearInputCovarianceCollector:
    """Collect ``E[x x.T]`` for explicitly selected ``nn.Linear`` modules.

    Module selection is exact by name so a calibration run cannot silently
    capture a broader model scope than its manifest declares.
    """

    def __init__(
        self,
        model: nn.Module,
        module_names: Iterable[str],
        *,
        accumulator_device: torch.device | str = "cpu",
        accumulator_dtype: torch.dtype = torch.float32,
    ) -> None:
        names = tuple(module_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("module_names must be nonempty and unique")
        modules = dict(model.named_modules())
        missing = [name for name in names if name not in modules]
        non_linear = [name for name in names if name in modules and not isinstance(modules[name], nn.Linear)]
        if missing:
            raise ValueError(f"unknown module names: {missing}")
        if non_linear:
            raise ValueError(f"selected modules are not nn.Linear: {non_linear}")
        self.accumulators = {
            name: ActivationCovarianceAccumulator(
                modules[name].in_features,
                device=accumulator_device,
                dtype=accumulator_dtype,
            )
            for name in names
        }
        self._handles = [
            modules[name].register_forward_pre_hook(self._make_hook(name)) for name in names
        ]
        self.closed = False

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                raise RuntimeError(f"{name} received no positional input")
            self.accumulators[name].update(inputs[0])
        return hook

    def close(self) -> None:
        if not self.closed:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self.closed = True

    def __enter__(self) -> "LinearInputCovarianceCollector":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def covariances(self, *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        return {name: accumulator.covariance().to(device) for name, accumulator in self.accumulators.items()}

    def metadata(self) -> dict[str, dict[str, int | str]]:
        return {
            name: {
                "feature_count": accumulator.feature_count,
                "sample_count": accumulator.sample_count,
                "accumulator_dtype": str(accumulator.dtype).removeprefix("torch."),
                "accumulator_device": str(accumulator.device),
                "dense_gram_bytes": accumulator.gram.numel() * accumulator.gram.element_size(),
            }
            for name, accumulator in self.accumulators.items()
        }
