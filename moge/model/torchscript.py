from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from . import import_model_class_by_version


class TorchScriptMoGeModel(nn.Module):
    """TorchScript-friendly wrapper around :class:`~moge.model.MoGeModel`.

    The wrapper follows the guidance laid out in ``CatFileCreation.md`` by
    exposing static attributes that can be promoted to Nuke knobs and by
    constraining the forward signature to a single tensor input. The wrapped
    model is expected to behave identically to the original implementation,
    while avoiding dynamic Python features that TorchScript cannot compile.
    """

    def __init__(
        self,
        version: str = "v1",
        num_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        model_cls = import_model_class_by_version(version)
        base_model = model_cls()
        base_model.eval()
        self.model = base_model

        max_tokens = int(self.model.num_tokens_range[1])
        self.numTokens: int = max_tokens if num_tokens is None else int(num_tokens)
        self.applyMask: bool = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        outputs = self.model.forward(image, self.numTokens)
        points = outputs["points"]
        mask = outputs["mask"]

        if self.applyMask:
            expanded_mask = mask.unsqueeze(-1)
            points = points * expanded_mask
        else:
            expanded_mask = mask.unsqueeze(-1)

        points_chw = points.permute(0, 3, 1, 2)
        mask_chw = expanded_mask.permute(0, 3, 1, 2)
        return torch.cat([points_chw, mask_chw], dim=1)

    @torch.jit.export
    def set_num_tokens(self, value: int) -> None:
        self.numTokens = int(value)

    @torch.jit.export
    def set_apply_mask(self, value: bool) -> None:
        self.applyMask = bool(value)


__all__ = ["TorchScriptMoGeModel"]
