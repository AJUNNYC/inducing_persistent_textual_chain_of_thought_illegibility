"""Explicit execution candidates; scientific SFT settings remain frozen.

The default is the published eager baseline. Candidate implementations require
measurements on the selected GPU before production activation; a CPU equality
check is not a GPU throughput or scientific-outcome certificate.
"""
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SFTExecutionProfile:
    attention_implementation: str = "eager"
    validated_loss_counts: bool = False
    defer_finite_checks: bool = False
    cache_collated_inputs: bool = False
    gradient_checkpointing: bool = True
    loss_implementation: str = "spatial_ce"

    def __post_init__(self) -> None:
        if self.attention_implementation not in ("eager", "sdpa"):
            raise ValueError("Supported SFT attention candidates are eager and sdpa")
        if self.loss_implementation not in ("spatial_ce", "flat_ce", "fused_ce", "spatial_fused_backward", "spatial_scheduled_ce"):
            raise ValueError("Supported completion-loss candidates are spatial_ce, flat_ce, fused_ce, spatial_fused_backward and spatial_scheduled_ce")
        if any(type(value) is not bool for value in (
                self.validated_loss_counts, self.defer_finite_checks,
                self.cache_collated_inputs, self.gradient_checkpointing)):
            raise ValueError("SFT execution flags must be actual booleans")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SFTExecutionProfile":
        """Read earlier explicit profiles with newly added baseline defaults."""
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown SFT execution-profile fields")
        return cls(**value)


def resolve_execution_profile(profile: SFTExecutionProfile | None) -> SFTExecutionProfile:
    if profile is None:
        return SFTExecutionProfile()
    if not isinstance(profile, SFTExecutionProfile):
        raise ValueError("SFT execution profile must be an explicit validated profile")
    return profile
