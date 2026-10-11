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

    def __post_init__(self) -> None:
        if self.attention_implementation not in ("eager", "sdpa"):
            raise ValueError("Supported SFT attention candidates are eager and sdpa")
        if type(self.validated_loss_counts) is not bool or type(self.defer_finite_checks) is not bool:
            raise ValueError("SFT execution flags must be actual booleans")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_execution_profile(profile: SFTExecutionProfile | None) -> SFTExecutionProfile:
    if profile is None:
        return SFTExecutionProfile()
    if not isinstance(profile, SFTExecutionProfile):
        raise ValueError("SFT execution profile must be an explicit validated profile")
    return profile
