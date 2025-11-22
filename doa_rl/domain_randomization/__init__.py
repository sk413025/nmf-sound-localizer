"""Domain-randomization utilities for generating and loading angle-range shards."""

from doa_rl.domain_randomization.dataset import AngleRangeDataset, ShardSummary, load_shard_summary
from doa_rl.domain_randomization.generator import AngleRange, AngleRangeShardGenerator, GenerationConfig

__all__ = [
    "AngleRange",
    "AngleRangeDataset",
    "AngleRangeShardGenerator",
    "GenerationConfig",
    "ShardSummary",
    "load_shard_summary",
]
