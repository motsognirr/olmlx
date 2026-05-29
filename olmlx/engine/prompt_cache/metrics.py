"""Per-store hit/miss/eviction counters for prompt cache (issue #365)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class CacheMetrics:
    cache_id_hits: int = 0
    cache_id_misses: int = 0
    radix_hits: int = 0
    radix_misses: int = 0
    evictions_ram: int = 0
    evictions_disk: int = 0
    bytes_in_ram: int = 0
    bytes_on_disk: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
