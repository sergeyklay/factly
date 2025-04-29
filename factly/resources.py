import os
from functools import lru_cache


class ResourceManager:
    @staticmethod
    @lru_cache(maxsize=1)
    def get_optimal_workers(min_workers=2, max_workers=None, memory_factor=0.8):
        """Calculate optimal worker count based on system resources.

        Args:
            min_workers: Minimum number of workers regardless of resources
            max_workers: Hard cap on maximum workers (None for no limit)
            memory_factor: Reduce worker count if memory is constrained (0-1)

        Returns:
            int: Recommended worker count
        """
        # Base calculation on CPU count
        cpu_count = os.cpu_count() or 4

        # Start with CPU-based calculation (typically 2x CPU cores for I/O bound tasks)
        workers = cpu_count * 2

        # Check memory constraints
        if memory_factor < 1.0:
            import psutil

            try:
                memory = psutil.virtual_memory()
                # Reduce workers if memory utilization is high
                if memory.percent > 75:
                    memory_workers = int(cpu_count * memory_factor)
                    workers = min(workers, memory_workers)
            except Exception:
                pass  # Fall back to CPU calculation if memory check fails

        # Apply limits
        workers = max(min_workers, workers)
        if max_workers:
            workers = min(workers, max_workers)

        return workers

    @classmethod
    def default_workers(cls) -> int:
        """Dynamically calculate optimal worker count."""
        return ResourceManager.get_optimal_workers(min_workers=2, max_workers=30)
