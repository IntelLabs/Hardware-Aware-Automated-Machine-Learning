import random
import torch
from typing import Dict, Optional


class CacheStrategy:
    def add(self, cache, query, query_embedding, adapter_params):
        pass

    def evict(self, cache):
        pass

    def update(self, cache, index):
        pass


class RandomReplacementStrategy(CacheStrategy):
    def evict(self, cache):
        return random.randint(0, len(cache) - 1)


class LFUStrategy(CacheStrategy):
    def __init__(self):
        self.usage_count = []

    def add(self, cache, query, query_embedding, adapter_params):
        self.usage_count.append(0)

    def evict(self, cache):
        least_used_index = self.usage_count.index(min(self.usage_count))
        self.usage_count.pop(least_used_index)
        return least_used_index

    def update(self, cache, index):
        self.usage_count[index] += 1


class LocalAdapterManager:
    def __init__(self, size: int, threshold: float, strategy: str) -> None:
        self.size = size
        self.threshold = threshold
        self.cache = []
        self.strategy = self._get_strategy(strategy)
    
    def _get_strategy(self, strategy: str) -> CacheStrategy:
        if strategy == "random":
            return RandomReplacementStrategy()
        elif strategy == "lfu":
            return LFUStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def add_to_cache(self, query: str, query_embedding: torch.Tensor, adapter_params: Dict[str, torch.Tensor]) -> None:
        if len(self.cache) >= self.size:
            evict_index = self.strategy.evict(self.cache)
            self.cache.pop(evict_index)
        
        self.cache.append((query, query_embedding, adapter_params))
        self.strategy.add(self.cache, query, query_embedding, adapter_params)

    def retrieve_best_cache_entry(self, query_embedding: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        if not self.cache:
            return None

        # Stack all cached embeddings into a single tensor
        cached_embeddings = torch.stack([item[1].squeeze(0) for item in self.cache])

        # Compute the inner product between the query embedding and all cached embeddings
        scores = torch.matmul(query_embedding, cached_embeddings.transpose(0, 1)).squeeze()

        # Find the best score and its index
        best_score, best_index = torch.max(scores, dim=0)

        if best_score.item() >= self.threshold:
            self.strategy.update(self.cache, best_index)
            return self.cache[best_index]
        else:
            return None

    def __len__(self) -> int:
        return len(self.cache)
