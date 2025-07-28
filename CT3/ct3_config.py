import argparse
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class CT3Config:
    ttt: bool = True
    knowledge_base_path: Optional[str] = None
    num_ttt_samples: int = 32
    num_epochs: int = 2
    learning_rate: float = 5e-5
    batch_size: int = 1
    lora_rank: int = 128
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"])
    data_reduction: bool = False
    faiss_index: str = "IndexFlatIP"
    only_faiss: bool = False
    shuffle_ttt_samples: bool = False
    weighted_sample_training: bool = False
    local_adapter_managing: bool = False
    local_adapter_manager_size: int = 4
    local_adapter_manager_threshold: float = 0.4
    local_adapter_manager_strategy: str = "lfu"
    use_ipex_llm: bool = False
    analysis: bool = False
    cache_search_info_path: Optional[str] = None

    def generate_result_folder_name(self, task: str, limit: Optional[int] = None) -> str:
        """Generate a result folder name based on the configuration and task details."""
        filename_parts = [f"results.{task}"]
        if limit is not None and limit != -1:
            filename_parts.append(f"limit_{limit}")
        if self.ttt:
            filename_parts.append("ttt")
            if self.data_reduction:
                filename_parts.append("data_reduction")
            if self.faiss_index != "IndexFlatIP":
                filename_parts.append(self.faiss_index)
            if self.only_faiss:
                filename_parts.append("only_faiss")
            if self.shuffle_ttt_samples:
                filename_parts.append("sts")
            if self.weighted_sample_training:
                filename_parts.append("wst")
            if self.local_adapter_managing:
                filename_parts.append(f"local_adapter_managing_{self.local_adapter_manager_size}_{self.local_adapter_manager_threshold}_{self.local_adapter_manager_strategy}")
            if self.use_ipex_llm:
                filename_parts.append("use_ipex_llm")
            filename_parts.append(f"samples_{self.num_ttt_samples}")
            filename_parts.append(f"epochs_{self.num_epochs}")
            filename_parts.append(f"lr_{self.learning_rate}")
            filename_parts.append(f"batch_{self.batch_size}")
            filename_parts.append(f"lora_rank_{self.lora_rank}")
            filename_parts.append(f"lora_alpha_{self.lora_alpha}")
            filename_parts.append(f"target_modules_{'_'.join(self.target_modules)}")
        return ".".join(filename_parts)


def add_ct3_args(parser: argparse.ArgumentParser) -> None:
    """Add CT3 configuration arguments to the parser."""
    parser.add_argument("--ttt", action="store_true", help="Flag to indicate whether to perform test-time training.")
    parser.add_argument("--knowledge_base_path", type=str, default=None, help="Path to the knowledge base.")
    parser.add_argument("--num_ttt_samples", type=int, default=32, help="Number of training samples for test-time training.")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs for test-time training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for test-time training.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for test-time training.")
    parser.add_argument("--lora_rank", type=int, default=128, help="LoRA rank for test-time training.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha for test-time training.")
    parser.add_argument("--target_modules", type=str, nargs='+', default=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"], help="Target modules for LoRA.")
    parser.add_argument("--data_reduction", action="store_true", help="Flag to indicate whether to perform data reduction in test-time training.")
    parser.add_argument("--faiss_index", type=str, default="IndexFlatIP", help="Faiss index")
    parser.add_argument("--only_faiss", action="store_true", help="Flag to indicate whether to only use Faiss for search.")
    parser.add_argument("--shuffle_ttt_samples", action="store_true", help="Flag to indicate whether to shuffle TTT training samples.")
    parser.add_argument("--weighted_sample_training", action="store_true", help="Flag to indicate whether to perform weighted sample test-time training.")
    parser.add_argument("--local_adapter_managing", action="store_true", help="Flag to indicate whether to use TTT cache.")
    parser.add_argument("--local_adapter_manager_size", type=int, default=4, help="Size of the cache.")
    parser.add_argument("--local_adapter_manager_threshold", type=float, default=0.4, help="Threshold for cache.")
    parser.add_argument("--local_adapter_manager_strategy", type=str, default="lfu", help="Threshold for cache.")
    parser.add_argument("--use_ipex_llm", action="store_true", help="Flag to indicate whether to apply IPEX-LLM")
    parser.add_argument("--analysis", action="store_true", help="Flag to indicate whether to log some analytical data.")
    parser.add_argument("--cache_search_info_path", type=str, help="Path to the cache search results")


def parse_ct3_args(args: argparse.Namespace) -> CT3Config:
    """Parse arguments into a CT3Config object."""
    return CT3Config(
        ttt=args.ttt,
        knowledge_base_path=args.knowledge_base_path,
        num_ttt_samples=args.num_ttt_samples,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        data_reduction=args.data_reduction,
        faiss_index=args.faiss_index,
        only_faiss=args.only_faiss,
        shuffle_ttt_samples=args.shuffle_ttt_samples,
        weighted_sample_training=args.weighted_sample_training,
        local_adapter_managing=args.local_adapter_managing,
        local_adapter_manager_size=args.local_adapter_manager_size,
        local_adapter_manager_threshold=args.local_adapter_manager_threshold,
        local_adapter_manager_strategy=args.local_adapter_manager_strategy,
        use_ipex_llm=args.use_ipex_llm,
        analysis=args.analysis,
        cache_search_info_path=args.cache_search_info_path
    )


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT3 Configuration")
    add_ct3_args(parser)
    args = parser.parse_args()
    config = parse_ct3_args(args)
    print(config)
