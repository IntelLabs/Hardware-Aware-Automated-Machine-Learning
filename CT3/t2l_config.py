import argparse
from typing import Optional
from dataclasses import dataclass


@dataclass
class T2LConfig:
    t2l: bool = True
    hypernetwork_path: Optional[str] = None

    def generate_result_folder_name(self, task: str, limit: Optional[int] = None) -> str:
        """Generate a result folder name based on the configuration and task details."""
        filename_parts = [f"results.{task}"]
        if limit is not None and limit != -1:
            filename_parts.append(f"limit_{limit}")
        if self.t2l:
            filename_parts.append("t2l")
        return ".".join(filename_parts)


def add_t2l_args(parser: argparse.ArgumentParser) -> None:
    """Add T2L configuration arguments to the parser."""
    parser.add_argument("--t2l", action="store_true", help="Flag to indicate whether to perform test-time training.")
    parser.add_argument("--hypernetwork_path", type=str, default=None, help="Path to the hypernetwork.")


def parse_t2l_args(args: argparse.Namespace) -> T2LConfig:
    """Parse arguments into a T2LConfig object."""
    return T2LConfig(
        t2l=args.t2l,
        hypernetwork_path=args.hypernetwork_path
    )
