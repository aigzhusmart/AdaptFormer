from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    root_dir: str = field(metadata={"help": "Dataset root_dir"})
    image_size: Optional[int] = field(default=256, metadata={"help": "image_size"})
