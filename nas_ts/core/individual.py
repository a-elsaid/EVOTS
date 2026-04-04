from dataclasses import dataclass, field
from typing import Dict, Optional, ClassVar
from .genome_v2 import Genome

@dataclass
class Individual:
    genome: Genome
    metrics: Optional[Dict[str, float]] = None
    fitness: Optional[float] = None
    _id_counter: ClassVar[int] = 0

    # Instance ID (auto-filled)
    id: int = field(init=False)

    def __post_init__(self):
        self.id = Individual._id_counter
        Individual._id_counter += 1