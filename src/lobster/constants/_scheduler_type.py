from typing import Literal

SchedulerType = Literal[
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
    "reduce_lr_on_plateau",
    "cosine_with_min_lr",
    "warmup_stable_decay",
]
