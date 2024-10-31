from dataclasses import asdict, dataclass, fields


@dataclass
class Descriptors:
    @classmethod
    def descriptor_names(cls) -> list[str]:
        return [f.name for f in fields(cls) if "not_a_descriptor" not in f.metadata]

    def asdict(self) -> dict:
        return asdict(self)
