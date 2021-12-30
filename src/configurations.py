from dataclasses import dataclass, asdict


@dataclass
class Configuration:

    def as_dict(self):
        return asdict(self)
