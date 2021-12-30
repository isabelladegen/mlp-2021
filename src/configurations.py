from dataclasses import dataclass, asdict
import enum


class WandbMode(enum.Enum):
    DISABLED = 'disabled'  # choose for tests
    ONLINE = 'online'
    OFFLINE = 'offline'


@dataclass
class Configuration:
    wandb_project_name: str = 'mlp-2021'
    wandb_entity: str = 'idegen'
    wandb_mode: str = WandbMode.ONLINE.value

    def as_dict(self):
        return asdict(self)


@dataclass
class TestConfiguration(Configuration):
    wandb_mode: str = WandbMode.DISABLED.value
