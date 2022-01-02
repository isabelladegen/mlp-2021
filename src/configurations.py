from dataclasses import dataclass, asdict
import enum


class WandbMode(enum.Enum):
    disabled = 'disabled'  # choose for tests
    online = 'online'
    offline = 'offline'


@dataclass
class Configuration:
    wandb_project_name: str = 'mlp-2021'
    wandb_entity: str = 'idegen'
    wandb_mode: str = WandbMode.online.value

    # data
    test_data_path: str = '../data/Test.csv'
    training_data_path: str = '../data/Train/'

    # data processing
    no_nan_in_bikes: bool = True  # removes rows that don't have a label, e.g for means square calculations

    # results
    log_predictions: bool = True
    write_predictions_to_path: str = '../experiment-results/'
    write_results_start_name: str = 'predictions_'

    def as_dict(self):
        return asdict(self)


@dataclass
class TestConfiguration(Configuration):
    wandb_mode: str = WandbMode.disabled.value
    log_predictions: bool = False
    write_results_start_name: str = 'testing_predictions_'
