from dataclasses import dataclass, asdict, field
import enum

from src.Data import Columns


class RunResults(enum.Enum):
    predictions = 1
    wandb = 2


class WandbMode(enum.Enum):
    disabled = 'disabled'  # choose for tests
    online = 'online'
    offline = 'offline'


class WandbLogs(enum.Enum):
    mean_absolute_error = 'Mean absolute error'
    predictions = 'Predictions'


@dataclass
class Configuration:
    # Wandb
    wandb_project_name: str = 'mlp-2021'
    wandb_entity: str = 'idegen'
    wandb_mode: str = WandbMode.online.value

    # data
    test_data_path: str = '../data/test.csv'
    training_data_path: str = '../data/Train/'

    # Development data
    development_data_path: str = '../data/Dev/'
    dev_data_filename: str = 'dev.csv'
    val_data_filename: str = 'validation.csv'
    dev_validation_data_split: int = 10  # take 10% of the labelled rows away for validation during development

    # data processing
    no_nan_in_bikes: bool = True  # removes rows that don't have a label, e.g for means square calculations

    # results
    log_predictions: bool = True
    write_predictions_to_path: str = '../experiment-results/'
    write_results_start_name: str = 'predictions_'

    # Models
    features_data_type: {} = field(
        default_factory=lambda: {Columns.data_3h_ago.value: 'category', Columns.num_docks.value: 'category'})
    poisson_features: [str] = field(default_factory=lambda: [Columns.data_3h_ago.value])
    poisson_alpha: float = 1.0
    poisson_fit_intercept: bool = True
    poisson_max_iter: int = 100
    poisson_tol: float = 0.0001
    poisson_verbose: int = 0
    poisson_warm_start: bool = False

    def as_dict(self):
        return asdict(self)


@dataclass
class TestConfiguration(Configuration):
    # Wandb
    wandb_mode: str = WandbMode.disabled.value

    # Development data
    development_data_path: str = ''  # write files into test folder for testing

    # Results
    log_predictions: bool = False
    write_results_start_name: str = 'testing_predictions_'

    # Models  the default model values will change and we don't want flaky tests
    poisson_features: [str] = field(
        default_factory=lambda: [Columns.data_3h_ago.value])
    poisson_alpha: float = 1.0
    poisson_fit_intercept: bool = True
    poisson_max_iter: int = 100
    poisson_tol: float = 0.0001
    poisson_verbose: int = 0
    poisson_warm_start: bool = False
