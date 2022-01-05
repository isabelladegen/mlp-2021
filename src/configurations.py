from dataclasses import dataclass, asdict, field
import enum

from src.Data import Columns

ONE_MODEL = 'One model'
ONE_MODEL = 'One model'
PER_STATION_MODEL = 'Model per station'
TRAINING = 'Training data'
VALIDATION = 'Validation data'
MAE = 'MAE'
PER_STATION = 'per station'


class RunResults(enum.Enum):
    predictions = 1
    wandb = 2


class WandbMode(enum.Enum):
    disabled = 'disabled'  # choose for tests
    online = 'online'
    offline = 'offline'


class WandbLogs(enum.Enum):
    mean_absolute_error = 'Mean absolute error'
    one_model_mae_dev = ONE_MODEL + ' - ' + TRAINING + ': ' + MAE
    one_model_mae_val = ONE_MODEL + ' - ' + VALIDATION + ': ' + MAE
    one_model_mae_per_station_dev = ONE_MODEL + ' - ' + TRAINING + ': ' + MAE + ' ' + PER_STATION
    one_model_mae_per_station_val = ONE_MODEL + ' - ' + VALIDATION + ': ' + MAE + ' ' + PER_STATION
    per_station_mae_dev = PER_STATION_MODEL + ' - ' + TRAINING + ': ' + MAE
    per_station_mae_val = PER_STATION_MODEL + ' - ' + VALIDATION + ': ' + MAE
    per_station_mae_per_station_dev = PER_STATION_MODEL + ' - ' + TRAINING + ': ' + MAE + ' ' + PER_STATION
    per_station_mae_per_station_val = PER_STATION_MODEL + ' - ' + VALIDATION + ': ' + MAE + ' ' + PER_STATION
    predictions = 'Predictions'
    one_model_predictions_dev = 'Predictions one model - ' + TRAINING
    one_model_predictions_val = 'Predictions one model - ' + VALIDATION
    per_station_predictions_dev = 'Predictions model per station - ' + TRAINING
    per_station_predictions_val = 'Predictions model per station - ' + VALIDATION


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
        default_factory=lambda: {Columns.station.value: 'category',
                                 Columns.data_3h_ago.value: 'category',
                                 Columns.num_docks.value: 'category'})
    # Poisson Regressor
    poisson_features: [str] = field(default_factory=lambda: [Columns.data_3h_ago.value])
    poisson_alpha: float = 1.0
    poisson_fit_intercept: bool = True
    poisson_max_iter: int = 100
    poisson_tol: float = 0.0001
    poisson_verbose: int = 0
    poisson_warm_start: bool = False

    # Random Forest Regressor
    random_forest_features: [str] = field(
        default_factory=lambda: [Columns.station.value,
                                 Columns.data_3h_ago.value,
                                 Columns.num_docks.value])
    random_forest_n_estimators: int = 100
    random_forest_criterion: str = "absolute_error"
    random_forest_max_depth: int = None
    random_forest_min_samples_split: int = 2
    random_forest_min_samples_leaf: int = 1
    random_forest_min_weight_fraction_leaf: float = 0.0
    random_forest_max_features: str = "auto"
    random_forest_max_leaf_nodes: int = None
    random_forest_min_impurity_decrease: float = 0.0
    random_forest_bootstrap: bool = True
    random_forest_oob_score: bool = False
    random_forest_n_jobs: int = 5
    random_forest_random_state: int = None
    random_forest_verbose: int = 0
    random_forest_warm_start: bool = False
    random_forest_ccp_alpha: float = 0.0  # 0 no pruning
    random_forest_max_samples: int = None

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
