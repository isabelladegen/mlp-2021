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

    # Pretrained models
    pretrained_models_path: str = '../data/Models/'

    # Run
    run_one_model: bool = True
    run_model_per_station: bool = True
    log_predictions_to_wandb: bool = True
    run_test_predictions: bool = True

    # Round
    intermediate_rounding: bool = True

    # Sweep
    sweep_training_path: str = '../data/Dev/Sweeping/dev.csv'
    sweep_validation_path: str = '../data/Dev/Sweeping/val.csv'
    sweep_data_percentage: int = 30  # only take x% of the labelled for training and validation for a sweep

    # data processing
    no_nan_in_bikes: bool = True  # removes rows that don't have a label, e.g for means square calculations

    # results
    write_predictions_to_path: str = '../experiment-results/'
    write_results_start_name: str = 'predictions_'

    # Models
    features_data_type: {} = field(
        default_factory=lambda: {Columns.station.value: 'category',
                                 Columns.data_3h_ago.value: 'category',
                                 Columns.num_docks.value: 'category',
                                 Columns.week_hour.value: 'category',
                                 Columns.is_holiday.value: 'category',
                                 Columns.wind_mean_speed.value: 'float64',
                                 Columns.wind_direction.value: 'float64',
                                 Columns.rel_humidity.value: 'float64',
                                 Columns.air_pressure.value: 'float64',
                                 Columns.temperature.value: 'float64',
                                 Columns.full_profile_bikes.value: 'float64',
                                 Columns.full_profile_3h_diff_bikes.value: 'float64',
                                 Columns.short_profile_bikes.value: 'float64',
                                 Columns.short_profile_3h_diff_bikes.value: 'float64'
                                 })
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
                                 Columns.num_docks.value,
                                 Columns.week_hour.value,
                                 Columns.is_holiday.value,
                                 Columns.full_profile_bikes.value,
                                 Columns.full_profile_3h_diff_bikes.value,
                                 Columns.air_pressure.value,
                                 Columns.rel_humidity.value,
                                 Columns.wind_mean_speed.value
                                 ])

    random_forest_n_estimators: int = 100
    random_forest_criterion: str = "absolute_error"
    random_forest_max_depth: int = None
    random_forest_min_samples_split: int = 2
    random_forest_min_samples_leaf: int = 1
    random_forest_min_weight_fraction_leaf: float = 0.0
    random_forest_max_features: str = "auto"  # auto did worst
    random_forest_max_leaf_nodes: int = None
    random_forest_min_impurity_decrease: float = 0.001  # 0 in theory trains better but validates worse
    random_forest_bootstrap: bool = True
    random_forest_oob_score: bool = False
    random_forest_n_jobs: int = 5
    random_forest_random_state: int = 0  # attempt to reproduce
    random_forest_verbose: int = 0
    random_forest_warm_start: bool = False
    random_forest_ccp_alpha: float = 0.001  # small value seems to do better than 0 which is no pruning
    random_forest_max_samples: int = None

    # MLP Regressor
    mlp_features: [str] = field(
        default_factory=lambda: [
                                 Columns.week_hour.value,
                                 Columns.data_3h_ago.value,
                                 Columns.full_profile_bikes.value,
                                 Columns.full_profile_3h_diff_bikes.value,
                                 Columns.short_profile_bikes.value,
                                 Columns.short_profile_3h_diff_bikes.value,
                                 ])
    mlp_hidden_layer_sizes: tuple = (100,)
    mlp_activation: str = 'relu'  # ‘identity’, ‘logistic’, ‘tanh’, ‘relu’
    mlp_solver: str = 'adam'  # ‘lbfgs’, ‘sgd’, ‘adam’ - adam default
    mlp_alpha: float = 0.0001  # L2 penalty
    mlp_batch_size: int = 'auto'
    mlp_learning_rate: str = 'adaptive'  # ‘adaptive’, ‘invscaling’, ‘adaptive’  - constant default
    mlp_learning_rate_init: float = 0.001  # Only used when solver=’sgd’ or ‘adam’.
    mlp_power_t: float = 0.5  # Only used when solver=’sgd’.
    mlp_max_iter: int = 600  # upped max iterations
    mlp_shuffle: bool = True  # Only used when solver=’sgd’ or ‘adam’.
    mlp_random_state: int = None
    mlp_tol: float = 1e-4
    mlp_momentum: float = 0.9  # Should be between 0 and 1. Only used when solver=’sgd’.
    mlp_nesterovs_momentum: bool = True  # Only used when solver=’sgd’ and momentum > 0.
    mlp_early_stopping: bool = False  # only effective when solver=’sgd’ or ‘adam’.
    mlp_validation_fraction: float = 0.1  # Only used if early_stopping is True.
    mlp_beta_1: float = 0.9  # should be in [0, 1). Only used when solver=’adam’
    mlp_beta_2: float = 0.999  # should be in [0, 1). Only used when solver=’adam’.
    mlp_epsilon: float = 1e-8  # Only used when solver=’adam’.
    mlp_n_iter_no_change: int = 10  # Only effective when solver=’sgd’ or ‘adam’.
    mlp_max_fun: int = 15000  # Only used when solver=’lbfgs’. Maximum number of function calls

    def as_dict(self):
        return asdict(self)


@dataclass
class TestConfiguration(Configuration):
    # Wandb
    wandb_mode: str = WandbMode.disabled.value

    # Development data
    development_data_path: str = ''  # write files into test folder for testing

    # Run
    run_one_model: bool = True
    run_model_per_station: bool = True
    log_predictions_to_wandb: bool = True
    run_test_predictions: bool = False

    # Round
    intermediate_rounding: bool = True

    # Results
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
