import wandb
import enum
from src.configurations import Configuration
from src.test_data import TestData


class WandbLogs(enum.Enum):
    MEAN_AVERAGE_ERROR = 'Mean average error'


class OneModel:
    pass


def run(config: Configuration = Configuration()):
    wandb.init(
        project=config.wandb_project_name,
        entity=config.wandb_entity,
        mode=config.wandb_mode,
        name="getting started",
        notes="just testing",
        tags=["testing"],
        config=Configuration().as_dict()
    )
    # Reload the Configuration (to allow for sweeps)
    configuration = Configuration(**wandb.config)

    # Load training data
    # training_data = TrainingData()

    # Train models
    # model_per_station = ModelPerStation()
    one_model = OneModel(configuration)

    # Load test data
    test_data = TestData(configuration)  # do all required preprocessing in here

    # Use models to predict number of bikes
    if configuration.predict_random_numbers:
        predictions = one_model.predict_random_numbers_for(test_data)

        # Calculate and log mean average error
        mean_average_error = predictions.mean_average_error()
        wandb.log({
            WandbLogs.MEAN_AVERAGE_ERROR.value: mean_average_error
        })

        # Write csv for submission
        predictions.write_results_to_csv()

    return wandb, predictions
