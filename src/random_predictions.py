import wandb

from src.models.RandomEstimator import RandomEstimator
from src.configurations import Configuration, RunResults, WandbLogs
from src.Data import Data


def run(config: Configuration):
    results = {}  # keep experiment-results of run
    run = wandb.init(project=config.wandb_project_name, entity=config.wandb_entity, mode=config.wandb_mode,
                     notes="random predictions", tags=["random prediction"],
                     config=config.as_dict())
    # Reload the Configuration (to allow for sweeps)
    configuration = Configuration(**wandb.config)

    # No training for random predictions
    one_model = RandomEstimator(configuration)

    # Load test data
    labelled_data = Data(config.no_nan_in_bikes, config.training_data_path)  # do all required preprocessing in here

    # Use models to predict random number of bikes
    random_predictions = one_model.predict_random_numbers_for(labelled_data)
    results[RunResults.predictions] = random_predictions

    # Calculate and log mean average error
    mae = random_predictions.mean_absolute_error()
    wandb.log({
        WandbLogs.mean_absolute_error.value: mae
    })

    results[RunResults.wandb] = wandb

    # Write predictions to csv
    if configuration.run_test_predictions:
        csv_filename = random_predictions.write_to_csv('', configuration)

        # Log predictions to wandb
        prediction_table = wandb.Table(dataframe=random_predictions.predictions_as_df())
        run.log({WandbLogs.predictions.value: prediction_table})

    return results


def main():
    run(Configuration())


if __name__ == "__main__":
    main()
