import enum

import wandb


class LogKeys(enum.Enum):
    mae_dev = 'mae dev'
    mae_val = 'mae val'
    mae_per_station_dev = 'mae per station dev'
    mae_per_station_val = 'mae per station val'
    predictions_dev = 'predictions dev'
    predictions_val = 'predictions val'


def train_predict_evaluate_log_for_model_and_data(model, training_data, validation_data, keys, wandb_run,
                                                  log_prediction_tables):
    # train
    model.fit()

    # predict
    training_result = model.predict(training_data)
    validation_result = model.predict(validation_data)

    # evaluate
    mae_dev = training_result.mean_absolute_error()
    mae_val = validation_result.mean_absolute_error()
    mae_per_station_dev = training_result.mean_absolute_error_per_station()
    mae_per_station_val = validation_result.mean_absolute_error_per_station()
    # log results
    wandb.log({
        keys[LogKeys.mae_dev.value]: mae_dev,
        keys[LogKeys.mae_val.value]: mae_val,
    })
    log_per_station_mae_to_wand(keys[LogKeys.mae_per_station_dev.value], mae_per_station_dev)
    log_per_station_mae_to_wand(keys[LogKeys.mae_per_station_val.value], mae_per_station_val)

    # Log prediction tables to wandb
    if log_prediction_tables:
        prediction_table_dev = wandb.Table(dataframe=training_result.results_df)
        prediction_table_val = wandb.Table(dataframe=validation_result.results_df)
        wandb_run.log({keys[LogKeys.predictions_dev.value]: prediction_table_dev})
        wandb_run.log({keys[LogKeys.predictions_val.value]: prediction_table_val})

    return validation_result


# takes a  dictionary { station : mae }
def log_per_station_mae_to_wand(key: str, per_station_values: {}):  # {station:mae}
    for station, station_mae in per_station_values.items():
        wandb.log({key: station_mae, 'station': station})
