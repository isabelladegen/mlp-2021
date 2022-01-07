from hamcrest import *

from src.Data import PretrainedLinearModels, Data
from src.configurations import TestConfiguration
from src.models.BestPreTrainedModelForAStation import BestPreTrainedModelForAStation


def test_calculates_index_and_mae_for_the_best_pretrained_model_for_the_station():
    configuration = TestConfiguration()
    training_data = "../data/Train/station_201_deploy.csv"
    data = Data(configuration.no_nan_in_bikes, training_data)
    trained_models = PretrainedLinearModels(configuration.pretrained_models_path)

    model = BestPreTrainedModelForAStation(configuration, data, trained_models)
    model.fit()

    assert_that(model.best_model_index, equal_to(22))
    assert_that(model.best_mae, equal_to(2.7083333333333335))


def test_uses_best_pretrained_model_to_make_predictions_for_test_data():
    configuration = TestConfiguration()
    training_data_path = "../data/Train/station_201_deploy.csv"
    val_data_path = "../data/Dev/validation.csv"
    val_data_for_201 = Data(configuration.no_nan_in_bikes, val_data_path).get_data_per_station()[201]
    training_data = Data(configuration.no_nan_in_bikes, training_data_path)
    trained_models = PretrainedLinearModels(configuration.pretrained_models_path)

    model = BestPreTrainedModelForAStation(configuration, training_data, trained_models)
    model.fit()

    result = model.predict(val_data_for_201)

    assert_that(result.mean_absolute_error(), equal_to(2.15625))
