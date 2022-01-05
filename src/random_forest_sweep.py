import wandb

from src.configurations import Configuration


def sweep():
    non_sweep_config = Configuration().as_dict()
    wandb.init(project=non_sweep_config.wandb_project_name,
               entity=non_sweep_config.wandb_entity,
               mode=non_sweep_config.wandb_mode,
               notes="feature testing, temperature and is holiday, avoiding overfitting",
               tags=['RandomForrest', 'one model', 'model per station'],
               config=non_sweep_config)

    sweeped_config = Configuration(**wandb.config)




if __name__ == '__main__':
    parameters_to_try = {
        'pre_process_rc_question': {
            'values': [QuestionPreProcessing.default.value, QuestionPreProcessing.user_question_only.value]
        },
        'vector_size': {
            'values': [100, 150, 200]
        },
        'epochs': {
            'values': [150, 200, 250]
        },
        'dm': {
            'values': [0, 1]
        },
        'number_of_most_likely_docs': {
            'values': [1, 3]
        }
    }

    sweep_config_grid = {
        'name': 'test',
        'method': 'grid',
        'parameters': parameters_to_try
    }

    sweep_config_bayes = {
        'name': 'test',
        'method': 'bayes',
        'metric': {
            'goal': 'maximize',
            'name': SCORE_F1
        },
        'parameters': parameters_to_try,
        # 'early_terminate': {
        #     'type': 'hyperband',
        #     's': 2,
        #     'eta': 3,
        #     'max_iter': 27
        # }
    }

    sweep_id = wandb.sweep(sweep_config_grid, project=config.wandb_project_name)
    wandb.agent(sweep_id, function=sweep)
