import wandb
from src.configurations import Configuration


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

    print("Do something clever here")

    return wandb
