# Machine Learning Paradigms Course Work 2021

## Project Structure

TODO

## Development

### Conda environment

**Create**

    conda env create -f conda.yml
    conda activate mlp-2021

**Update**

    conda env update -n mlp-2021 --file conda.yml --prune

### Configuration and Experiment Tracking

Keeping a record of all the runs and configuration using Weights & Biases [MLP-2021](https://wandb.ai/idegen/mlp-2021).

Configurations are kept in a Python Dataclass `configurations.py`. They are used for the `wandb.config` and 
are therefore automatically logged.

### Testing

Using Pytest and Hamcrest for nicer assertion matcher syntax. Learning from the NLP coursework
this project was test driven from the get go.
