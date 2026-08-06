import models

MODELS = {'llm': models.TransformerDecoder}

import datasetss

DATASETS = {
    'pretraining': datasetss.PretrainingDataset,
}

import trainers

TRAINERS = {'lm_trainer': trainers.LMTrainer}

import evaluators

EVALUATORS = {
    'pretraining': evaluators.PretrainingEvaluator,
}
