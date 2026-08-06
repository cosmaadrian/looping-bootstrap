import models

MODELS = {'llm': models.TransformerDecoder}

import datasetss

DATASETS = {
    'pretraining': datasetss.PretrainingDataset,
}

import trainers

TRAINERS = {
    'lm_trainer': trainers.LMTrainer,
    'recurrent_distillation_trainer': trainers.RecurrentDistillationTrainer,
}

import evaluators

EVALUATORS = {
    'pretraining': evaluators.PretrainingEvaluator,
    'recurrent_distillation': evaluators.RecurrentDistillationEvaluator,
}
