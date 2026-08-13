import models

MODELS = {'llm': models.TransformerDecoder}

import datasetss

DATASETS = {
    'deepmind_mathematics': datasetss.DeepMindMathematicsDataset,
    'pretraining': datasetss.PretrainingDataset,
}

import trainers

TRAINERS = {
    'lm_trainer': trainers.LMTrainer,
    'recurrent_distillation_trainer': trainers.RecurrentDistillationTrainer,
}

import evaluators

EVALUATORS = {
    'deepmind_mathematics': evaluators.DeepMindMathematicsEvaluator,
    'pretraining': evaluators.PretrainingEvaluator,
    'recurrent_distillation': evaluators.RecurrentDistillationEvaluator,
}
