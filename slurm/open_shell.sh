#!/bin/bash

srun --partition=dgxa100 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
