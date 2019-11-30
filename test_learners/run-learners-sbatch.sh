#!/bin/bash

sbatch run-OneSlackSSVM.sh
sbatch run-PrimalDSStructuredSVM.sh
sbatch run-SubgradientSSVM.sh
sbatch run-FrankWolfeSSVM.sh
sbatch run-NSlackSSVM.sh
sbatch run-StructuredPerceptron.sh
sbatch run-LatentSSVM.sh
sbatch run-SubgradientLatentSSVM.sh

