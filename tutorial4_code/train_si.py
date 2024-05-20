from avalanche.benchmarks import PermutedMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import SynapticIntelligence

import torch
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss

import os
import json

# Set seed
torch.manual_seed(0)

# CPU / GPU DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_count = os.cpu_count()

benchmark = PermutedMNIST(10)

# MODEL CREATION
model = SimpleMLP(num_classes=benchmark.n_classes,
                  hidden_size=2000,
                  hidden_layers=2,
                  drop_rate=0.)

# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns 
# them to the strategy it is attached to.

# log to text file
text_logger = TextLogger(open('log_si.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True),
    loggers=[interactive_logger, text_logger],
    benchmark=benchmark
)

# CREATE THE STRATEGY INSTANCE (SI)
cl_strategy = SynapticIntelligence(
    model, Adam(model.parameters(), lr=0.001), CrossEntropyLoss(),
    si_lambda = 1, eps=0.1, train_epochs=20,
    eval_mb_size=128, train_mb_size=256, device=device, evaluator=eval_plugin)


# TRAINING LOOP
print('Starting experiment...')
results_ewc = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience, num_workers=cpu_count)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # eval also returns a dictionary which contains all the metric values
    results_ewc.append(cl_strategy.eval(benchmark.test_stream, num_workers=cpu_count))

# save results in .json file
with open("eval_results_si.json", "w") as fp:
    json.dump(results_ewc, fp)