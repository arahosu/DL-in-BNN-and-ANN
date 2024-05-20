import json
import matplotlib.pyplot as plt
import numpy as np

def compute_results(result_dict):
    all_results = []

    for i, experience in enumerate(result_dict):
        test_result = np.array([value for key, value in experience.items() if 'Top1_Acc_Stream/eval_phase/test_stream/' in key])
        all_results.append(test_result[:i+1].mean())

    return all_results

# with open("eval_results_ewc_sgd.json", "r") as fp:
#     ewc_sgd_dict = json.load(fp)
# with open("eval_results_ewc_adam.json", "r") as fp:
#     ewc_adam_dict = json.load(fp)
# with open("eval_results_sgd.json", "r") as fp:
#     sgd_dict = json.load(fp)
with open("eval_results_adam.json", "r") as fp:
    adam_dict = json.load(fp)
with open("eval_results_si.json", "r") as fp:
    si_dict = json.load(fp)

# ewc_sgd_results = compute_results(ewc_sgd_dict)
# ewc_adam_results = compute_results(ewc_adam_dict)
# sgd_results = compute_results(sgd_dict)
adam_results = compute_results(adam_dict)
si_results = compute_results(si_dict)

print(adam_results[-1], si_results[-1])

# plt.plot(ewc_sgd_results, label='EWC + SGD')
# plt.plot(ewc_adam_results, label='EWC + Adam')
# plt.plot(sgd_results, label='Naive + SGD')
x = [i for i in range(1, len(adam_results) + 1)]
plt.plot(x, adam_results, label='Naive')
plt.plot(x, si_results, label='Synaptic Intelligence')
plt.xlabel('Number of tasks')
plt.ylabel('Fraction correct')
plt.legend()
plt.show()