import numpy as np
import json
import matplotlib.pyplot as plt

with open("eval_results_ewc.json", "r") as ewc:
    ewc_results = json.load(ewc)

with open("eval_results_naive.json", "r") as naive:
    naive_results = json.load(naive)
