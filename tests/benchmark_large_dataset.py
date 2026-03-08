import matplotlib.pyplot as plt
import numpy as np

methods = ["LazyTune", "GridSearchCV", "RandomizedSearchCV", "Optuna", "Hyperopt"]

accuracy = [
    (0.9649 + 1.0) / 2,
    (0.9561 + 1.0) / 2,
    (0.9561 + 1.0) / 2,
    (0.9561 + 1.0) / 2,
    (0.967 + 0.9862) / 2
]

runtime = [
    (161.28 + 83.26) / 2,
    (197.18 + 89.75) / 2,
    (33.93 + 15.15) / 2,
    (88.73 + 63.76) / 2,
    (128.44 + 44.68) / 2
]

x = np.arange(len(methods))

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Accuracy Plot
axes[0].bar(methods, accuracy)
axes[0].set_title("Average Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0.95, 1.0)

for i, v in enumerate(accuracy):
    axes[0].text(i, v + 0.002, f"{v:.3f}", ha='center')

# Runtime Plot
axes[1].bar(methods, runtime)
axes[1].set_title("Average Runtime")
axes[1].set_ylabel("Seconds")

for i, v in enumerate(runtime):
    axes[1].text(i, v + 2, f"{v:.1f}s", ha='center')

plt.suptitle("Hyperparameter Optimization Benchmark", fontsize=14)

plt.tight_layout()
plt.show()
