import matplotlib.pyplot as plt
import numpy as np


def plot_confusion(cm: np.ndarray, title: str = "Confusion"):
    plt.imshow(cm, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

