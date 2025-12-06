import matplotlib.pyplot as plt


# -----------------------------
# Utility plotting helpers
# -----------------------------

def plot_time_series(t, series, labels, title, ylabel, filename=None):
    plt.figure(figsize=(8,4))
    for s, lab in zip(series, labels):
        plt.plot(t, s, label=lab)
    plt.xlabel('Time step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if filename:
        plt.tight_layout()
        plt.savefig(filename, transparent=True)
    else:
        plt.show()