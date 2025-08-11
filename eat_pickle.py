import dill
import matplotlib.pyplot as plt
import os
pwd = os.environ.get("PWD")
path = f"{pwd}/coupled_mode_figs/"
filenames = os.listdir(path)
files_sorted = sorted(filenames, key=lambda f: os.path.getmtime(os.path.join(path, f)))
with open(f"{path}{files_sorted[-1]}", "rb") as f:
    figs=dill.load(f)

for fig, axs in figs:
    for ax in axs:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 1)     
plt.show()

