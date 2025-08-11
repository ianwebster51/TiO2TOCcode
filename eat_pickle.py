import dill
import matplotlib.pyplot as plt
import os
pwd = os.environ.get("PWD")
path = f"{pwd}/coupled_mode_figs/"
filenames = os.listdir(path)
files_sorted = sorted(filenames, key=lambda f: os.path.getmtime(os.path.join(path, f)))
with open(f"{path}{files_sorted[-1]}", "rb") as f:
    figs=dill.load(f)

for bob in figs:
    plt.show()

