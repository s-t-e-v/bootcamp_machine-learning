from TinyStatistician import TinyStatistician
import numpy as np

x1 = np.array([1, 2, 3, 4, 5])

x1_bar = TinyStatistician.mean(x1)
print(f"x1 = {x1}, x1_bar = {x1_bar}")