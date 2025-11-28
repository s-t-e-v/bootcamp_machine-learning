from TinyStatistician import TinyStatistician
import numpy as np

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ])

x1_bar = TinyStatistician.mean(x1)
print(f"x1 = {x1}, x1_bar = {x1_bar}")

x2_bar = TinyStatistician.mean(x2)
print(f"x2 = {x2}, x2_bar = {x2_bar}")
print(TinyStatistician.mean(np.array([[10]])))

print(TinyStatistician.median(x1))

x3 = np.array([2, 3, 2, 5, 9, 1])

x4 = np.array(
    [4, 5, 1, 0, 10]
)

print(f"median(x3) = {TinyStatistician.median(x3)}")
print(f"median(x4) = {TinyStatistician.median(x4)}")
