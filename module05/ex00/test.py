from matrix import Matrix, Vector

A = Matrix(
    [
        [1, 2],
        [3, 4]
    ]
)

B = A

C = Matrix(
    [
        [0, 1],
        [1, 1]
    ]
)

res1 = A + A - C
# res2 = B + C
print(f"{res1.data}, shape: {res1.shape}")
# print(f"{res2.data}, shape: {res2.shape}")
print(A/2)
print(2/A)
print(A * 2)
print(A * C)
print(repr(A))
print(A.T())
v1 = Vector(
    [1, 2, 3]
)
print(v1)
v2 = Vector(
    [
        [1],
        [2],
        [3]
    ]
)
print(repr(v2))
print(f"v2.T(): {v2.T()}")
print(f"v1.v2 = {v1.dot(v2)}")
print(f"v2.v1 = {v2.dot(v1)}")
E = Matrix(
    [1, 2, 3]
)
print(repr(E))

F = Matrix(
    [
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50]
    ]
)
print(repr(F))
v3 = v2 + v2
print(f"type(v3): {type(v3), {repr(v3)}}")
v4 = v1 + v1
print(f"type(v4): {type(v4), {repr(v4)}}")

v10 = Matrix(
    [10, 20, 30]
)
v20 = Matrix(
    [
        [10],
        [20],
        [30]
    ]
)

v5 = v10 - v1
print(f"type(v5): {type(v5), {repr(v5)}}")
v6 = v20 - v2
print(f"type(v6): {type(v6), {repr(v6)}}")

print(f"v1 / 2 = {v1 / 2}")
print(f"v2 / 2 = {v2 / 2}")
print(f"2 / v1 = {2 / v1}")
print(f"2 / v2 = {2 / v2}")

print(f"T(v1): {v1.T()}")
print(f"T(v2): {v2.T()}")
print(f"T(T(v1)): {v1.T().T()}")
print(f"T(F): {F.T()}")
print(f"T(T(F)): {F.T().T()}")

G = Matrix(
    [1, 2, 3]
)

H = Matrix(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)

I = G * H
print(f"G * H = {I}")
J = Matrix(
    [
        [1],
        [2],
        [3]
    ]
)
K = H * J
print(f"H * J = {K}")
L = G * J
print(f"G * J = {L}")
M = J * G
print(f"J * G = {M}")

print(f"2 * J = { 2 * J}")
print(f"J * 2 = { J * 2}")

print(f"2 * G = { 2 * G}")
print(f"G * 2 = { G * 2}")
