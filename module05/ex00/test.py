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
print(v1 + v1)