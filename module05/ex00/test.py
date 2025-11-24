from matrix import Matrix

A = Matrix.from_data(
    [
        [1, 2],
        [3, 4]
    ]
)

B = A

C = Matrix.from_data(
    [
        [0, 1],
        [1, 1]
    ]
)

res1 = A + A - C
# res2 = B + C
print(f"{res1.data}, shape: {res1.shape}")
# print(f"{res2.data}, shape: {res2.shape}")
print((A/2).data)
print((2/A).data)
print((A * 2).data)
print((A * C).data)