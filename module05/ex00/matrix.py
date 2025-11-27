import sys


class Matrix:
    def __init__(self, arg):
        if isinstance(arg, (int, float)):
            self.shape = arg
            self.data = [
                [0 for j in range(self.shape[0])]
                for i in range(self.shape[1])
                ]
            return
        if isinstance(arg, list):
            if all(isinstance(elem, (int, float)) for elem in arg):
                m = 1
                n = len(arg)
            elif all(isinstance(elem, list) and len(elem) == len(arg[0])
                     for elem in arg):
                m = len(arg)
                n = len(arg[0])
            else:
                print(f"Error: Invalid matrix data: {arg}", file=sys.stderr)
                raise NotImplementedError
            self.shape = (m, n)
            self.data = arg
            return
        raise NotImplementedError

    # add : only matrices of same dimensions.
    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise NotImplementedError
        if self.shape != other.shape:
            raise NotImplementedError
        return Matrix([
            [val_a + val_b for val_a, val_b in zip(row_a, row_b)]
            if isinstance(row_a, list) and isinstance(row_b, list)
            else row_a + row_a
            for row_a, row_b in zip(self.data, other.data)
            ])

    __radd__ = __add__

    # sub : only matrices of same dimensions.
    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise NotImplementedError
        if self.shape != other.shape:
            raise NotImplementedError
        return Matrix([
            [val_a - val_b for val_a, val_b in zip(row_a, row_b)]
            if isinstance(row_a, list) and isinstance(row_b, list)
            else row_a - row_b
            for row_a, row_b in zip(self.data, other.data)
            ])

    __rsub__ = __sub__

    # div : only scalars.
    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise NotImplementedError
        if scalar == 0:
            raise ZeroDivisionError
        return Matrix([
            [v / scalar for v in row]
            if isinstance(row, list)
            else row / scalar
            for row in self.data
            ])

    def __rtruediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise NotImplementedError
        return Matrix([
            [scalar / v for v in row]
            if isinstance(row, list)
            else scalar / row
            for row in self.data
            ])

    # mul : scalars, vectors and matrices , can have errors with vectors
    # and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            data = [
                [v * other for v in row]
                if isinstance(row, list)
                else row * other
                for row in self.data
                ]
        elif isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ArithmeticError
            data = []
            for m in range(self.shape[0]):
                row = self.data if self.shape[0] == 1 else self.data[m]
                new_row = []
                for p in range(other.shape[1]):
                    new_val = sum(
                        val * other.data[n][p]
                        if other.shape[0] > 1 else val * other.data[p]
                        for val, n in zip(row, range(other.shape[0])))
                    new_row.append(new_val)
                if self.shape[0] > 1:
                    data.append(new_row)
                else:
                    data = new_row
                    break
        else:
            raise NotImplementedError
        if (Vector.one_dimension(data) and
                (isinstance(self, Vector) or isinstance(other, Vector))):
            return Vector(data)
        return Matrix(data)

    __rmul__ = __mul__

    def T(self):
        if self.shape[1] == 1:
            return Matrix([v[0] for v in self.data])
        if self.shape[0] == 1:
            return Matrix([[v] for v in self.data])
        return Matrix(
            [[row[i] for row in self.data] for i in range(self.shape[1])]
            )

    def __str__(self):
        return f"{self.data}"

    def __repr__(self):
        return f"Matrix(data={self.data}, shape={self.shape})"


class Vector(Matrix):
    def __init__(self, arg):
        if not isinstance(arg, list):
            raise NotImplementedError
        if not Vector.one_dimension(arg):
            raise NotImplementedError
        super().__init__(arg)

    def dot(self, v):
        if not isinstance(v, Vector):
            raise NotImplementedError
        if self.shape[1] != v.shape[0]:
            raise NotImplementedError
        if self.shape[0] == 1:
            return sum(va * vb for va, vb in zip(self.data, v.T().data))
        return sum(va * vb for va, vb in zip(self.T().data, v.data))

    @staticmethod
    def one_dimension(data):
        return all(isinstance(v, (int, float)) for v in data) or\
                all(isinstance(elem, list) and len(elem) == 1 for elem in data)

    def __repr__(self):
        return f"Vector(data={self.data}, shape={self.shape})"
