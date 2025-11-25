import sys

class Matrix:

    def __init__(self, arg):
        if isinstance(arg, (int, float)):
            self.shape = arg
            self.data = [[0 for j in range(self.shape[0])] for i in range(self.shape[1])]
            return
        if isinstance(arg, list):
            if all(isinstance(elem, int) for elem in arg):
                m = 1
                n = len(arg)
            elif all(isinstance(elem, list) and len(elem) == len(arg[0]) for elem in arg):
                m = len(arg)
                n = len(arg[0])
            else:
                print(f"Error: Invalid matrix data: {arg}", file=sys.stderr)
                raise NotImplemented
            self.shape = (m, n)
            self.data = arg
            return
        raise NotImplemented
    
    # add : only matrices of same dimensions.
    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise NotImplemented
        if self.shape != other.shape:
            raise NotImplemented
        return Matrix(
            [[val_a + val_b for val_a, val_b in zip(row_a, row_b)] for row_a, row_b in zip(self.data, other.data)]
        )
        
    __radd__ = __add__

    # sub : only matrices of same dimensions.
    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise NotImplemented
        if self.shape != other.shape:
            raise NotImplemented
        return Matrix(
            [[val_a - val_b for val_a, val_b in zip(row_a, row_b)] for row_a, row_b in zip(self.data, other.data)]
        )

    __rsub__ = __sub__
    
    # div : only scalars.
    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise NotImplemented
        if scalar == 0:
            raise ZeroDivisionError
        return Matrix(
            [[v / scalar for v in row] for row in self.data]
        )
    
    def __rtruediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise NotImplemented
        return Matrix(
            [[scalar / v for v in row] for row in self.data]
        )


    # mul : scalars, vectors and matrices , can have errors with vectors and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Matrix(
                [[v * other for v in row] for row in self.data]
            )
        if isinstance(other, Vector):
            pass
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ArithmeticError
            data = []
            for row_a in self.data:
                new_row = []
                for j in range(other.shape[1]):
                    new_val = 0
                    for i, val_a in enumerate(row_a):
                        new_val += val_a * other.data[i][j]
                    new_row.append(new_val)
                data.append(new_row)
            return Matrix(data)
        raise NotImplemented

        

    # __rmul__

    def T(self):
        if self.shape[1] == 1:
            return Matrix([v[0] for v in self.data])
        return Matrix([[row[i] for row in self.data] for i in range(self.shape[1])])

    def __str__(self):
        return f"{self.data}"
    
    def __repr__(self):
        return f"Matrix(data={self.data}, shape={self.shape})"

class Vector(Matrix):
    def __init__(self, arg):
        if not isinstance(arg, list):
            raise NotImplemented
        if all(isinstance(v, int) for v in arg):
            super().__init__(arg)
            return
        elif all(isinstance(elem, list) and len(elem) == 1 for elem in arg):
            super().__init__(arg)
            return
        raise NotImplemented
            
    def dot(self, v):
        if not isinstance(v, Vector):
            raise NotImplemented
        if self.shape[1] != v.shape[0]:
            raise NotImplemented
        if self.shape[0] == 1:
            return sum(va * vb for va, vb in zip(self.data, v.T().data))
        return sum(va * vb for va, vb in zip(self.T().data, v.data))
    