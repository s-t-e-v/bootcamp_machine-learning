class Matrix:

    def __init__(self, data, shape):
        self.data = data
        self.shape = shape

    @classmethod
    def from_data(cls, data):
        """Initialize the maxtrix from a list of lists"""
        # TODO - add a check for data
        shape = (len(data), len(data[0]))
        return cls(data, shape)

    @classmethod
    def from_shape(cls, shape):
        """Initialize the matrix from shape with zeroes"""
        # TODO - add a check for shape
        data = [[0 for j in range(shape[0])] for i in range(shape[1])]
        return cls(data, shape)

    # add : only matrices of same dimensions.
    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise NotImplemented
        if self.shape != other.shape:
            raise NotImplemented
        return Matrix.from_data(
            [[val_a + val_b for val_a, val_b in zip(row_a, row_b)] for row_a, row_b in zip(self.data, other.data)]
        )
        
    # __radd__ = __add__

    # sub : only matrices of same dimensions.
    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise NotImplemented
        if self.shape != other.shape:
            raise NotImplemented
        return Matrix.from_data(
            [[val_a - val_b for val_a, val_b in zip(row_a, row_b)] for row_a, row_b in zip(self.data, other.data)]
        )

    __rsub__ = __sub__
    
    # div : only scalars.
    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise NotImplemented
        if scalar == 0:
            raise ZeroDivisionError
        return Matrix.from_data(
            [[v / scalar for v in row] for row in self.data]
        )
    
    def __rtruediv__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise NotImplemented
        return Matrix.from_data(
            [[scalar / v for v in row] for row in self.data]
        )


    # mul : scalars, vectors and matrices , can have errors with vectors and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Matrix.from_data(
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
            return Matrix.from_data(data)
        raise NotImplemented

        

    # __rmul__
    # __str__
    # __repr__

class Vector(Matrix):
    def __init__(self, data, shape):
        super().__init__(data, shape)