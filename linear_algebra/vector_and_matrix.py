"""Some foundamentals of vectors and matrices for rational numbers."""

from typing import Any, List

class Vector:
    """
    A class of vector.
    """
    def __init__(self, value: List[Union[int, float]]) -> None:
        """
        Initialize a vector.
        """
        self.dim = len(value)
        self.vector = value

    def __eq__(self, other: Any) -> bool:
        """
        Return whether self is equal to other.
        """
        return type(self) == type(other) and self.vector == other.vector

    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Return the sum of two vectors.
        """
        if self.dim != other.dim:
            raise AssertionError('two vectors have different dimensions')
        new_vector = []
        for i in range(len(self.vector)):
            new_vector.append(self.vector[i] + other.vector[i])
        return Vector(new_vector)

    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Return the difference between two vectors.
        """
        if self.dim != other.dim:
            raise AssertionError('two vectors have different dimensions')
        new_vector = []
        for i in range(len(self.vector)):
            new_vector.append(self.vector[i] - other.vector[i])
        return Vector(new_vector)

    def __repr__(self) -> str:
        """
        Return the visualization of vector self.
        """
        string = 'Vector('
        for num in self.vector[:-1]:
            string += '{}'.format(num)
            string += ','
        string += '{}'.format(self.vector[-1])
        return string + ')'

    def scale(self, n: Union[int, float]) -> 'Vector':
        """
        Return the product of self and n.
        """
        new_vector = []
        for i in range(len(self.vector)):
            new_vector.append(n * self.vector[i])
        return Vector(new_vector)

    def norm(self) -> Union[int, float]:
        """
        Return the norm of self.
        """
        value = 0
        for i in range(len(self.vector)):
            value += self.vector[i] ** 2
        return value ** 0.5

    def dot(self, other: 'Vector') -> Union[int, float]:
        """
        Return the dot product of two vectors.
        """
        if self.dim != other.dim:
            raise AssertionError('two vectors have different dimensions')
        value = 0
        for i in range(len(self.vector)):
            value += self.vector[i] * other.vector[i]
        return value

    def proj(self, other: Union['Subspace', 'Vector']) -> 'Vector':
        """
        Return the projection of self on other.
        """
        s = {other} if other is Vector else other.basis
        vector_sum = self.get_zero_vector(self.dim)
        for vector in s:
            if self.dim != vector.dim:
                raise AssertionError('two vectors have different dimensions')
            constant = self.dot(vector) / self.norm()
            vector_sum += vector.scale(constant)
        return vector_sum

    def unit_vector(self) -> 'Vector':
        """
        Return the unit vector of self.
        """
        return self.scale(1 / self.norm())

    def d(self, other: 'Vector') -> Union[int, float]:
        """
        Return the distance between self and other.
        """
        return (self - other).norm()

    def is_parallel(self, other: 'Vector') -> bool:
        """
        Return whether two vectors are parallel.
        """
        return self.scale(other.vector[0] / self.vector[0]) == other

    def is_orthogonal(self, other: 'Vector') -> bool:
        """
        Return whether two vectors are orthogonal.
        """
        return self.dot(other) == 0

    def is_linearly_independent(self, other: 'Vector') -> bool:
        """
        Return whether two vectors are linearly independent.
        """
        return not self.is_parallel(other)

    def get_zero_vector(self, n: int) -> 'Vector':
        """
        Return a zero vector with a size of n.
        """
        size = n
        vector = []
        while size != 0:
            vector.append(0)
            size -= 1
        return Vector(vector)


def get_linearly_independent_set(s: Set[Vector]) -> Set[Vector]:
    """
    Return the span of the set of vectors s in the form of (basis, dimension).
    """
    st = []
    for v in s:
        st.append(v)
    dct = {'basis': [], 'repeats': []}
    for i in range(len(st)):
        j = i + 1
        repeats = 0
        while j < len(st) and repeats < 3 and st[i] not in dct['repeats']:
            independent = st[i].is_linearly_indepent(st[j])
            if not independent:
                repeats += 1
                dct['repeats'].append(st[j])
        if repeats == 3:
            dct['repeats'].append(st[i])
        if i == len(st) - 1 and st[i] not in dct['repeats']:
            dct['basis'].append(st[i])
    return set(dct['basis'])


class Matrix:
    """
    A class of matrix.
    """
    def __init__(self, value: List[Vector]) -> None:
        """
        Initialize a matrix.
        """
        self.size = (len(value[0].dim), len(value))
        self.matrix = value
        self.pivots = self.get_pivots()
        self.zero_dict = {}

    def __eq__(self, other: Any) -> bool:
        """
        Return whether self is equal to other.
        """
        return type(self) == type(other) and self.matrix == other.matrix

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """
        Return the sum of two matrices.
        """
        if self.size != other.size:
            raise AssertionError('two matrices have different sizes')
        new_matrix = []
        for i in range(len(other.matrix)):
            new_matrix.append(self.matrix[i] + other.matrix[i])
        return Matrix(new_matrix)

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """
        Return the difference between two matrices.
        """
        if self.size != other.size:
            raise AssertionError('two matrices have different sizes')
        new_matrix = []
        for i in range(len(other.matrix)):
            new_matrix.append(self.matrix[i] - other.matrix[i])
        return Matrix(new_matrix)

    def __mul__(self, other: 'Matrix') -> 'Matrix':
        """
        Return the product of two matrices.
        """
        if self.size[1] != other.size[0]:
            raise AssertionError('the column size of first matrix and the row '
                                 'size of second matrix are different')
        transpose = self.get_transpose()
        new_matrix = []
        for v in other.matrix:
            vector = []
            for u in transpose.matrix:
                vector.append(v.dot(u))
            new_matrix.append(Vector(vector))
        return Matrix(new_matrix)

    def __repr__(self) -> str:
        """
        Return the visualization of matrix self.
        """
        string = 'Matrix('
        for vector in self.matrix[:-1]:
            string += '{}'.format(repr(vector))
            string += ','
        string += '{}'.format(repr(self.matrix[-1]))
        return string + ')'

    def get_transpose(self) -> 'Matrix':
        """
        Return the transpose of self.
        """
        transpose = []
        for i in range(self.size[0]):
            vector = []
            for j in range(self.size[1]):
                vector.append(self.matrix[j].vector[i])
            transpose.append(Vector(vector))
        return Matrix(transpose)

    def get_identy_matrix(self, n: int) -> 'Matrix':
        """
        Return the identy matrix with a size of n x n.
        """
        matrix = []
        for i in range(n):
            vector = []
            j = 0
            while j != i - 1:
                vector.append(0)
            vector.append(1)
            j += 1
            while j != n - 1:
                vector.append(0)
            matrix.append(Vector(vector))
        return Matrix(matrix)

    def get_inverse(self) -> 'Matrix':
        """
        Return the inverse of self.
        """
        if not self.is_invertable():
            raise AssertionError('self is not invertable')
        matrix = self.matrix[:]
        matrix.extend(self.get_identy_matrix(self.size[0]).matrix)
        a = Matrix(matrix).get_rref()
        inverse = a.matrix[len(a.matrix) / 2:]
        return Matrix(inverse)

    def get_rref(self) -> 'Matrix':
        """
        Return self in RREF
        """
        ref = self.get_ref()
        ref.update_zero_dict()
        transpose = ref.get_transpose()
        source = [i for i in ref.zero_dict]
        source.sort()
        while source != []:
            current_zero = source.pop()
            if current_zero != ref.size[1]:
                row_ind = sum(ref.zero_dict[current_zero], [])
                current_row = transpose.matrix[row_ind]
                for i in range(row_ind):
                    c = - transpose.matrix[i].vector[current_zero]
                    replaced = transpose.matrix[i] + current_row.scale(c)
                    transpose.matrix[i] = replaced
        return transpose.get_transpose()

    def get_ref(self, row_ind=0) -> 'Matrix':
        """
        Return self in REF..
        """
        ordered = self.get_in_order()
        transpose = ordered.get_transpose()
        i = 0
        current_row = transpose.matrix[row_ind]
        while current_row.vector[i] == 0:
            i += 1
        scaled = current_row.scale(1 / current_row.vector[i])
        current_row = scaled
        j = row_ind + 1
        while transpose.matrix[j].vector[i] != 0:
            c = - transpose.matrix[j].vector[i]
            replaced = transpose.matrix[j] + current_row.scale(c)
            transpose.matrix[j] = replaced
        if transpose.get_transpose().is_ref():
            return transpose.get_transpose()
        else:
            return transpose.get_transpose().get_ref(row_ind + 1)

    def get_in_order(self) -> 'Matrix':
        """
        Return a matrix in a specific order that a row starting with less zeros
        is in front of the one starting with more zeros.
        """
        self.update_zero_dict()
        transpose = self.get_transpose()
        new_transpose_matrix = []
        source = [i for i in self.zero_dict]
        source.sort()
        while source != []:
            s = source.pop(0)
            for v in self.zero_dict[s]:
                new_transpose_matrix.append(transpose.matrix[v])
        new_transpose = Matrix(new_transpose_matrix)
        return new_transpose.get_transpose()

    def update_zero_dict(self) -> None:
        """
        Update self.zero_dict.
        """
        transpose = self.get_transpose()
        for v in transpose.matrix:
            i = 0
            count_zero = 0
            while v.vector[i] == 0:
                count_zero += 1
                i += 1
            if count_zero not in self.zero_dict:
                self.zero_dict[count_zero] = [transpose.matrix.index(v)]
            else:
                self.zero_dict[count_zero].append(transpose.matrix.index(v))

    def get_pivots(self, row_ind=0) -> list:
        """
        Return a list of entries of pivots.
        """
        pivots = []
        ordered = self.get_in_order()
        transpose = ordered.get_transpose()
        i = 0
        current_row = transpose.matrix[row_ind]
        while current_row.vector[i] == 0:
            i += 1
        pivots.append((row_ind + 1, i + 1))
        scaled = current_row.scale(1 / current_row.vector[i])
        current_row = scaled
        j = row_ind + 1
        while transpose.matrix[j].vector[i] != 0:
            c = - transpose.matrix[j].vector[i]
            repaced = transpose.matrix[j] - current_row.scale(c)
            transpose.matrix[j] = repaced
        if transpose.get_transpose().is_ref():
            return pivots
        else:
            return pivots + transpose.get_transpose().get_pivots(row_ind + 1)

    def is_ref(self) -> bool:
        """
        Return whether self is in REF.
        """
        transpose = self.get_transpose()
        previous_zero = -1
        for v in transpose.matrix:
            i = 0
            zero = 0
            while v.vector[i] == 0:
                zero += 1
                i += 1
            if previous_zero > zero or (previous_zero == zero
                                        != transpose.size[0]) \
                    or (i in range(transpose.size[0]) and v.vector[i] != 1):
                return False
            previous_zero = zero
        return True

    def is_rref(self) -> bool:
        """
        Return whether self is in RREF.
        """
        self.pivots = self.get_pivots()
        if not self.is_ref():
            return False
        for pivot in self.pivots:
            for i in range(self.size[0]):
                if self.matrix[pivot[1] - 1][i] != 0 and i != pivot[0]:
                    return False
        return True

    def is_invertable(self) -> bool:
        """
        Return whether self is invertable.
        """
        return self.get_rref() == self.get_identy_matrix(self.size[0])

    def is_symmetric(self) -> bool:
        """
        Return whether self is symmetric.
        """
        return self == self.get_transpose()
