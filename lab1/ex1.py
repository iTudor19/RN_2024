from cmath import sqrt
import pathlib

def parse_equation(equation: str) -> tuple[list[float], float]:
    coefficients = [0, 0, 0]
    equation = equation.replace(" ", "")

    lhs, rhs = equation.split('=')
    
    terms = lhs.replace('-', '+-').split('+')
    for term in terms:
        if term == '':
            continue
        elif 'x' in term:
            coefficient = term.replace('x', '')
            coefficients[0] = float(coefficient if coefficient not in ['-', ''] else (1 if coefficient == '' else -1))
        elif 'y' in term:
            coefficient = term.replace('y', '')
            coefficients[1] = float(coefficient if coefficient not in ['-', ''] else (1 if coefficient == '' else -1))
        elif 'z' in term:
            coefficient = term.replace('z', '')
            coefficients[2] = float(coefficient if coefficient not in ['-', ''] else (1 if coefficient == '' else -1))
    
    free_term = float(rhs)
    
    return coefficients, free_term

def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    matrix_A = []
    vector_B = []
    
    with path.open('r') as file:
        for line in file:
            line = line.strip()
            if line:
                coefficients, free_term = parse_equation(line)
                matrix_A.append(coefficients)
                vector_B.append(free_term)
    
    return matrix_A, vector_B

def determinant(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
        matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
        matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )

def trace(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] + matrix[1][1] + matrix[2][2]
    )

def norm(matrix: list[list[float]]) -> float:
    return (
        sqrt(matrix[0] ** 2 + matrix[1] ** 2 + matrix[2] ** 2)
    )

def multiply_matrix_vector(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = [0, 0, 0]
    
    for i in range(3):
        result[i] = (
            matrix[i][0] * vector[0] + 
            matrix[i][1] * vector[1] + 
            matrix[i][2] * vector[2]
        )
    
    return result

def cramer_solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_A = determinant(matrix)
    
    if det_A == 0:
        raise ValueError("nu se poate rezolva sistemul de ecuatii")
    
    solutions = []
    
    for i in range(3):
        modified_matrix = [row[:] for row in matrix]
        for j in range(3):
            modified_matrix[j][i] = vector[j]
        
        det_modified = determinant(modified_matrix)
        
        solutions.append(det_modified / det_A)
    
    return solutions

def transpose_matrix(matrix: list[list[float]]) -> list[list[float]]:
    transposed = [[0.0] * 3 for _ in range(3)]
    
    for i in range(3):
        for j in range(3):
            transposed[j][i] = matrix[i][j]
    
    return transposed

def determinant(matrix: list[list[float]]) -> float:
    return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
            matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
            matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))

def inverse(matrix: list[list[float]]) -> list[list[float]]:
    det = determinant(matrix)
    if det == 0:
        raise ValueError("matricea nu este inversabila")

    adjugate = [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]),
            -(matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]),
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]),
        ],
        [
            -(matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1]),
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]),
            -(matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0]),
        ],
        [
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]),
            -(matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0]),
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]),
        ],
    ]

    return [[elem / det for elem in row] for row in adjugate]

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    inv_matrix = inverse(matrix)
    
    solution = multiply_matrix_vector(transpose_matrix(inv_matrix), vector)
    
    return solution

path = pathlib.Path("C:/Users/Tudor/Documents/RN_2024/lab1/ecuatie.txt")
A, B = load_system(path)
print(f"A = {A}")
print(f"B = {B}")
print(f"Determinant of A = {determinant(A)}")
print(f"Trace of A = {trace(A)}")
print(f"Norm of B = {norm(B)}")
print(f"A * B = {multiply_matrix_vector(A, B)}")
print(f"Solutions of the system: {cramer_solve(A, B)}")
print(f"Transpose of A: {transpose_matrix(A)}")
print(f"Solutions with inverse: {solve(A, B)}")