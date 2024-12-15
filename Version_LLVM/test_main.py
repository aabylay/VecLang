# test_main.py
import sys
from main import compile_vec_code
from lexer import Lexer
from parser import Parser
from ast import ASTNode  # assuming you have ast.py with ASTNode
# from compiler import LLVMCompiler # If needed directly
# We assume main.py and other modules are in the same directory

# Define a set of test inputs that cover a wide range of functionality.
tests = {
    "basic_vectors": """
    # Define vectors and operations
    v1 = vector([1, 2, 3])
    v2 = vector([4, 5, 6])
    v3 = v1 + v2
    v4 = v1 * 2
    """,

    "matrix_operations": """
    # Define matrices and do matrix-vector multiplication
    v1 = vector([1, 2, 3]) 
    m1 = matrix([[1,2,3],[4,5,6],[7,8,9]])
    res = m1 @ v1
    """,

    "similarity_operations": """
    # Similarity tests
    v1 = vector([1, 2, 3])
    v2 = vector([4, 5, 6])
    cos_sim = similarity(v1, v2, method="cosine")
    eu_dist = similarity(v1, v2, method="euclidean")
    """,

    "knn_single": """
    # Single vector KNN
    v1 = vector([1,2,3])
    m1 = matrix([[1,2,3],[4,5,6],[7,8,9]])
    knn_res = knn_bf(v1, m1, 2, method="euclidean")
    knn_res_cos = knn_bf(v1, m1, 2, method="cosine")
    """,

    "knn_batch": """
    # Batch KNN
    m1 = matrix([[1,2,3],[4,5,6],[7,8,9]])
    m2 = matrix([[1,1,1],[2,2,2],[9,9,9]])
    min_sum_eu, mapping_eu = knn_bf_batch(m1, m2, method="euclidean")
    min_sum_cos, mapping_cos = knn_bf_batch(m1, m2, method="cosine")
    """,

    "dot_product": """
    # Dot product test
    v1 = vector([1,2,3])
    v2 = vector([4,5,6])
    dp = v1 @ v2
    """
}

def run_test(name, code):
    print("=================================")
    print(f"TEST: {name}")
    print("=================================")
    print("Code:\n", code.strip(), "\n")

    # Lexing
    lexer = Lexer(code)
    tokens = list(lexer)
    print("----- TOKENS -----")
    print(tokens)

    # Parsing
    parser = Parser(tokens)
    ast = parser.parse()
    print("----- AST -----")
    for node in ast:
        print(node)

    # Compiling to LLVM IR
    llvm_ir = compile_vec_code(code)
    print("----- LLVM IR -----")
    print(llvm_ir)
    print("\n\n")


if __name__ == "__main__":
    # Run all predefined tests
    for test_name, test_code in tests.items():
        run_test(test_name, test_code)

    # Optionally, allow user input for additional testing:
    print("=================================")
    print("USER INPUT TEST")
    print("=================================")
    print("Enter VecLang code. Press Enter on an empty line to finish:")

    user_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        user_lines.append(line)
    user_code = "\n".join(user_lines)

    if user_code.strip():
        run_test("user_input", user_code)
    else:
        print("No user code provided.")
