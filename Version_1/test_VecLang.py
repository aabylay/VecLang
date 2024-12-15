from VecLang import Lexer, Parser, Interpreter, Vector, Matrix

# Pre-setup code that defines basic vectors and matrices
pre_setup_code = """
v1 = vector([1, 2, 3])
v2 = vector([4, 5, 6])
m1 = matrix([[1,2,3],[4,5,6],[7,8,9]])
m2 = matrix([[1,1,1],[2,2,2],[9,9,9]])
"""

# Define test input code snippets
tests = {
    "basic_vectors": """
    # Define some vectors and perform operations
    v3 = v1 + v2
    v4 = v1 * 2
    """,

    "matrix_operations": """
    # Check matrix-vector multiplication
    res = m1 @ v1
    """,

    "similarity_operations": """
    # Using previously defined v1 and v2
    cos_sim = similarity(v1, v2, method="cosine")
    eu_dist = similarity(v1, v2, method="euclidean")
    """,

    "knn_single": """
    # KNN single vector
    knn_res = knn_bf(v1, m1, 2, method="euclidean")
    knn_res_cos = knn_bf(v1, m1, 2, method="cosine")
    """,

    "knn_batch": """
    # KNN batch
    min_sum_eu, mapping_eu = knn_bf_batch(m1, m2, method="euclidean")
    min_sum_cos, mapping_cos = knn_bf_batch(m1, m2, method="cosine")
    """,

    "dot_product": """
    # Dot product test
    dp = v1 @ v2
    """
}


def run_code(code):
    lexer = Lexer(code)
    tokens = list(lexer)
    parser = Parser(tokens)
    ast = parser.parse()
    return ast

def run_test(name, code):
    print("=================================")
    print(f"TEST: {name}")
    print("=================================\n")
    print("Code:\n", code.strip(), "\n")

    # Create a new interpreter and run the pre-setup code first
    interpreter = Interpreter()
    try:
        pre_ast = run_code(pre_setup_code)
        for node in pre_ast:
            interpreter.evaluate(node)
    except Exception as e:
        print("Error in pre-setup code:", e)
        return

    try:
        # Now run the actual test code
        ast = run_code(code)
        print("----- TOKENS & AST -----")
        # Re-lex and parse for displaying tokens and AST (already done above, but to show explicitly):
        lexer_for_display = Lexer(code)
        tokens = list(lexer_for_display)
        print("Tokens:", tokens)
        print("AST:", ast)

        # Evaluate AST
        for node in ast:
            interpreter.evaluate(node)

        # Print environment after execution
        print("\n----- ENVIRONMENT AFTER TEST -----")
        for k, v in interpreter.environment.items():
            # Print variable names and their values
            print(f"{k}: {v}")

    except Exception as e:
        print("Error during test:", e)

    print("\n\n")
    
    
if __name__ == "__main__":
    for test_name, test_code in tests.items():
        run_test(test_name, test_code)
    
    # Now prompt the user for custom input
    print("=================================")
    print("USER INPUT TEST")
    print("=================================\n")
    print("Enter your code snippet line by line. Press Enter on an empty line to finish:")

    user_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        user_lines.append(line)
    user_code = "\n".join(user_lines)

    if user_code.strip():
        # Run user code with the same setup
        interpreter = Interpreter()
        try:
            # Pre-setup code to define v1, v2, m1, m2
            pre_ast = run_code(pre_setup_code)
            for node in pre_ast:
                interpreter.evaluate(node)
        except Exception as e:
            print("Error in pre-setup code:", e)
        else:
            try:
                ast = run_code(user_code)
                print("----- USER TOKENS & AST -----")
                lexer_for_display = Lexer(user_code)
                tokens = list(lexer_for_display)
                print("Tokens:", tokens)
                print("AST:", ast)

                for node in ast:
                    interpreter.evaluate(node)

                print("\n----- ENVIRONMENT AFTER USER CODE -----")
                for k, v in interpreter.environment.items():
                    print(f"{k}: {v}")

            except Exception as e:
                print("Error during user test:", e)
    else:
        print("No user code provided.")
