import math
import re

########################################
# Data Structures: Vector and Matrix
########################################

class Vector:
    def __init__(self, data):
        if not isinstance(data, list) or not all(isinstance(x, (int, float)) for x in data):
            raise TypeError("Vector data must be a list of numbers.")
        self.data = data

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Addition is only supported between two vectors.")
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must be the same length for addition.")
        return Vector([x + y for x, y in zip(self.data, other.data)])

    def __mul__(self, scalar):
        # Scalar multiplication
        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar multiplication requires a number.")
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        if not isinstance(other, Vector):
            raise TypeError("Dot product is only supported between two vectors.")
        if len(self.data) != len(other.data):
            raise ValueError("Vectors must be the same length for dot product.")
        return sum(x * y for x, y in zip(self.data, other.data))

    def norm(self):
        return math.sqrt(sum(x**2 for x in self.data))

    def similarity(self, other, method="cosine"):
        if not isinstance(other, Vector):
            raise TypeError("Similarity is only supported between two vectors.")
        if method == "cosine":
            # Cosine similarity
            dot = self.dot(other)
            norm_product = self.norm() * other.norm()
            return dot / norm_product if norm_product != 0 else 0.0
        elif method == "euclidean":
            # Euclidean distance
            dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(self.data, other.data)))
            # If we use this in a "similarity" context, we are consistent that this returns a distance.
            return dist
        else:
            raise ValueError(f"Unknown method: {method}")


class Matrix:
    def __init__(self, data):
        if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
            raise TypeError("Matrix data must be a list of lists.")
        if len(set(len(row) for row in data)) > 1:
            raise ValueError("All rows in the matrix must have the same length.")
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0

    def __repr__(self):
        return f"Matrix({self.data})"

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar multiplication requires a number.")
        return Matrix([[x * scalar for x in row] for row in self.data])

    def dot(self, vector):
        if not isinstance(vector, Vector):
            raise TypeError("Matrix-vector multiplication requires a Vector.")
        if self.cols != len(vector.data):
            raise ValueError("Matrix columns must match vector length.")
        return Vector([sum(row[i] * vector.data[i] for i in range(self.cols)) for row in self.data])

    def get_row_vector(self, index):
        if index < 0 or index >= self.rows:
            raise IndexError("Index out of range for matrix rows.")
        return Vector(self.data[index])

########################################
# KNN Functions
########################################

def compute_distance(v1, v2, method):
    """
    Compute a 'distance-like' measure between v1 and v2.
    For euclidean: distance = euclidean distance.
    For cosine: distance = 1 - cosine_similarity.
    """
    if method == "cosine":
        sim = v1.similarity(v2, method="cosine")
        return 1.0 - sim
    elif method == "euclidean":
        dist = v1.similarity(v2, method="euclidean")
        return dist
    else:
        raise ValueError("Unsupported method.")

def knn_bf(v, M, k, method="cosine"):
    """
    Returns a dictionary:
    {
      "knns": [indices_of_k_nearest_neighbors],
      "distances": [distances_of_those_neighbors]
    }
    """
    k = int(k)
    
    if not isinstance(v, Vector):
        raise TypeError("v must be a Vector.")
    if not isinstance(M, Matrix):
        raise TypeError("M must be a Matrix.")

    distances = []
    for i in range(M.rows):
        row_vec = M.get_row_vector(i)
        d = compute_distance(v, row_vec, method)
        distances.append((i, d))

    # Sort by distance ascending
    distances.sort(key=lambda x: x[1])
    top_k = distances[:k]

    return {
        "knns": [idx for idx, dist in top_k],
        "distances": [dist for idx, dist in top_k]
    }

def knn_bf_batch(M1, M2, method="cosine"):
    """
    Returns:
    (min_sum, {m1_index: (closest_m2_index, distance)})
    Where min_sum is sum of minimal distances of each point in M1 to M2
    """
    if not isinstance(M1, Matrix) or not isinstance(M2, Matrix):
        raise TypeError("M1 and M2 must be Matrix types.")

    mapping = {}
    aggregate_distance = 0.0

    for i in range(M1.rows):
        v = M1.get_row_vector(i)
        best_index = None
        best_distance = None

        for j in range(M2.rows):
            w = M2.get_row_vector(j)
            d = compute_distance(v, w, method)
            if best_distance is None or d < best_distance:
                best_distance = d
                best_index = j

        aggregate_distance += best_distance
        mapping[i] = (best_index, best_distance)

    return aggregate_distance, mapping

########################################
# Lexer
########################################

TOKENS = [
    ("COMMENT", r"#[^\n]*"),  # Match '#' followed by any characters until a newline
    ("NUMBER", r"\d+(\.\d+)?"),
    ("VECTOR", r"vector"),
    ("MATRIX", r"matrix"),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z_0-9]*"),
    ("OPERATOR", r"[\+\-\*@]"),
    ("COMMA", r","),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("EQUALS", r"="),
    ("STRING", r'"[^"]*"'),
    ("WHITESPACE", r"\s+"),
]

class Lexer:
    def __init__(self, code):
        self.code = code
        self.tokens = []
        self.tokenize()

    def tokenize(self):
        code = self.code
        while code:
            for token_type, regex in TOKENS:
                match = re.match(regex, code)
                if match:
                    if token_type not in ("WHITESPACE", "COMMENT"):
                        self.tokens.append((token_type, match.group(0)))
                    code = code[len(match.group(0)):]
                    break
            else:
                raise ValueError(f"Unknown token in code: {code}")

    def __iter__(self):
        return iter(self.tokens)

########################################
# AST Nodes
########################################

class ASTNode:
    def __init__(self, type, value=None, children=None):
        self.type = type
        self.value = value
        self.children = children or []

    def __repr__(self):
        return f"ASTNode(type={self.type}, value={self.value}, children={self.children})"

########################################
# Parser
########################################

class Parser:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.current = None
        self.advance()

    def advance(self):
        try:
            self.current = next(self.tokens)
        except StopIteration:
            self.current = None

    def expect(self, token_type):
        if not self.current or self.current[0] != token_type:
            raise ValueError(f"Expected {token_type}, got {self.current}")
        val = self.current[1]
        self.advance()
        return val

    def parse(self):
        nodes = []
        while self.current:
            if self.current and self.current[0] == "IDENTIFIER":
                # Could be an assignment or expression
                # Look ahead: if next is EQUALS, it's assignment, else it's an expression statement (optional)
                var_name = self.current[1]
                self.advance()
                if self.current and self.current[0] == "EQUALS":
                    # Assignment
                    self.advance()  # consume '='
                    expr = self.parse_expr()
                    nodes.append(ASTNode("ASSIGNMENT", value=var_name, children=[expr]))
                else:
                    # This could be just an expression with a variable or maybe re-check tokens
                    # For simplicity, we won't handle standalone expr statements here.
                    pass
            else:
                self.advance()
        return nodes

    def parse_expr(self):
        # Parse based on current token
        if self.current is None:
            return ASTNode("NOOP")

        if self.current[0] == "VECTOR":
            return self.parse_vector_literal()
        elif self.current[0] == "MATRIX":
            return self.parse_matrix_literal()
        elif self.current[0] == "IDENTIFIER":
            # Could be variable reference or function call
            name = self.current[1]
            self.advance()
            if self.current and self.current[0] == "LPAREN":
                return self.parse_func_call(name)
            else:
                # Variable reference
                return ASTNode("VARIABLE", value=name)
        elif self.current[0] == "NUMBER":
            val = float(self.current[1])
            self.advance()
            return ASTNode("NUMBER", value=val)
        elif self.current[0] == "STRING":
            val = self.current[1].strip('"')
            self.advance()
            return ASTNode("STRING", value=val)
        else:
            # If no match, noop
            self.advance()
            return ASTNode("NOOP")

    def parse_func_call(self, func_name):
        # func_call: IDENTIFIER LPAREN arg_list? RPAREN
        self.expect("LPAREN")
        args = []
        if self.current and self.current[0] != "RPAREN":
            args = self.parse_arg_list()
        self.expect("RPAREN")
        return ASTNode("FUNC_CALL", value=func_name, children=args)

    def parse_arg_list(self):
        # arg_list: argument (COMMA argument)*
        args = [self.parse_argument()]
        while self.current and self.current[0] == "COMMA":
            self.advance()
            args.append(self.parse_argument())
        return args

    def parse_argument(self):
        # argument: expr | keyword_arg
        if self.current and self.next_is_keyword_arg():
            return self.parse_keyword_arg()
        else:
            return self.parse_expr()

    def next_is_keyword_arg(self):
        # Peek ahead: If current is IDENTIFIER and next is EQUALS, it's keyword arg
        # Since we don't have direct peek, we can temporarily store current token.
        if self.current and self.current[0] == "IDENTIFIER":
            # Need to peek next token
            # We'll need to save state, advance, then restore.
            save = self.current
            try:
                next_token = next(self.tokens)
                # We'll revert to previous state by re-inserting tokens
                self.tokens = self.reinsert_token(next_token, self.tokens)
                if next_token[0] == "EQUALS":
                    return True
            except StopIteration:
                pass
            # If no next token or not equals
            # revert to original current
        return False

    def reinsert_token(self, token, iterator):
        # Utility to revert an iterator after peek
        # Convert iterator to list and put token back front
        tokens_list = [token] + list(iterator)
        return iter(tokens_list)

    def parse_keyword_arg(self):
        # keyword_arg: IDENTIFIER EQUALS expr
        key = self.expect("IDENTIFIER")
        self.expect("EQUALS")
        value = self.parse_expr()
        return ASTNode("KEYWORD_ARG", value=key, children=[value])

    def parse_vector_literal(self):
        # vector_literal : VECTOR LPAREN array RPAREN
        self.expect("VECTOR")
        self.expect("LPAREN")
        arr = self.parse_array()
        self.expect("RPAREN")
        return ASTNode("VECTOR", value=arr)

    def parse_matrix_literal(self):
        # matrix_literal : MATRIX LPAREN array RPAREN
        self.expect("MATRIX")
        self.expect("LPAREN")
        arr = self.parse_array()
        # Verify arr is list of lists
        if not arr or not all(isinstance(row, list) for row in arr):
            raise ValueError("Matrix must be a list of lists.")
        self.expect("RPAREN")
        return ASTNode("MATRIX", value=arr)

    def parse_array(self):
        # parse_array: LBRACKET element (, element)* RBRACKET
        self.expect("LBRACKET")
        elements = []
        while self.current and self.current[0] != "RBRACKET":
            el = self.parse_element()
            elements.append(el)
            if self.current and self.current[0] == "COMMA":
                self.advance()
        self.expect("RBRACKET")
        return elements

    def parse_element(self):
        # element can be NUMBER or LBRACKET... (nested array)
        if not self.current:
            raise ValueError("Unexpected end in array")

        if self.current[0] == "NUMBER":
            val = float(self.current[1])
            self.advance()
            return val
        elif self.current[0] == "LBRACKET":
            return self.parse_array()
        else:
            raise ValueError("Expected NUMBER or '[' in array")

########################################
# Interpreter
########################################

class Interpreter:
    def __init__(self):
        self.environment = {
            "knn_bf": knn_bf,
            "knn_bf_batch": knn_bf_batch,
            "similarity": self.similarity_wrapper
        }

    def evaluate(self, node):
        if node.type == "ASSIGNMENT":
            var_name = node.value
            value = self.evaluate(node.children[0])
            self.environment[var_name] = value
            return value
        elif node.type == "VECTOR":
            return Vector(node.value)
        elif node.type == "MATRIX":
            return Matrix(node.value)
        elif node.type == "VARIABLE":
            return self.environment.get(node.value, None)
        elif node.type == "NUMBER":
            return node.value
        elif node.type == "STRING":
            return node.value
        elif node.type == "FUNC_CALL":
            return self.call_function(node.value, node.children)
        elif node.type == "KEYWORD_ARG":
            # Should not evaluate alone; handled in call_function
            pass
        else:
            return None

    def call_function(self, func_name, args):
        # Separate positional and keyword arguments
        positional_args = []
        keyword_args = {}
        for arg in args:
            if arg.type == "KEYWORD_ARG":
                key = arg.value
                val = self.evaluate(arg.children[0])
                keyword_args[key] = val
            else:
                val = self.evaluate(arg)
                positional_args.append(val)

        func = self.environment.get(func_name, None)
        if not func:
            raise ValueError(f"Function {func_name} not defined.")

        # Call the function
        return func(*positional_args, **keyword_args)

    def similarity_wrapper(self, v1, v2, method="cosine"):
        if not isinstance(v1, Vector) or not isinstance(v2, Vector):
            raise TypeError("similarity requires two vectors.")
        return v1.similarity(v2, method=method)

########################################
# Example Usage
########################################

if __name__ == "__main__":
    code = """
    v1 = vector([1, 2, 3])
    m1 = matrix([[1,2,3],[4,5,6],[7,8,9]])
    m2 = matrix([[1,1,1],[2,2,2],[9,9,9]])
    """

    lexer = Lexer(code)
    tokens = list(lexer)
    parser = Parser(tokens)
    ast = parser.parse()
    interpreter = Interpreter()

    for node in ast:
        interpreter.evaluate(node)

    v1 = interpreter.environment["v1"]
    m1 = interpreter.environment["m1"]
    m2 = interpreter.environment["m2"]

    # Test knn_bf
    print("knn_bf (cosine):", knn_bf(v1, m1, 2, method="cosine"))
    print("knn_bf (euclidean):", knn_bf(v1, m1, 2, method="euclidean"))

    # Test knn_bf_batch
    min_sum_eu, mapping_eu = knn_bf_batch(m1, m2, method="euclidean")
    print("knn_bf_batch (euclidean):", min_sum_eu, mapping_eu)

    min_sum_cos, mapping_cos = knn_bf_batch(m1, m2, method="cosine")
    print("knn_bf_batch (cosine):", min_sum_cos, mapping_cos)
