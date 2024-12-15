# main.py
from lexer import Lexer
from parser import Parser
from compiler import LLVMCompiler

def compile_vec_code(code: str) -> str:
    lexer = Lexer(code)
    tokens = list(lexer)
    parser = Parser(tokens)
    ast = parser.parse()

    compiler = LLVMCompiler()
    compiler.compile(ast)
    llvm_ir = compiler.finalize()
    return llvm_ir

if __name__ == "__main__":
    # Example usage:
    sample = """
    v1 = vector([1, 2, 3])
    v2 = vector([4, 5, 6])
    cos_sim = similarity(v1, v2, method="cosine")
    """
    print(compile_vec_code(sample))
