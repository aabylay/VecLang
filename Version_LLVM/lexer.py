import re

TOKENS = [
    ("COMMENT", r"#[^\n]*"),
    ("NEWLINE", r"\n+"),
    ("NUMBER", r"\d+(\.\d+)?"),
    ("VECTOR", r"vector"),
    ("MATRIX", r"matrix"),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z_0-9]*"),
    ("OPERATOR", r"[\+\-\*@]"),  # +, -, *, @
    ("COMMA", r","),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("EQUALS", r"="),
    ("STRING", r'"[^"]*"'),
    ("WHITESPACE", r"[ \t]+"),
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
                    text = match.group(0)
                    # Ignore WHITESPACE and COMMENT
                    if token_type not in ("WHITESPACE", "COMMENT"):
                        self.tokens.append((token_type, text))
                    code = code[len(text):]
                    break
            else:
                raise ValueError(f"Unknown token in code: {code}")

    def __iter__(self):
        return iter(self.tokens)
