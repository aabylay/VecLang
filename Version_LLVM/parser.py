from ast import ASTNode

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
        # Skip leading newlines
        while self.current and self.current[0] == "NEWLINE":
            self.advance()

        statements = []
        while self.current:
            while self.current and self.current[0] == "NEWLINE":
                self.advance()

            if self.current is None:
                break

            if self.current[0] == "IDENTIFIER":
                stmt = self.parse_statement()
                statements.append(stmt)
            else:
                raise ValueError(f"Expected a statement (IDENTIFIER), got {self.current}")

            while self.current and self.current[0] == "NEWLINE":
                self.advance()

        return statements

    def parse_statement(self):
        return self.parse_assignment()

    def parse_assignment(self):
        # assignment : var_list EQUALS expr
        var_names = self.parse_var_list()
        self.expect("EQUALS")
        expr_node = self.parse_expr()
        # var_names is a list of variable names
        return ASTNode("ASSIGNMENT", value=var_names, children=[expr_node])

    def parse_var_list(self):
        # var_list : IDENTIFIER (COMMA IDENTIFIER)*
        var_names = []
        var_names.append(self.expect("IDENTIFIER"))
        while self.current and self.current[0] == "COMMA":
            self.advance()
            var_names.append(self.expect("IDENTIFIER"))
        return var_names

    def parse_expr(self):
        node = self.parse_term()
        while self.current and self.current[0] == "OPERATOR":
            op = self.current[1]
            self.advance()
            right = self.parse_term()
            node = ASTNode("BINARY_OP", value=op, children=[node, right])
        return node

    def parse_term(self):
        return self.parse_primary()

    def parse_primary(self):
        if not self.current:
            return ASTNode("NOOP")

        if self.current[0] == "IDENTIFIER":
            name = self.current[1]
            self.advance()
            if self.current and self.current[0] == "LPAREN":
                return self.parse_func_call(name)
            else:
                return ASTNode("VARIABLE", value=name)

        elif self.current[0] == "NUMBER":
            val = float(self.current[1])
            self.advance()
            return ASTNode("NUMBER", value=val)

        elif self.current[0] == "STRING":
            val = self.current[1].strip('"')
            self.advance()
            return ASTNode("STRING", value=val)

        elif self.current[0] == "VECTOR":
            return self.parse_vector_literal()

        elif self.current[0] == "MATRIX":
            return self.parse_matrix_literal()

        elif self.current[0] == "LPAREN":
            self.advance()
            node = self.parse_expr()
            self.expect("RPAREN")
            return node

        else:
            self.advance()
            return ASTNode("NOOP")

    def parse_func_call(self, func_name):
        self.expect("LPAREN")
        args = []
        if self.current and self.current[0] != "RPAREN":
            args = self.parse_arg_list()
        self.expect("RPAREN")
        return ASTNode("FUNC_CALL", value=func_name, children=args)

    def parse_arg_list(self):
        args = [self.parse_argument()]
        while self.current and self.current[0] == "COMMA":
            self.advance()
            args.append(self.parse_argument())
        return args

    def parse_argument(self):
        # argument can be expr or keyword_arg
        if self.current and self.current[0] == "IDENTIFIER":
            # Peek ahead for EQUALS to detect keyword arg
            save_current = self.current
            peek = self._peek()
            if peek and peek[0] == "EQUALS":
                # keyword arg
                key = self.expect("IDENTIFIER")
                self.expect("EQUALS")
                val_expr = self.parse_expr()
                return ASTNode("KEYWORD_ARG", value=key, children=[val_expr])
            else:
                # Normal expr
                return self.parse_primary()
        else:
            return self.parse_expr()

    def parse_vector_literal(self):
        self.expect("VECTOR")
        self.expect("LPAREN")
        arr = self.parse_array()
        self.expect("RPAREN")
        return ASTNode("VECTOR", value=arr)

    def parse_matrix_literal(self):
        self.expect("MATRIX")
        self.expect("LPAREN")
        arr = self.parse_array()
        if not arr or not all(isinstance(r, list) for r in arr):
            raise ValueError("Matrix must be a list of lists.")
        self.expect("RPAREN")
        return ASTNode("MATRIX", value=arr)

    def parse_array(self):
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

    def _peek(self):
        # Peeks next token without consuming current
        try:
            nxt = next(self.tokens)
            self.tokens = self.reinsert_token(nxt, self.tokens)
            return nxt
        except StopIteration:
            return None

    def reinsert_token(self, token, iterator):
        tokens_list = [token] + list(iterator)
        return iter(tokens_list)
