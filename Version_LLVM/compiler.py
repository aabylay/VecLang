from llvmlite import ir

class LLVMCompiler:
    def __init__(self):
        self.module = ir.Module(name="vecmodule")
        self.builder = None
        self.func = None
        self.env = {}
        self.global_counter = 0  # This counter ensures unique global names

        self.define_runtime_functions()

    def define_runtime_functions(self):
        vec_type = ir.IntType(8).as_pointer()
        dbl_ptr = ir.DoubleType().as_pointer()
        int_type = ir.IntType(32)

        # External function declarations from runtime
        self.fn_vector_create = ir.Function(self.module, 
            ir.FunctionType(vec_type, [dbl_ptr, int_type]), 
            name="vector_create")

        self.fn_matrix_create = ir.Function(self.module,
            ir.FunctionType(vec_type, [dbl_ptr, int_type, int_type]),
            name="matrix_create")

        self.fn_vector_add = ir.Function(self.module,
            ir.FunctionType(vec_type, [vec_type, vec_type]),
            name="vector_add")

        self.fn_vector_dot = ir.Function(self.module,
            ir.FunctionType(ir.DoubleType(), [vec_type, vec_type]),
            name="vector_dot")

        self.fn_similarity = ir.Function(self.module,
            ir.FunctionType(ir.DoubleType(), [vec_type, vec_type, int_type]),
            name="similarity")

        # If you have knn_bf, knn_bf_batch, define them similarly
        # self.fn_knn_bf = ...
        # self.fn_knn_bf_batch = ...

    def compile(self, ast_nodes):
        func_type = ir.FunctionType(ir.VoidType(), ())
        self.func = ir.Function(self.module, func_type, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        for node in ast_nodes:
            self.compile_node(node)

        self.builder.ret_void()

    def compile_node(self, node):
        if node.type == "ASSIGNMENT":
            # node.value is a list of variable names
            var_names = node.value
            val = self.compile_node(node.children[0])

            if len(var_names) == 1:
                # Single variable assignment
                var_name = var_names[0]
                self.env[var_name] = val
            else:
                # Multiple assignment scenario
                # If your language supports multiple returns or needs to handle multiple assignments,
                # you'd unpack the 'val' accordingly. For now, if you only have single assignments tested,
                # just handle the single variable case or raise an error.
                raise NotImplementedError("Multiple assignment not yet supported.")

            return val

        elif node.type == "VECTOR":
            arr_vals = node.value
            const_arr = ir.Constant(ir.ArrayType(ir.DoubleType(), len(arr_vals)),
                                    [ir.Constant(ir.DoubleType(), x) for x in arr_vals])

            # Use a unique name for each vector constant
            name = f"vec_const_{self.global_counter}"
            self.global_counter += 1

            global_arr = ir.GlobalVariable(self.module, const_arr.type, name=name)
            global_arr.initializer = const_arr
            global_arr.global_constant = True
            global_arr.linkage = 'internal'

            arr_ptr = self.builder.bitcast(global_arr, ir.DoubleType().as_pointer())
            vec_ptr = self.builder.call(self.fn_vector_create, [arr_ptr, ir.Constant(ir.IntType(32), len(arr_vals))])
            return vec_ptr

        elif node.type == "MATRIX":
            # Handle matrix similarly, ensuring unique names for each global
            arr = node.value
            rows = len(arr)
            cols = len(arr[0]) if rows > 0 else 0
            flat = []
            for r in arr:
                flat.extend(r)

            const_arr = ir.Constant(ir.ArrayType(ir.DoubleType(), len(flat)),
                                    [ir.Constant(ir.DoubleType(), x) for x in flat])

            name = f"mat_const_{self.global_counter}"
            self.global_counter += 1

            global_arr = ir.GlobalVariable(self.module, const_arr.type, name=name)
            global_arr.initializer = const_arr
            global_arr.global_constant = True
            global_arr.linkage = 'internal'

            arr_ptr = self.builder.bitcast(global_arr, ir.DoubleType().as_pointer())
            mat_ptr = self.builder.call(self.fn_matrix_create, [
                arr_ptr,
                ir.Constant(ir.IntType(32), rows),
                ir.Constant(ir.IntType(32), cols)])
            return mat_ptr

        elif node.type == "VARIABLE":
            return self.env[node.value]

        elif node.type == "NUMBER":
            return ir.Constant(ir.DoubleType(), node.value)

        elif node.type == "STRING":
            # Convert strings to method codes if needed, similar logic as before
            val = 0 if node.value == "cosine" else 1
            return ir.Constant(ir.IntType(32), val)

        elif node.type == "BINARY_OP":
            # Handle binary operations. For example, if op is '+', call vector_add if both sides are vectors
            # or handle scalar multiplication if one side is a number. Simplified example:
            left = self.compile_node(node.children[0])
            right = self.compile_node(node.children[1])
            op = node.value

            # You will need logic to distinguish if left/right are vectors or numbers.
            # For simplicity, assume variables referencing vectors or numbers directly:
            # If we detect a vector operation:
            # Example: if op == '+', call self.fn_vector_add if both are vectors
            # If op == '*', if right is a number, call vector_scalar_mul (not yet defined in runtime)
            # Implement according to your runtime functions.

            # Pseudocode:
            if op == '+':
                # Assume both sides are vectors (in real code, check types)
                return self.builder.call(self.fn_vector_add, [left, right])
            elif op == '*':
                # If right is number, call vector_scalar_mul
                # If you have vector_scalar_mul defined similarly in runtime:
                #   fn_vector_scalar_mul = ...
                # In that case:
                # return self.builder.call(self.fn_vector_scalar_mul, [left, right])
                # For now, let's assume vector_scalar_mul is defined similarly to vector_add:
                vec_scalar_mul = self.define_vector_scalar_mul()
                return self.builder.call(vec_scalar_mul, [left, right])
            else:
                # Handle other ops '@', etc.
                if op == '@':
                    # Dot product
                    return self.builder.call(self.fn_vector_dot, [left, right])

            # If not handled, raise error:
            raise ValueError(f"Unsupported operator {op}")
        
        else:
            return None

    def define_vector_scalar_mul(self):
        # If not already defined, define vector_scalar_mul function
        # In a real scenario, you would have defined it in define_runtime_functions or runtime.
        # Let's just declare it:
        if hasattr(self, 'fn_vector_scalar_mul'):
            return self.fn_vector_scalar_mul
        vec_type = ir.IntType(8).as_pointer()
        dbl = ir.DoubleType()
        self.fn_vector_scalar_mul = ir.Function(self.module,
            ir.FunctionType(vec_type, [vec_type, dbl]),
            name="vector_scalar_mul")
        return self.fn_vector_scalar_mul

    def finalize(self):
        return str(self.module)
