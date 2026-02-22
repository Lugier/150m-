"""
CODA (Plan §2.6): Code Difference-Guided Adversarial Augmentation.
Marginale Deltas (z.B. < → <=, + → -) für Stage-2-Daten; in prepare_data.stage_stream
bei coda_mutation_rate angewendet.
"""

import ast
import random

class CODAMutator(ast.NodeTransformer):
    def __init__(self):
        self.mutations = 0

    def visit_Compare(self, node):
        self.generic_visit(node)
        if random.random() < 0.2:
            for i, op in enumerate(node.ops):
                if isinstance(op, ast.Lt):
                    node.ops[i] = ast.LtE()
                elif isinstance(op, ast.LtE):
                    node.ops[i] = ast.Lt()
                elif isinstance(op, ast.Gt):
                    node.ops[i] = ast.GtE()
                elif isinstance(op, ast.GtE):
                    node.ops[i] = ast.Gt()
                elif isinstance(op, ast.Eq):
                    node.ops[i] = ast.NotEq()
                elif isinstance(op, ast.NotEq):
                    node.ops[i] = ast.Eq()
            self.mutations += 1
        return node
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if random.random() < 0.2:
            if isinstance(node.op, ast.Add):
                node.op = ast.Sub()
            elif isinstance(node.op, ast.Sub):
                node.op = ast.Add()
            self.mutations += 1
        return node

def apply_coda(code: str) -> str:
    """AST-basierte Mutation (Compare/BinOp); für Stage-2-Stream in prepare_data."""
    try:
        tree = ast.parse(code)
        mutator = CODAMutator()
        mutated_tree = mutator.visit(tree)
        if mutator.mutations > 0 and hasattr(ast, 'unparse'):
            return ast.unparse(mutated_tree)
        return code
    except Exception:
        return code

if __name__ == "__main__":
    example = "def compute(a, b):\n    if a < b:\n        return a + b\n    return a"
    print("Original:\n", example)
    print("CODA Mutated:\n", apply_coda(example))
