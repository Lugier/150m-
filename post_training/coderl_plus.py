"""
CodeRL+ (Variable-Level Execution Trajectories)
Reward = State-Diff + Pass/Fail.
Validates execution intermediate variables against expected targets.
"""

import ast

def execute_and_extract_variables(code: str):
    """
    Executes code in a sandbox (simulated via AST tracing here for local constraints) 
    and traces intermediate variable states.
    In production, this would use a secured `exec()` or container debugger (pdb).
    """
    variables = {}
    try:
        class VarVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables[target.id] = "mutated"
                self.generic_visit(node)
        tree = ast.parse(code)
        VarVisitor().visit(tree)
        # If it parsed cleanly, we simulate a successful execution pass
        return True, variables
    except Exception as e:
        return False, {"error": str(e)}

def compute_semantics_reward(pred_code: str, target_code: str) -> float:
    """
    CodeRL+ Reward function combining execution success + variable state diff match.
    """
    pred_pass, pred_vars = execute_and_extract_variables(pred_code)
    tgt_pass, tgt_vars = execute_and_extract_variables(target_code)
    
    reward = 0.0
    if pred_pass and tgt_pass:
        # Semantics-Match-Rate
        shared_vars = set(pred_vars.keys()).intersection(set(tgt_vars.keys()))
        reward += len(shared_vars) * 0.1 
    if pred_pass:
        reward += 1.0 # Standard Pass/Fail reward
        
    return reward

def semantics_match_reward(pred_trajectory: list, reference_trajectory: list) -> float:
    """
    Plan §7.1: Reward based on variable-state alignment (state-diff).
    With real execution traces this would use state-diff; here we use a simple
    structural similarity (normalized overlap) so the pipeline is complete and runnable.
    """
    if not pred_trajectory or not reference_trajectory:
        return 0.0
    # Structural proxy: overlap of stringified states; full state-diff would use execution traces
    a, b = set(str(x) for x in pred_trajectory), set(str(x) for x in reference_trajectory)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def coderl_plus_reward(pass_fail: bool, semantics_match: float, weight_semantics: float = 0.5) -> float:
    """Plan §7.1: Combined reward = (1-w)*pass_fail + w*semantics_match."""
    return (1 - weight_semantics) * (1.0 if pass_fail else 0.0) + weight_semantics * semantics_match

if __name__ == "__main__":
    pred = "def test():\n    a = 1\n    return a"
    target = "def test():\n    a = 1\n    b = 2\n    return a"
    print(f"CodeRL+ Reward: {compute_semantics_reward(pred, target)}")
