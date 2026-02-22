"""
Execution Reward für RL (GRPO): Syntax -1.0, Runtime/Timeout -0.5, kein Test 0.0, Tests bestanden +1.0.
Wird in rl_train.grpo_train_step verwendet; mit CodeRL+ bei reference_code gemischt (Semantics-Match).
"""

import ast
from typing import List, Optional
from evaluation.eval_repair import run_tests_in_sandbox

def compute_execution_reward(code: str, tests: Optional[List[str]] = None) -> float:
    """
    Computes a float reward for a generated payload of code.
    Base Rule:
        -1.0: Syntax Error / Unparseable
        -0.5: Runtime Error / Timed Out during execution
         0.0: Execution Success but no tests were provided to prove correctness
        +1.0: Execution Success and all tests passed
    """
    # 1. Syntax Check
    try:
        ast.parse(code)
    except SyntaxError:
        return -1.0
        
    # We do a quick strip to ensure we aren't executing completely blank code
    if not code.strip():
        return -1.0

    # 2. Execution / Runtime Check
    # We fall back to standard Python run if no tests are needed, otherwise
    # we inject the tests and see if the sandbox survives them.
    # Note: run_tests_in_sandbox evaluates passing standard tests,
    # returning True/False.
    
    if not tests:
        # Just run the code to see if it throws random TypeErrors/NameErrors
        # We wrap it in a pseudo-test logic. run_tests_in_sandbox returns (passed, output).
        passed, _ = run_tests_in_sandbox(code, ["pass"])
        if passed:
            return 0.0  # Standard execution, safe, neutral bonus
        return -0.5

    # 3. Test Validation
    # In rigorous RL datasets (like APPS/HumanEval as seeds), we have tests.
    passed, _ = run_tests_in_sandbox(code, tests)
    return 1.0 if passed else -0.5
