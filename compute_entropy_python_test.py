
import unittest

import wordle_solver
import compute_entropy_python

class ComputeEntropyPythonTest(unittest.TestCase):
    def test_compute_entropy_happy_case(self):
        answer_list = [str(x).zfill(2) for x in range(100)]
        guess = '78'

        cpp_entropy = compute_entropy_python.compute_entropy(guess, answer_list)
        python_entropy = wordle_solver.compute_entropy(guess, answer_list)

        self.assertAlmostEqual(cpp_entropy, python_entropy)


if __name__ == "__main__":
    unittest.main()
