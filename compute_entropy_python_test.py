
import unittest

import wordle_solver
import compute_entropy_python as cep

class ComputeEntropyPythonTest(unittest.TestCase):
    def test_compute_entropy_happy_case(self):
        answer_list = [str(x).zfill(2) for x in range(100)]
        guess = '78'

        cpp_entropy = cep.compute_entropy(guess, answer_list)
        python_entropy = wordle_solver.compute_entropy(guess, answer_list)

        self.assertAlmostEqual(cpp_entropy, python_entropy)

    def test_compute_counts_in_word_no_repeat(self):
        answer_list = ['01']
        guess = '12'

        counts = cep._compute_counts(guess, answer_list)
        for idx, count in enumerate(counts):
            categories = cep._get_categories_for_index(idx, len(guess))
            expected_count = (1
                              if categories[0] == cep.Category.IN_WORD and categories[1] == cep.Category.WRONG
                              else 0)
            self.assertEqual(count, expected_count)

    def test_compute_counts_in_word_repeat_in_guess(self):
        answer_list = ['01']
        guess = '11'

        counts = cep._compute_counts(guess, answer_list)
        for idx, count in enumerate(counts):
            categories = cep._get_categories_for_index(idx, len(guess))
            with self.subTest(categories=categories, guess=guess, answer_list=answer_list):
                expected_count = (1
                                if categories[0] == cep.Category.WRONG and categories[1] == cep.Category.RIGHT
                                else 0)
                self.assertEqual(count, expected_count)

    def test_compute_counts_in_word_repeat_in_guess_and_answer(self):
        answer_list = ['011']
        guess = '112'

        counts = cep._compute_counts(guess, answer_list)
        for idx, count in enumerate(counts):
            categories = cep._get_categories_for_index(idx, len(guess))
            with self.subTest(categories=categories, guess=guess, answer_list=answer_list):
                expected_count = (1
                                if categories[0] == cep.Category.IN_WORD
                                  and categories[1] == cep.Category.RIGHT
                                  and categories[2] == cep.Category.WRONG
                                else 0)
                self.assertEqual(count, expected_count)


if __name__ == "__main__":
    unittest.main()
