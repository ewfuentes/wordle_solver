import unittest

import wordle_solver
import wordle
import compute_entropy_python as cep

import words

class WordleSolverTest(unittest.TestCase):
    def test_filter_answer_list_right_answer(self):
        answer_list = [str(x).zfill(2) for x in range(100)]
        info = [wordle.CharInfo('6', wordle.Info.RIGHT), wordle.CharInfo('7', wordle.Info.RIGHT)]

        valid_answers = wordle_solver.filter_answer_list(info, answer_list)

        self.assertEqual(len(valid_answers), 1)
        self.assertEqual(valid_answers[0], '67')

    def test_filter_answer_list_wrong_answer(self):
        answer_list = [str(x).zfill(2) for x in range(100)]
        info = [wordle.CharInfo('6', wordle.Info.WRONG), wordle.CharInfo('7', wordle.Info.WRONG)]

        valid_answers = wordle_solver.filter_answer_list(info, answer_list)

        for i in range(100):
            number_str = str(i).zfill(2)
            if '6' in number_str or '7' in number_str:
                self.assertNotIn(number_str, valid_answers)
            else:
                self.assertIn(number_str, valid_answers)

    def test_filter_answer_list_in_word(self):
        answer_list = [str(x).zfill(2) for x in range(100)]
        info = [wordle.CharInfo('6', wordle.Info.IN_WORD), wordle.CharInfo('7', wordle.Info.IN_WORD)]

        valid_answers = wordle_solver.filter_answer_list(info, answer_list)

        self.assertEqual(len(valid_answers), 1)
        self.assertEqual(valid_answers[0], '76')

    def test_filter_answer_in_word_repeated(self):
        answer_list = ['bundh', 'bundu']
        guess = 'zulus'
        cat = [wordle.Info.WRONG, wordle.Info.RIGHT, wordle.Info.WRONG, wordle.Info.IN_WORD, wordle.Info.WRONG]
        info = [wordle.CharInfo(char, info) for char, info in zip(guess, cat)]

        valid_answers = wordle_solver.filter_answer_list(info, answer_list)

        self.assertEqual(len(valid_answers), 1)
        self.assertEqual(valid_answers[0], 'bundu')

    def test_best_guess_single_answer(self):
        guess_list = [str(x).zfill(2) for x in range(100)]
        answer_list = ['78']
        ranked_guesses = wordle_solver.compute_ranked_guesses(guess_list, answer_list)

        self.assertEqual(ranked_guesses[-1].guess, answer_list[0])

    def test_best_guess_beginning(self):
        guess_list = [str(x).zfill(2) for x in range(100)]

        ranked_guesses = wordle_solver.compute_ranked_guesses(guess_list, guess_list)

        score_from_guess = {x.guess: x.score for x in ranked_guesses}

        repeats = [f'{i}{i}' for i in range(10)]
        for guess, score in score_from_guess.items():
            if guess in repeats:
                self.assertEqual(score, score_from_guess['00'])
                self.assertLess(score, score_from_guess['01'])
            else:
                self.assertGreater(score, score_from_guess['00'])
                self.assertEqual(score, score_from_guess['01'])

class WordleSolverPerformanceTest(unittest.TestCase):
    def test_1(self):
        answer_list = ['babes', 'faxes', 'gages']
        guess_list = ['zaxes', 'galax']

        self.assertGreater(cep.compute_entropy('galax', answer_list), cep.compute_entropy('zaxes', answer_list))


if __name__ == '__main__':
    unittest.main()
