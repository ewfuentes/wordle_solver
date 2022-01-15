import unittest

from wordle import Wordle, compute_information, Info

class WordleTest(unittest.TestCase):
    GUESS_LIST = ['aaaaa', 'aaaab', 'aaaac', 'aaaad', 'aaaae']
    ANSWER_LIST = ['aaaaa', 'aaaab', 'aaaac', 'aaaad']
    def test_provide_answer(self):
        wordle = Wordle(self.GUESS_LIST, answer=self.ANSWER_LIST[0])
        self.assertEqual(wordle._answer, self.ANSWER_LIST[0])

    def test_provide_answer_list(self):
        wordle = Wordle(self.GUESS_LIST, possible_answers=self.ANSWER_LIST)
        self.assertIn(wordle._answer, self.ANSWER_LIST)

    def test_provide_answer_and_answer_list(self):
        with self.assertRaises(AssertionError):
            Wordle(self.GUESS_LIST, answer=self.ANSWER_LIST[0], possible_answers=self.ANSWER_LIST)

    def test_provide_no_answer_or_answer_list(self):
        with self.assertRaises(AssertionError):
            Wordle(self.GUESS_LIST)

class ComputeInformationTest(unittest.TestCase):
    def test_unequal_length_throws(self):
        with self.assertRaises(AssertionError):
            compute_information('123', '9')

    def test_compute_info_unique(self):
        GUESS = '123'
        ANSWER = '324'
        EXPECTED_INFO = [Info.WRONG, Info.RIGHT, Info.IN_WORD]
        is_winner, info = compute_information(GUESS, ANSWER)
        self.assertFalse(is_winner)
        self.assertEqual(len(info), len(GUESS))
        for i, char_info in enumerate(info):
            self.assertEqual(char_info.char, GUESS[i])
            self.assertEqual(char_info.info, EXPECTED_INFO[i])

    def test_compute_info_repeated(self):
        GUESS = '222'
        ANSWER = '122'
        EXPECTED_INFO = [Info.IN_WORD, Info.RIGHT, Info.RIGHT]
        is_winner, info = compute_information(GUESS, ANSWER)
        self.assertFalse(is_winner)
        self.assertEqual(len(info), len(GUESS))
        for i, char_info in enumerate(info):
            self.assertEqual(char_info.char, GUESS[i])
            self.assertEqual(char_info.info, EXPECTED_INFO[i])

    def test_compute_info_winner(self):
        GUESS = '123'
        ANSWER = '123'
        EXPECTED_INFO = [Info.RIGHT, Info.RIGHT, Info.RIGHT]
        is_winner, info = compute_information(GUESS, ANSWER)
        self.assertTrue(is_winner)
        self.assertEqual(len(info), len(GUESS))
        for i, char_info in enumerate(info):
            self.assertEqual(char_info.char, GUESS[i])
            self.assertEqual(char_info.info, EXPECTED_INFO[i])

if __name__ == '__main__':
    unittest.main()
