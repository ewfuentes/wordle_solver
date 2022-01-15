
import itertools
import numpy as np
from typing import List, NamedTuple

from wordle import Info, CharInfo, main


class Guess(NamedTuple):
    guess: str
    score: float

def filter_answer_list(info: List[CharInfo], answer_list: List[str]):
    options = list(answer_list)
    for i, char_info in enumerate(info):
        if char_info.info == Info.WRONG:
            # If this letter does not exist in word, remove all words with this letter
            options = [x for x in options if char_info.char not in x]
        elif char_info.info == Info.IN_WORD:
            # This this letter exists in the word, remove all words that don't have this letter,
            # or have this letter in this position
            options = [x for x in options if x[i] != char_info.char]
            options = [x for x in options if char_info.char in x]
        elif char_info.info == Info.RIGHT:
            # This letter is in the correct position, remove all words that don't have this
            # in this position
            options = [x for x in options if x[i] == char_info.char]
        else:
            assert False, f'Unknown Info type: {char_info.info} at index {i}'
    return options

def compute_entropy(guess: str, answer_list: List[str]):
    # each character in guess can either:
    #  - not be in the answer
    #  - be in the answer, but in the wrong position
    #  - be in the answer in the right position
    # as a result, each item in answer list will fall into one of 3 ** len(guess) categories

    # compute counts
    counts = [0] * (3 ** len(guess))
    for i, categories in enumerate(itertools.product([Info.WRONG, Info.IN_WORD, Info.RIGHT], repeat=len(guess))):
        info = [CharInfo(guess[idx], cat) for idx, cat in enumerate(categories)]
        counts[i] = len(filter_answer_list(info, answer_list))

    num_answers = len(answer_list)
    probabilities = np.array(counts) / num_answers
    return -1.0 * np.nansum(probabilities * np.log2(probabilities, where=probabilities > 0.0))

def compute_ranked_guesses(guess_list: List[str], answer_list: List[str]) -> List[Guess]:
    if len(answer_list) == 1:
        assert answer_list[0] in guess_list
        return [Guess(guess=answer_list[0], score=0.0)]

    guesses = []
    for guess in guess_list:
        guesses.append(Guess(guess=guess, score=compute_entropy(guess, answer_list)))

    return sorted(guesses, key=lambda x: x.score)

class WordleSolver:
    def __init__(self, valid_guesses: List[str], possible_answers: List[str]):
        self._valid_guesses = list(valid_guesses)
        self._possible_answers = list(possible_answers)

    def compute_ranked_guesses(self) -> List[Guess]:
        return compute_ranked_guesses(self._valid_guesses, self._possible_answers)


    def handle_information(self, info: List[CharInfo]):
        self._possible_answers = filter_answer_list(info, self._possible_answers)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Wordle, but with numbers')
    parser.add_argument('--num_digits', default=2, type=int)
    parser.add_argument('--num_guesses', default=10, type=int)
    args = parser.parse_args()

    solver = None
    def handle_guess_lists(valid_guesses: List[str], possible_answers: List[str]):
        global solver
        solver = WordleSolver(valid_guesses, possible_answers)

    def get_input() -> str:
        ranked_guesses = solver.compute_ranked_guesses()
        guesses_from_score: Dict[float, List[str]] = {}
        for g in ranked_guesses:
            guess, score = g
            guesses_from_score[score] = guesses_from_score.get(score, []) + [guess]

        for score, guesses in guesses_from_score.items():
            print('Score:', score)
            print(guesses)
        print('Possible Answers:', solver._possible_answers)
        return ranked_guesses[-1].guess

    def handle_result(result: List[CharInfo]):
        solver.handle_information(result)

    main(args.num_digits,
         args.num_guesses,
         handle_guess_lists=handle_guess_lists,
         get_input=get_input,
         handle_result=handle_result)
