
import itertools
import numpy as np
import tqdm
import multiprocessing
import functools
from typing import List, NamedTuple

from wordle import Info, CharInfo, main, GameState

import compute_entropy_python

class Guess(NamedTuple):
    guess: str
    score: float

def matches_info(info: List[CharInfo], answer: str) -> bool:
    assert len(info) == len(answer)
    letter_counts: Dict[str, int] = {}
    right_counts: Dict[str, int] = {}
    for i, char_info in enumerate(info):
        if char_info.info == Info.RIGHT and answer[i] == char_info.char:
            right_counts[char_info.char] = right_counts.get(char_info.char, 0) + 1
        letter_counts[char_info.char] = 0
        for answer_char in answer:
            if answer_char == char_info.char:
                letter_counts[char_info.char] += 1

    unaccounted_counts = {char: letter_counts[char] - right_counts.get(char, 0) for char in letter_counts}

    for i, char_info in enumerate(info):
        if char_info.info == Info.RIGHT and char_info.char != answer[i]:
            return False
        elif char_info.info == Info.IN_WORD:
            if char_info.char == answer[i]:
                return False

            char_in_word = letter_counts[char_info.char] > 0
            if char_in_word and unaccounted_counts[char_info.char] > 0:
                unaccounted_counts[char_info.char] -= 1
            elif char_in_word and unaccounted_counts[char_info.char] == 0:
                return False
            elif not char_in_word:
                return False

        elif char_info.info == Info.WRONG:
            if char_info.char == answer[i]:
                return False
            char_in_word = letter_counts[char_info.char] > 0
            if char_in_word and unaccounted_counts[char_info.char] > 0:
                return False

    return True


def filter_answer_list(info: List[CharInfo], answer_list: List[str]):
    options = list(answer_list)

    return list(filter(lambda x: matches_info(info, x), options))

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

def compute_guess_scores(guess_list: List[str], answer_list: List[str]):
    guesses = []
    for guess in guess_list:
        guesses.append(Guess(guess=guess, score=compute_entropy_python.compute_entropy(guess, answer_list)))
    return guesses

def compute_ranked_guesses(guess_list: List[str], answer_list: List[str], quiet: bool=False) -> List[Guess]:
    assert len(answer_list) > 0
    if len(answer_list) == 1:
        assert answer_list[0] in guess_list
        return [Guess(guess=answer_list[0], score=0.0)]

    func = functools.partial(compute_guess_scores, answer_list=answer_list)
    with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pool:
        def chunker(l: List[str], chunksize: int):
            out = []
            for start_idx in range(0, len(l), chunksize):
                out.append(l[start_idx:start_idx+chunksize])
            return out

        results = pool.map_async(func, tqdm.tqdm(chunker(guess_list, 100), disable=quiet))
        pool.close()
        pool.join()

    guesses = itertools.chain(*results.get())

    return sorted(guesses, key=lambda x: x.score)

class WordleSolver:
    def __init__(self, valid_guesses: List[str], possible_answers: List[str], quiet: bool = False):
        self._valid_guesses = list(valid_guesses)
        self._possible_answers = list(possible_answers)
        self._quiet = quiet

    def compute_ranked_guesses(self) -> List[Guess]:
        return compute_ranked_guesses(self._valid_guesses, self._possible_answers, self._quiet)


    def handle_information(self, info: List[CharInfo]):
        self._possible_answers = filter_answer_list(info, self._possible_answers)

class FixedWordleSolver:
    def __init__(self, filename: str):
        with open(filename, 'rb') as file_in:
            import pickle
            self._root = pickle.load(file_in)
            self._curr = self._root

    def handle_information(self, info: List[CharInfo]):
        only_infos = tuple([x.info for x in info])
        self._curr = self._curr.options[only_infos]

    def compute_ranked_guesses(self) -> List[Guess]:
        return [Guess(guess=self._curr.guess, score=0.0)]

    def reset(self):
        self._curr = self._root


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Wordle, but with numbers')
    parser.add_argument('--num_digits', default=2, type=int)
    parser.add_argument('--num_guesses', default=10, type=int)
    parser.add_argument('--use_answer', default=None, type=str)
    parser.add_argument('--use_words', default=False, action='store_true')
    args = parser.parse_args()

    solver = None
    def handle_guess_lists(valid_guesses: List[str], possible_answers: List[str]):
        global solver
        solver = WordleSolver(valid_guesses, possible_answers)

    def get_input() -> str:
        ranked_guesses = solver.compute_ranked_guesses()
        print('Possible answers left:', len(solver._possible_answers))
        if len(solver._possible_answers) < 20:
            print(solver._possible_answers)
        print('Best Guesses:')
        for i, guess in enumerate(ranked_guesses[-10:]):
            print(i, guess)

        return ranked_guesses[-1].guess


    def handle_result(result: GameState):
        assert result.answer in solver._possible_answers, 'Answer not in possible list before update'
        solver.handle_information(result.information[-1])
        assert result.answer in solver._possible_answers, 'Answer not in possible list after update'

    main(args.num_digits,
         args.num_guesses,
         str(args.use_answer).zfill(args.num_digits) if args.use_answer else None,
         use_words=args.use_words,
         handle_guess_lists=handle_guess_lists,
         get_input=get_input,
         handle_result=handle_result)
