
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
