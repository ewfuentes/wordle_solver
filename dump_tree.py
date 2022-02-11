
from __future__ import annotations

import words
import wordle
import wordle_solver
import itertools
import compute_entropy_python as cep
import copy
from typing import NamedTuple, Dict, Optional, List

class GameNode(NamedTuple):
    guess: str
    options: Optional[Dict[List[wordle.Info], Optional[GameNode]]]

def dump_tree(solver: wordle_solver.WordleSolver, depth=0, category=None) -> Optional[GameNode]:
    try:
        guesses = solver.compute_ranked_guesses()
        print('-'*depth, guesses[-1].guess, len(solver._possible_answers),
              '' if len(solver._possible_answers) > 5 else solver._possible_answers, category)
        if len(guesses) == 1:
            return GameNode(guess=guesses[0].guess, options=None)
        guess = guesses[-1].guess
        options = {}
        cats = [wordle.Info.WRONG, wordle.Info.IN_WORD, wordle.Info.RIGHT]
        for categories in itertools.product(cats, repeat=len(guess)):
            solver_copy = copy.deepcopy(solver)
            char_info = [wordle.CharInfo(char, info) for char, info in zip(guess, categories)]
            solver_copy.handle_information(char_info)
            options[categories] = dump_tree(solver_copy, depth=depth+1, category=categories)
        return GameNode(guess=guess, options=options)
    except Exception as e:
        return None

def main():
    all_words = words.possible_answers + words.valid_guesses
    solver = wordle_solver.WordleSolver(all_words, all_words, quiet=True)
    game_tree = dump_tree(solver)
    with open('tree.p', 'wb') as file_out:
        import pickle
        pickle.dump(game_tree, file_out)



if __name__ == '__main__':
    main()
