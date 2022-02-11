
from typing import List, NamedTuple, Optional
import wordle
import wordle_solver
import words
import tqdm
import argparse

def run_game(answer: str):
    solver = None
    is_first_guess = True
    final_state = None
    def handle_guess_lists(valid_guesses: List[str], possible_answers: List[str]):
        nonlocal solver
        solver = wordle_solver.WordleSolver(valid_guesses, possible_answers, quiet=True)

    def get_input() -> str:
        nonlocal is_first_guess
        if is_first_guess:
            is_first_guess = False
            return 'tares'

        ranked_guesses = solver.compute_ranked_guesses()
        return ranked_guesses[-1].guess

    def handle_result(result: wordle.GameState):
        nonlocal final_state
        if result.is_finished:
            final_state = result
        assert result.answer in solver._possible_answers, 'Answer not in possible list before update'
        solver.handle_information(result.information[-1])
        assert result.answer in solver._possible_answers, 'Answer not in possible list after update'

    wordle.main(num_digits = 0, num_guesses=10, use_answer=answer, use_words=True,
                handle_guess_lists=handle_guess_lists, get_input=get_input, handle_result=handle_result,
                quiet=True)
    return final_state

def main(filename: str):
    all_words = words.valid_guesses + words.possible_answers
    final_states = []
    for word in tqdm.tqdm(all_words):
        try:
            final_states.append(run_game(word))
        except:
            break
    with open(filename, 'wb') as file_out:
        import pickle
        pickle.dump(final_states, file_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()
    main(args.output)
