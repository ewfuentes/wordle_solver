

from typing import Optional, List

import words
import wordle_solver
import wordle

def parse_response(feedback: str) -> Optional[List[wordle.CharInfo]]:
    print(feedback)
    elems = feedback.strip().split(' ')
    if len(elems) != 5:
        print('Invalid Feedback', elems)
        return None

    out: List[wordle.CharInfo] = []
    for i, elem in enumerate(elems):
        if len(elem) != 2:
            print(f'Invalid Feedback at {i}: {elem}')
        letter, category = elem
        letter = letter.lower()
        category = category.lower()

        if not letter.isalpha():
            print(f'Invalid letter {letter} at {i}: {elem}')
            return None

        if category not in ['c', 'x', 'i']:
            print(f'Invalid category {category} at {i}: {elem}')
            return None

        info_from_category = {
            'c': wordle.Info.RIGHT,
            'i': wordle.Info.IN_WORD,
            'x': wordle.Info.WRONG,
        }
        out.append(wordle.CharInfo(char=letter, info=info_from_category[category]))

    return out

def main():
    valid_guesses = words.valid_guesses + words.possible_answers
    possible_answers = words.possible_answers
    solver = wordle_solver.WordleSolver(valid_guesses=valid_guesses, possible_answers=possible_answers)

    print('Enter feedback as Xy Xy Xy Xy Xy where X are the letters,')
    print('and y is c if correct, i if in word or x if wrong')
    run = True
    while run:
        ranked_guesses = solver.compute_ranked_guesses()
        print('Top Guesses:')
        for g in ranked_guesses[-10:]:
            print(g)

        while True:
            maybe_info = parse_response(input('Enter feedback:'))
            if maybe_info is not None:
                break

        print('Parsed:')
        for elem in maybe_info:
            print(elem)

        solver.handle_information(maybe_info)

if __name__ == "__main__":
    main()
