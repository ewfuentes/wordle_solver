
from typing import List, Optional, NamedTuple, Tuple
from enum import Enum
import functools

class Info(Enum):
    WRONG = 0
    IN_WORD = 1
    RIGHT = 2

class CharInfo(NamedTuple):
    char: str
    info: Info

class GameState(NamedTuple):
    is_finished: bool
    is_winner: bool
    guesses_remaining: int
    information: List[List[CharInfo]]


def compute_information(guess: str, answer: str) -> Tuple[bool, List[CharInfo]]:
    assert len(guess) == len(answer), f'guess must be of length {len(answer)}'
    out = []
    is_winner = True
    for i, char in enumerate(guess):
        if answer[i] == char:
            out.append(CharInfo(char, Info.RIGHT))
        elif char in answer:
            out.append(CharInfo(char, Info.IN_WORD))
            is_winner = False
        else:
            out.append(CharInfo(char, Info.WRONG))
            is_winner = False
    return is_winner, out


class Wordle:
    def __init__(self, valid_guesses: List[str], answer: Optional[str] = None, possible_answers: Optional[List[str]] = None, num_guesses: int = 6):
        assert (answer is None) ^ (possible_answers is None), 'answer or possible_answers must be specified, but not both'
        if answer:
            self._answer = answer.lower()
        elif possible_answers:
            import random
            self._answer = random.choice(possible_answers)
        self._num_guesses = num_guesses

    def step(self, guess: str, game_state: Optional[GameState]) -> GameState:
        if game_state is None:
            game_state = GameState(is_finished=False, is_winner=False, guesses_remaining=self._num_guesses, information=[])

        if game_state.is_finished:
            return game_state

        try:
            is_winner, new_information = compute_information(guess.lower(), self._answer)
        except:
            print('Invalid Guess')
            return game_state

        guesses_remaining = game_state.guesses_remaining - 1
        is_finished = guesses_remaining == 0 or is_winner

        return GameState(is_finished=is_finished,
                         is_winner=is_winner,
                         guesses_remaining=guesses_remaining,
                         information = game_state.information + [new_information])

def handle_guess_lists(valid_guesses: List[str], possible_answers: List[str]):
    print(f'Valid guesses are between {valid_guesses[0]} and {valid_guesses[-1]}')

def get_input() -> str:
    return input(f'Enter a guess:').strip()

def main(num_digits: int, num_guesses: int, handle_guess_lists=handle_guess_lists, get_input=get_input, handle_result=None):
    import colorama
    colorama.init()
    color = {
        Info.WRONG: colorama.Back.WHITE + colorama.Fore.BLACK,
        Info.IN_WORD: colorama.Back.YELLOW + colorama.Fore.BLACK,
        Info.RIGHT: colorama.Back.GREEN + colorama.Fore.BLACK,
        'Reset': colorama.Back.RESET + colorama.Fore.RESET,
    }
    def info_to_str(info: List[CharInfo]) -> str:
        out = ''
        for char_info in info:
            out += color[char_info.info] + char_info.char
        out += color['Reset']
        return out

    valid_guesses = [str(x).zfill(num_digits) for x in range(10**num_digits)]

    if handle_guess_lists:
        handle_guess_lists(valid_guesses, valid_guesses)

    game = Wordle(valid_guesses, possible_answers=valid_guesses, num_guesses=num_guesses)

    print(f'If the guess has a character in the correct position, it will appear in {color[Info.RIGHT]}GREEN{color["Reset"]}')
    print(f'If the guess has a character in the wrong position but in the word, it will appear in {color[Info.IN_WORD]}YELLOW{color["Reset"]}')
    print(f'If the guess has a character that is not in the answer, it will appear in {color[Info.WRONG]}WHITE{color["Reset"]}')

    state = None
    while state is None or not state.is_finished:
        guess = get_input()
        state = game.step(guess, state)
        if handle_result:
            handle_result(state.information[-1])
        if not state.is_finished:
            print(f'Guesses Remaining: {state.guesses_remaining} Result: {info_to_str(state.information[-1])}')
    print(f'Winner! The answer was {info_to_str(state.information[-1])}.'
          if state.is_winner else f'Sorry! The answer was {game._answer}. Try Again!')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Wordle, but with numbers')
    parser.add_argument('--num_digits', default=2, type=int)
    parser.add_argument('--num_guesses', default=10, type=int)
    args = parser.parse_args()

    main(args.num_digits, args.num_guesses)
