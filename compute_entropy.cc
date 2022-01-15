
#include "compute_entropy.hh"

#include <algorithm>
#include <cmath>

namespace erick {
enum Category { WRONG = 0, IN_WORD = 1, RIGHT = 2 };

int num_categories(const int size) {
  int num_cats = 1;
  for (int i = 0; i < static_cast<int>(size); i++) {
    num_cats *= 3;
  }
  return num_cats;
}

bool word_matches_guess_categories(const std::string &guess,
                                   const std::string &word,
                                   const std::vector<Category> &categories) {
  for (int i = 0; i < static_cast<int>(categories.size()); i++) {
    const char letter = guess[i];
    // For the current letter in the guess...
    if (categories[i] == WRONG || categories[i] == IN_WORD) {
      // If the category is wrong or in word, find if the current letter in
      // guess is in the word
      const bool is_letter_in_answer = std::any_of(
          word.begin(), word.end(), [&](const auto &c) { return c == letter; });
      if (categories[i] == WRONG && is_letter_in_answer) {
        // If it is and the category is "WRONG" then this word can't be in this
        // category
        return false;
      } else if (categories[i] == IN_WORD &&
                 (!is_letter_in_answer || word[i] == letter)) {
        // If the category is "IN_WORD" and the letter is not in the answer,
        // this word can't be in the category If the category is "IN_WORD" and
        // the letter is in the current position, this word can't be in the
        // category
        return false;
      }
    } else if (categories[i] == RIGHT && letter != word[i]) {
      // If the category is "RIGHT" and we don't have a match, this can't be in
      // the category
      return false;
    }
  }
  // All categories are matched, return true
  return true;
}

int compute_counts(const std::string &guess,
                   const std::vector<std::string> &answers,
                   const std::vector<Category> &categories) {
  return std::count_if(answers.begin(), answers.end(), [&](const auto &word) {
    return word_matches_guess_categories(guess, word, categories);
  });
}

void get_categories_for_index(const int i, const int size,
                              std::vector<Category> &categories) {
  categories.clear();
  int tmp = i;
  for (int idx = 0; idx < size; idx++) {
    categories.push_back(static_cast<Category>(tmp % 3));
    tmp = tmp / 3;
  }
}

double compute_entropy(const std::string &guess,
                       const std::vector<std::string> &answers) {
  std::vector<int> counts(num_categories(guess.size()), 0);

  // Compute the counts
  std::vector<Category> categories;
  for (int idx = 0; idx < counts.size(); idx++) {
    get_categories_for_index(idx, counts.size(), categories);
    counts[idx] = compute_counts(guess, answers, categories);
  }

  // Compute entropy
  double entropy = 0.0;
  for (const int count : counts) {
    if (count > 0) {
      const double probability = static_cast<double>(count) / answers.size();
      entropy += -probability * std::log2(probability);
    }
  }

  return entropy;
}
} // namespace erick
