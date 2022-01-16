
#include "compute_entropy.hh"

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <ios>
#include <iostream>

namespace erick {
enum Category { WRONG = 0, IN_WORD = 1, RIGHT = 2 };

int num_categories(const int size) {
  int num_cats = 1;
  for (int i = 0; i < static_cast<int>(size); i++) {
    num_cats *= 3;
  }
  return num_cats;
}

std::ostream &operator<<(std::ostream &oss, const __m128i &reg) {
  oss << std::hex;
  oss << _mm_extract_epi8(reg, 0) << " ";
  oss << _mm_extract_epi8(reg, 1) << " ";
  oss << _mm_extract_epi8(reg, 2) << " ";
  oss << _mm_extract_epi8(reg, 3) << " ";
  oss << _mm_extract_epi8(reg, 4) << " ";
  oss << _mm_extract_epi8(reg, 5) << " ";
  oss << _mm_extract_epi8(reg, 6) << " ";
  oss << _mm_extract_epi8(reg, 7) << " ";
  oss << _mm_extract_epi8(reg, 8) << " ";
  oss << _mm_extract_epi8(reg, 9) << " ";
  oss << _mm_extract_epi8(reg, 10) << " ";
  oss << _mm_extract_epi8(reg, 11) << " ";
  oss << _mm_extract_epi8(reg, 12) << " ";
  oss << _mm_extract_epi8(reg, 13) << " ";
  oss << _mm_extract_epi8(reg, 14) << " ";
  oss << _mm_extract_epi8(reg, 15) << " ";
  oss << std::dec;
  return oss;
}

template <typename T>
std::ostream &operator<<(std::ostream &oss, const std::vector<T> &vec) {
  for (const auto &item : vec) {
    oss << item << " ";
  }
  return oss;
}

__m128i get_size_mask(const int i) {
  const __m128i all_ones = _mm_set1_epi8(0xFF);
  switch (i) {
  case 15:
    return _mm_bsrli_si128(all_ones, 1);
  case 14:
    return _mm_bsrli_si128(all_ones, 2);
  case 13:
    return _mm_bsrli_si128(all_ones, 3);
  case 12:
    return _mm_bsrli_si128(all_ones, 4);
  case 11:
    return _mm_bsrli_si128(all_ones, 5);
  case 10:
    return _mm_bsrli_si128(all_ones, 6);
  case 9:
    return _mm_bsrli_si128(all_ones, 7);
  case 8:
    return _mm_bsrli_si128(all_ones, 8);
  case 7:
    return _mm_bsrli_si128(all_ones, 9);
  case 6:
    return _mm_bsrli_si128(all_ones, 10);
  case 5:
    return _mm_bsrli_si128(all_ones, 11);
  case 4:
    return _mm_bsrli_si128(all_ones, 12);
  case 3:
    return _mm_bsrli_si128(all_ones, 13);
  case 2:
    return _mm_bsrli_si128(all_ones, 14);
  case 1:
    return _mm_bsrli_si128(all_ones, 15);
  case 0:
    return _mm_bsrli_si128(all_ones, 16);
  default:
    return all_ones;
  }
}

bool word_matches_guess_categories(const std::string &guess,
                                   const std::string &word,
                                   const std::vector<Category> &categories) {
  // Load the word data into a register. Note that this reads 16 bytes, which
  // likely extends beyond the end of the word
  __m128i word_reg = _mm_set1_epi8(0);
  word_reg = *reinterpret_cast<const __m128i *>(word.data());
  // Create a mask for the bytes that are valid. Note that valid bytes are set
  // to zero
  const __m128i word_mask = get_size_mask(word.size());
  for (int i = 0; i < static_cast<int>(categories.size()); i++) {
    const char letter = guess[i];
    // Pack the current character into all bytes of a register
    const __m128i letter_reg = _mm_set1_epi8(guess[i]);
    // For the current letter in the guess...
    if (categories[i] == WRONG || categories[i] == IN_WORD) {
      // If the category is wrong or in word, find if the current letter in
      // guess is in the word
      const __m128i is_equal_reg = _mm_cmpeq_epi8(word_reg, letter_reg);
      const bool is_letter_in_answer =
          !_mm_test_all_zeros(is_equal_reg, word_mask);

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
    get_categories_for_index(idx, guess.size(), categories);
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
