
#include "compute_entropy.hh"

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <vectormath_exp.h>

#include <boost/dynamic_bitset.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <map>
#include <numeric>
#include <set>

namespace erick {

int num_categories(const int size) {
  int num_cats = 1;
  for (int i = 0; i < static_cast<int>(size); i++) {
    num_cats *= 3;
  }
  return num_cats;
}

template <typename T> T get_byte_mask(const int bit_mask) {
  std::array<int, sizeof(T) / 4> bytes;
  auto bytes_from_bits = [](int bits) -> int {
    int out = 0;
    for (int i = 3; i >= 0; i--) {
      out = (out << 8) | ((bits & (1 << i)) ? 0xFF : 0x00);
    }
    return out;
  };
  for (int byte_idx = 0; byte_idx < bytes.size(); byte_idx++) {
    bytes[byte_idx] = bytes_from_bits((bit_mask >> (byte_idx * 4)) & 0x0F);
  }
  if constexpr (bytes.size() == 4) {
    return std::apply(_mm_setr_epi32, bytes);
  } else if (bytes.size() == 8) {
    return std::apply(_mm256_setr_epi32, bytes);
  }
}

template <typename T> T get_byte_mask(const int size, const int start) {
  const int bit_mask = ((1 << size) - 1) << start;
  return get_byte_mask<T>(bit_mask);
}

std::vector<char> pack_strings(const std::vector<std::string> &strs) {
  std::vector<char> out;
  out.reserve(strs.size() * strs[0].size());
  for (const std::string &str : strs) {
    std::copy(str.begin(), str.end(), std::back_inserter(out));
  }
  return out;
}

boost::dynamic_bitset<>
compute_wrong_condition(const char letter, const int guess_size,
                        const std::vector<char> packed_answers) {
  boost::dynamic_bitset<> out(packed_answers.size() / guess_size);
  // Since we are using 256 bit registers, we can read up to 32/guess size words
  // at a time
  const int step_size = 32 / guess_size;
  const int num_words = packed_answers.size() / guess_size;
  const __m256i letter_reg = _mm256_set1_epi8(letter);
  std::vector<__m256i> word_masks;
  // Get the word masks
  for (int i = 0; i < step_size; i++) {
    word_masks.push_back(get_byte_mask<__m256i>(guess_size, i * guess_size));
  }

  for (int word_idx = 0; word_idx < num_words; word_idx += step_size) {
    const int start_idx = word_idx * guess_size;
    const __m256i answer_reg = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&packed_answers[start_idx]));
    const __m256i is_equal_reg = _mm256_cmpeq_epi8(answer_reg, letter_reg);
    for (int packed_word_idx = 0;
         packed_word_idx < step_size && word_idx + packed_word_idx < out.size();
         packed_word_idx++) {
      // Check if there were no matches for the current letter in this word
      const bool no_matches_in_word =
          _mm256_testz_si256(is_equal_reg, word_masks[packed_word_idx]);
      // This word belongs if there are no matches for the current letter in
      // this word
      out.set(word_idx + packed_word_idx, no_matches_in_word);
    }
  }
  return out;
}

boost::dynamic_bitset<>
compute_in_word_condition(const char letter, const int letter_idx,
                          const int guess_size,
                          const std::vector<char> packed_answers) {
  boost::dynamic_bitset<> out(packed_answers.size() / guess_size);
  // Since we are using 256 bit registers, we can read up to 32/guess size words
  // at a time
  const int step_size = 32 / guess_size;
  const int num_words = packed_answers.size() / guess_size;
  const __m256i letter_reg = _mm256_set1_epi8(letter);
  std::vector<__m256i> word_masks;
  // Get the word masks. Note that we remove the current letter from the match
  int bit_mask = ((1 << guess_size) - 1) & (~(1 << letter_idx));
  for (int i = 0; i < step_size; i++) {
    word_masks.push_back(get_byte_mask<__m256i>(bit_mask << (i * guess_size)));
  }

  for (int word_idx = 0; word_idx < num_words; word_idx += step_size) {
    const int start_idx = word_idx * guess_size;
    const __m256i answer_reg = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&packed_answers[start_idx]));
    const __m256i is_equal_reg = _mm256_cmpeq_epi8(answer_reg, letter_reg);
    for (int packed_word_idx = 0;
         packed_word_idx < step_size && word_idx + packed_word_idx < out.size();
         packed_word_idx++) {
      // Check if there were no matches for the current letter in this word
      const bool at_least_one_match_elsewhere_in_word =
          !_mm256_testz_si256(is_equal_reg, word_masks[packed_word_idx]);
      const bool no_match_in_position =
          letter != packed_answers[(word_idx + packed_word_idx) * guess_size +
                                   letter_idx];
      // This word belongs if there is at least one match for the current letter
      // in this word and it doesn't match at letter idx
      out.set(word_idx + packed_word_idx,
              at_least_one_match_elsewhere_in_word && no_match_in_position);
    }
  }
  return out;
}

boost::dynamic_bitset<>
compute_right_condition(const char letter, const int letter_idx,
                        const int guess_size,
                        const std::vector<char> packed_answers) {
  boost::dynamic_bitset<> out(packed_answers.size() / guess_size);
  for (int idx = letter_idx; idx < packed_answers.size(); idx += guess_size) {
    const int word_idx = idx / guess_size;
    out.set(word_idx, packed_answers[idx] == letter);
  }
  return out;
}

double compute_entropy(const std::string &guess,
                       const std::vector<std::string> &answers) {

  // Compute the counts
  const std::vector<int> counts = detail::compute_counts(guess, answers);

  // Compute entropy
  float entropy = 0.0;
  int start_idx = 0;
  const __m256i num_elements_int = _mm256_set1_epi32(answers.size());
  const __m256 num_elements_float = _mm256_cvtepi32_ps(num_elements_int);
  const __m256 ones = _mm256_set1_ps(1.0f);
  const __m256 normalizer_float = _mm256_div_ps(ones, num_elements_float);
  auto val_or_zero = [](const int val) {
    const float float_val = reinterpret_cast<const float &>(val);
    return std::isfinite(float_val) ? float_val : 0.0;
  };

  for (int idx = 0; idx < counts.size() / 8; idx++) {
    start_idx = 8 * idx;
    // load the integer counts into a register
    const __m256i packed_counts = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&(counts.data()[start_idx])));
    // convert the to a float
    const __m256 float_counts = _mm256_cvtepi32_ps(packed_counts);
    const __m256 probability = _mm256_mul_ps(float_counts, normalizer_float);
    const __m256 log_prob = log2(Vec8f(probability));
    const __m256 entropy_contrib = _mm256_mul_ps(probability, log_prob);

    const __m128 low_floats = _mm256_extractf128_ps(entropy_contrib, 0);
    const __m128 high_floats = _mm256_extractf128_ps(entropy_contrib, 1);
    const float sum_contrib = val_or_zero(_mm_extract_ps(low_floats, 0)) +
                              val_or_zero(_mm_extract_ps(low_floats, 1)) +
                              val_or_zero(_mm_extract_ps(low_floats, 2)) +
                              val_or_zero(_mm_extract_ps(low_floats, 3)) +
                              val_or_zero(_mm_extract_ps(high_floats, 0)) +
                              val_or_zero(_mm_extract_ps(high_floats, 1)) +
                              val_or_zero(_mm_extract_ps(high_floats, 2)) +
                              val_or_zero(_mm_extract_ps(high_floats, 3));
    entropy += -sum_contrib;
  }
  start_idx += 8;

  for (int idx = start_idx; idx < counts.size(); idx++) {
    if (counts[idx] > 0) {
      const float probability =
          static_cast<float>(counts[idx]) / answers.size();
      entropy += -probability * std::log2(probability);
    }
  }

  return entropy;
}

namespace detail {
void get_categories_for_index(const int i, const int size,
                              std::vector<Category> &categories) {
  categories.clear();
  int tmp = i;
  for (int idx = 0; idx < size; idx++) {
    categories.push_back(static_cast<Category>(tmp % 3));
    tmp = tmp / 3;
  }
}
std::vector<int> compute_counts(const std::string &guess,
                                const std::vector<std::string> &answers) {
  // This function computes whether or not a word belongs in anyone of the given
  // colorings of the word membership to a given coloring can be be determined
  // by ANDing the membership to the coloring of each character individually.
  // The first index is on the letter, the next is on the category and the last
  // is on whether the particular answer satisfies that condition;
  std::vector<std::vector<boost::dynamic_bitset<>>> conditions_satisfied;
  std::vector<char> packed_answers = pack_strings(answers);

  for (int letter_idx = 0; letter_idx < guess.size(); letter_idx++) {
    conditions_satisfied.push_back({});
    conditions_satisfied.back().push_back(compute_wrong_condition(
        guess[letter_idx], guess.size(), packed_answers));
    conditions_satisfied.back().push_back(compute_in_word_condition(
        guess[letter_idx], letter_idx, guess.size(), packed_answers));
    conditions_satisfied.back().push_back(compute_right_condition(
        guess[letter_idx], letter_idx, guess.size(), packed_answers));
  }

  // In word condition is set if the current letter does not exist at the
  // position indicated by the guess, but it does exist elsewhere in the answer.
  // This is almost correct. If the guess contains a repeated letter, the
  // answer has no repeated letters, and one of the repeated letters in the
  // guess is correct, then all repeated instances are marked as incorrect.
  // If the answer has repeated letters, then up to that many instances of the
  // letter are marked as being in the word or correct.

  // Create sets of repeated letters
  std::map<char, std::vector<int>> letter_sets;
  for (int i = 0; i < guess.size(); i++) {
    letter_sets[guess[i]].push_back(i);
  }

  for (const auto &letter_and_pos : letter_sets) {
    if (letter_and_pos.second.size() < 2) {
      continue;
    }
    // We have a guess that contains a double letter
    // Find all words that contain this letter by or-ing the right and in word
    // answers
    const auto all_with_letter = [&]() {
      boost::dynamic_bitset<> in_word(answers.size());
      for (const int pos : letter_and_pos.second) {
        in_word |= conditions_satisfied[pos][IN_WORD] |
                   conditions_satisfied[pos][RIGHT];
      }
      return in_word;
    }();

    for (int pos = all_with_letter.find_first(); pos != all_with_letter.npos;
         pos = all_with_letter.find_next(pos)) {
      // For each word identified, compute the number of letters of this value
      const std::string curr_answer = answers.at(pos);
      const int num_letter_in_answer = std::count(
          curr_answer.begin(), curr_answer.end(), letter_and_pos.first);
      // Count the number of this letter that is are correct
      const int num_right =
          std::count_if(letter_and_pos.second.begin(),
                        letter_and_pos.second.end(), [&](const int idx) {
                          return conditions_satisfied[idx][RIGHT].test(pos);
                        });
      // This means that we much have num_letter_in_answer - num_right in_words
      // mark the remaining ones as wrong
      int num_in_word_remaining = num_letter_in_answer - num_right;
      for (const int letter_idx : letter_and_pos.second) {
        if (conditions_satisfied[letter_idx][IN_WORD].test(pos)) {
          if (num_in_word_remaining) {
            num_in_word_remaining--;
          } else {
            conditions_satisfied[letter_idx][IN_WORD].reset(pos);
            conditions_satisfied[letter_idx][WRONG].set(pos);
          }
        }
      }
    }
  }

  // Now compute the counts per color
  std::vector<int> out(num_categories(guess.size()), 0);
  std::vector<Category> categories;
  for (int category_idx = 0; category_idx < out.size(); category_idx++) {
    get_categories_for_index(category_idx, guess.size(), categories);
    boost::dynamic_bitset<> curr_members =
        conditions_satisfied[0][categories[0]];
    for (int letter_idx = 1; letter_idx < guess.size(); letter_idx++) {
      curr_members = curr_members &
                     conditions_satisfied[letter_idx][categories[letter_idx]];
    }
    out.at(category_idx) = curr_members.count();
  }
  return out;
}
} // namespace detail
} // namespace erick
