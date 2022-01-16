
#include "compute_entropy.hh"

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <vectormath_exp.h>

#include <boost/dynamic_bitset.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <ios>
#include <iomanip>
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

bool word_matches_guess_categories(const std::string &guess,
                                   const std::string &word,
                                   const std::vector<Category> &categories) {
  // Load the word data into a register. Note that this reads 16 bytes, which
  // likely extends beyond the end of the word
  __m128i word_reg = _mm_set1_epi8(0);
  word_reg = *reinterpret_cast<const __m128i *>(word.data());
  // Create a mask for the bytes that are valid. Note that valid bytes are set
  // to zero
  const __m128i word_mask = get_byte_mask<__m128i>(word.size(), 0);
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

std::ostream &operator<<(std::ostream &oss, const __m256i &reg) {
  oss << "print int " << std::hex;
  oss << std::setw(2) << _mm256_extract_epi8(reg, 0) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 1) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 2) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 3) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 4) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 5) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 6) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 7) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 8 + 0) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 8 + 1) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 8 + 2) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 8 + 3) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 8 + 4) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 8 + 5) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 8 + 6) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 8 + 7) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 16 + 0) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 16 + 1) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 16 + 2) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 16 + 3) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 16 + 4) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 16 + 5) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 16 + 6) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 16 + 7) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 24 + 0) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 24 + 1) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 24 + 2) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 24 + 3) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 24 + 4) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 24 + 5) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 24 + 6) << " ";
  oss << std::setw(2) << _mm256_extract_epi8(reg, 24 + 7) << std::dec;
  return oss;
}

std::ostream &operator<<(std::ostream &oss, const __m256 &reg) {
  const __m128 low_floats = _mm256_extractf128_ps(reg, 0);
  const __m128 high_floats = _mm256_extractf128_ps(reg, 1);
  int tmp;
  oss << "print float ";
  tmp = _mm_extract_ps(low_floats, 0);
  oss << reinterpret_cast<float &>(tmp) << " ";
  tmp = _mm_extract_ps(low_floats, 1);
  oss << reinterpret_cast<float &>(tmp) << " ";
  tmp = _mm_extract_ps(low_floats, 2);
  oss << reinterpret_cast<float &>(tmp) << " ";
  tmp = _mm_extract_ps(low_floats, 3);
  oss << reinterpret_cast<float &>(tmp) << " ";
  tmp = _mm_extract_ps(high_floats, 0);
  oss << reinterpret_cast<float &>(tmp) << " ";
  tmp = _mm_extract_ps(high_floats, 1);
  oss << reinterpret_cast<float &>(tmp) << " ";
  tmp = _mm_extract_ps(high_floats, 2);
  oss << reinterpret_cast<float &>(tmp) << " ";
  tmp = _mm_extract_ps(high_floats, 3);
  oss << reinterpret_cast<float &>(tmp);
  return oss;
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
          letter !=
          packed_answers[(word_idx + packed_word_idx) * guess_size +
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

  // Now compute the counts per color
  std::vector<int> out(num_categories(guess.size()), 0);
  std::vector<Category> categories;
  for (int category_idx = 0; category_idx < out.size(); category_idx++) {
    get_categories_for_index(category_idx, guess.size(), categories);
    boost::dynamic_bitset<> curr_members = conditions_satisfied[0][categories[0]];
    for (int letter_idx = 1; letter_idx < guess.size(); letter_idx++) {
      curr_members = curr_members & conditions_satisfied[letter_idx][categories[letter_idx]];
    }
    out.at(category_idx) = curr_members.count();
  }
  return out;
}

double compute_entropy(const std::string &guess,
                       const std::vector<std::string> &answers) {

  // Compute the counts
  const std::vector<int> counts = compute_counts(guess, answers);

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

  for (int idx = start_idx; start_idx < counts.size(); start_idx++) {
    if (counts[idx] > 0) {
      const float probability =
          static_cast<float>(counts[idx]) / answers.size();
      entropy += -probability * std::log2(probability);
    }
  }

  return entropy;
}
} // namespace erick
