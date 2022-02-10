
#pragma once

#include <string>
#include <vector>

namespace erick {
double compute_entropy(const std::string &guess,
                       const std::vector<std::string> &answers);
namespace detail {
enum Category { WRONG = 0, IN_WORD = 1, RIGHT = 2 };
void get_categories_for_index(const int i, const int size,
                              std::vector<Category> &categories);
std::vector<int> compute_counts(const std::string &guess,
                                const std::vector<std::string> &answers);
} // namespace detail
} // namespace erick
