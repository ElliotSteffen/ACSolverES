#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif

namespace py = pybind11;

namespace {

std::uint64_t current_resident_memory_bytes() {
#ifdef __linux__
    std::ifstream statm("/proc/self/statm");
    std::uint64_t total_pages = 0;
    std::uint64_t resident_pages = 0;
    if (!(statm >> total_pages >> resident_pages)) return 0;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 0;
    return resident_pages * static_cast<std::uint64_t>(page_size);
#else
    return 0;
#endif
}

int letter_rank(char letter) {
    switch (letter) {
        case 'Y': return 0;
        case 'y': return 1;
        case 'X': return 2;
        case 'x': return 3;
        default: throw std::invalid_argument("Words may contain only x, X, y, Y.");
    }
}

char inverse_letter(char letter) {
    switch (letter) {
        case 'x': return 'X';
        case 'X': return 'x';
        case 'y': return 'Y';
        case 'Y': return 'y';
        default: throw std::invalid_argument("Words may contain only x, X, y, Y.");
    }
}

bool are_inverse(char left, char right) {
    return inverse_letter(left) == right;
}

bool solver_less(const std::string& left, const std::string& right) {
    const auto limit = std::min(left.size(), right.size());
    for (std::size_t index = 0; index < limit; ++index) {
        const int left_rank = letter_rank(left[index]);
        const int right_rank = letter_rank(right[index]);
        if (left_rank != right_rank) {
            return left_rank < right_rank;
        }
    }
    return left.size() < right.size();
}

std::string inverse_word(const std::string& word) {
    std::string result;
    result.reserve(word.size());
    for (auto iterator = word.rbegin(); iterator != word.rend(); ++iterator) {
        result.push_back(inverse_letter(*iterator));
    }
    return result;
}

std::string reduce_word(const std::string& word) {
    std::string stack;
    stack.reserve(word.size());
    for (char letter : word) {
        if (!stack.empty() && are_inverse(stack.back(), letter)) {
            stack.pop_back();
        } else {
            stack.push_back(letter);
        }
    }

    std::size_t left = 0;
    std::size_t right = stack.size();
    while (right > left + 1 && are_inverse(stack[left], stack[right - 1])) {
        ++left;
        --right;
    }
    return stack.substr(left, right - left);
}

std::size_t minimal_rotation_index(const std::string& word) {
    const std::size_t size = word.size();
    if (size < 2) {
        return 0;
    }
    const std::string doubled = word + word;
    std::size_t first = 0;
    std::size_t second = 1;
    std::size_t offset = 0;
    while (first < size && second < size && offset < size) {
        const int first_rank = letter_rank(doubled[first + offset]);
        const int second_rank = letter_rank(doubled[second + offset]);
        if (first_rank == second_rank) {
            ++offset;
            continue;
        }
        if (first_rank > second_rank) {
            first += offset + 1;
            if (first <= second) {
                first = second + 1;
            }
        } else {
            second += offset + 1;
            if (second <= first) {
                second = first + 1;
            }
        }
        offset = 0;
    }
    return std::min(first, second);
}

std::string minimal_rotation(const std::string& word) {
    if (word.empty()) {
        return word;
    }
    const std::size_t index = minimal_rotation_index(word);
    return word.substr(index) + word.substr(0, index);
}

std::string canonical_word(const std::string& word) {
    const std::string forward = minimal_rotation(word);
    const std::string inverse = minimal_rotation(inverse_word(word));
    return solver_less(forward, inverse) ? forward : inverse;
}

std::pair<std::string, std::string> canonical_pair(
    const std::string& first,
    const std::string& second
) {
    std::string canonical_first = canonical_word(first);
    std::string canonical_second = canonical_word(second);
    if (
        canonical_first.size() > canonical_second.size()
        || (
            canonical_first.size() == canonical_second.size()
            && !solver_less(canonical_first, canonical_second)
        )
    ) {
        std::swap(canonical_first, canonical_second);
    }
    return {std::move(canonical_first), std::move(canonical_second)};
}

std::vector<std::pair<std::string, std::string>> standard_successors(
    const std::string& first,
    const std::string& second,
    int max_total_length
) {
    if (max_total_length <= 0) {
        throw std::invalid_argument("max_total_length must be positive.");
    }
    for (char letter : first) {
        letter_rank(letter);
    }
    for (char letter : second) {
        letter_rank(letter);
    }

    std::vector<std::pair<std::string, std::string>> results;
    const std::vector<std::string> candidates = {second, inverse_word(second)};
    const std::string reduced_first = reduce_word(first);
    const std::string reduced_second = reduce_word(second);
    results.reserve(4 * first.size() * second.size());

    for (const std::string& candidate : candidates) {
        for (std::size_t first_rotation = 0; first_rotation < first.size(); ++first_rotation) {
            const std::size_t first_start =
                (first.size() - first_rotation) % first.size();
            const std::size_t first_last =
                (first_start + first.size() - 1) % first.size();
            for (
                std::size_t candidate_rotation = 0;
                candidate_rotation < candidate.size();
                ++candidate_rotation
            ) {
                const std::size_t candidate_start =
                    (candidate.size() - candidate_rotation) % candidate.size();
                if (!are_inverse(first[first_last], candidate[candidate_start])) {
                    continue;
                }

                std::string neighbor;
                neighbor.reserve(first.size() + candidate.size());
                for (std::size_t offset = 0; offset < first.size(); ++offset) {
                    neighbor.push_back(first[(first_start + offset) % first.size()]);
                }
                for (std::size_t offset = 0; offset < candidate.size(); ++offset) {
                    neighbor.push_back(
                        candidate[(candidate_start + offset) % candidate.size()]
                    );
                }
                const std::string reduced_neighbor = reduce_word(neighbor);
                if (
                    reduced_neighbor.size() + reduced_second.size()
                    < static_cast<std::size_t>(max_total_length)
                ) {
                    results.push_back(canonical_pair(reduced_neighbor, reduced_second));
                }

                if (
                    reduced_first.size() + reduced_neighbor.size()
                    < static_cast<std::size_t>(max_total_length)
                ) {
                    results.push_back(canonical_pair(reduced_first, reduced_neighbor));
                }
            }
        }
    }
    return results;
}

struct PackedWord {
    std::uint64_t low = 0;
    std::uint64_t high = 0;
    std::uint16_t length = 0;

    bool operator==(const PackedWord& other) const noexcept {
        return low == other.low && high == other.high && length == other.length;
    }
};

struct PackedState {
    PackedWord first;
    PackedWord second;

    bool operator==(const PackedState& other) const noexcept {
        return first == other.first && second == other.second;
    }
};

std::uint64_t mix_hash(std::uint64_t value) noexcept {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

struct PackedStateHash {
    std::size_t operator()(const PackedState& state) const noexcept {
        std::uint64_t hash = mix_hash(state.first.low);
        hash ^= mix_hash(state.first.high + 0x9e3779b97f4a7c15ULL);
        hash ^= mix_hash(state.second.low + 0x243f6a8885a308d3ULL);
        hash ^= mix_hash(state.second.high + 0x13198a2e03707344ULL);
        hash ^= (static_cast<std::uint64_t>(state.first.length) << 16)
            | state.second.length;
        return static_cast<std::size_t>(hash);
    }
};

std::uint8_t encode_letter(char letter) {
    switch (letter) {
        case 'x': return 0;
        case 'X': return 1;
        case 'y': return 2;
        case 'Y': return 3;
        default: throw std::invalid_argument("Words may contain only x, X, y, Y.");
    }
}

char decode_letter(std::uint8_t code) {
    static constexpr std::array<char, 4> letters = {'x', 'X', 'y', 'Y'};
    return letters[code & 3U];
}

PackedWord pack_word(const std::string& word) {
    if (word.size() > 64) {
        throw std::invalid_argument("Compact Dual GS supports relators of at most 64 letters.");
    }
    PackedWord packed;
    packed.length = static_cast<std::uint16_t>(word.size());
    for (std::size_t index = 0; index < word.size(); ++index) {
        const auto code = static_cast<std::uint64_t>(encode_letter(word[index]));
        if (index < 32) {
            packed.low |= code << (2 * index);
        } else {
            packed.high |= code << (2 * (index - 32));
        }
    }
    return packed;
}

std::string unpack_word(const PackedWord& packed) {
    std::string word;
    word.resize(packed.length);
    for (std::size_t index = 0; index < packed.length; ++index) {
        const auto storage = index < 32 ? packed.low : packed.high;
        const auto shift = 2 * (index < 32 ? index : index - 32);
        word[index] = decode_letter(static_cast<std::uint8_t>((storage >> shift) & 3U));
    }
    return word;
}

PackedState pack_state(const std::string& first, const std::string& second) {
    return {pack_word(first), pack_word(second)};
}

std::pair<std::string, std::string> unpack_state(const PackedState& state) {
    return {unpack_word(state.first), unpack_word(state.second)};
}

std::uint16_t packed_length(const PackedState& state) noexcept {
    return static_cast<std::uint16_t>(state.first.length + state.second.length);
}

struct MassMatrix {
    int a = 0;
    int b = 0;
    int c = 0;
    int d = 0;

    bool operator==(const MassMatrix& other) const noexcept {
        return a == other.a && b == other.b && c == other.c && d == other.d;
    }
};

struct MassMatrixHash {
    std::size_t operator()(const MassMatrix& matrix) const noexcept {
        std::uint64_t packed = static_cast<std::uint64_t>(matrix.a + 256);
        packed = (packed << 10) | static_cast<std::uint64_t>(matrix.b + 256);
        packed = (packed << 10) | static_cast<std::uint64_t>(matrix.c + 256);
        packed = (packed << 10) | static_cast<std::uint64_t>(matrix.d + 256);
        return static_cast<std::size_t>(mix_hash(packed));
    }
};

constexpr int MASS_SCORE_BOUND = 25;
constexpr std::uint16_t MASS_SCORE_MAX = 7;
constexpr std::uint16_t MASS_SCORE_FALLBACK = MASS_SCORE_MAX + 1;
constexpr std::uint16_t MASS_LEXICOGRAPHIC_BASE = MASS_SCORE_FALLBACK + 1;

bool mass_within_bound(const MassMatrix& matrix) noexcept {
    return std::abs(matrix.a) <= MASS_SCORE_BOUND
        && std::abs(matrix.b) <= MASS_SCORE_BOUND
        && std::abs(matrix.c) <= MASS_SCORE_BOUND
        && std::abs(matrix.d) <= MASS_SCORE_BOUND;
}

const std::unordered_map<MassMatrix, std::uint8_t, MassMatrixHash>& mass_score_table() {
    static const auto scores = [] {
        std::unordered_map<MassMatrix, std::uint8_t, MassMatrixHash> distances;
        distances.reserve(6'388);
        std::queue<MassMatrix> frontier;
        const MassMatrix identity{1, 0, 0, 1};
        distances.emplace(identity, 0);
        frontier.push(identity);

        auto visit = [&](const MassMatrix& neighbor, std::uint8_t distance) {
            if (!mass_within_bound(neighbor)) return;
            if (distances.emplace(neighbor, distance).second) frontier.push(neighbor);
        };

        while (!frontier.empty()) {
            const MassMatrix matrix = frontier.front();
            frontier.pop();
            const auto next_distance = static_cast<std::uint8_t>(distances.at(matrix) + 1);
            visit(
                {matrix.a + matrix.c, matrix.b + matrix.d, matrix.c, matrix.d},
                next_distance
            );
            visit(
                {matrix.a - matrix.c, matrix.b - matrix.d, matrix.c, matrix.d},
                next_distance
            );
            visit(
                {matrix.a, matrix.b, matrix.c + matrix.a, matrix.d + matrix.b},
                next_distance
            );
            visit(
                {matrix.a, matrix.b, matrix.c - matrix.a, matrix.d - matrix.b},
                next_distance
            );
            for (int exponent = -6; exponent <= 6; ++exponent) {
                if (exponent == 0) continue;
                visit(
                    {
                        matrix.a,
                        matrix.b + exponent * matrix.a,
                        matrix.c,
                        matrix.d + exponent * matrix.c,
                    },
                    next_distance
                );
                visit(
                    {
                        matrix.a + exponent * matrix.b,
                        matrix.b,
                        matrix.c + exponent * matrix.d,
                        matrix.d,
                    },
                    next_distance
                );
            }
        }
        if (distances.size() != 6'388) {
            throw std::runtime_error("Compact abelian mass table failed completeness validation.");
        }
        return distances;
    }();
    return scores;
}

std::pair<int, int> packed_exponent_sums(const PackedWord& word) noexcept {
    int x_mass = 0;
    int y_mass = 0;
    for (std::size_t index = 0; index < word.length; ++index) {
        const auto storage = index < 32 ? word.low : word.high;
        const auto shift = 2 * (index < 32 ? index : index - 32);
        switch ((storage >> shift) & 3U) {
            case 0: ++x_mass; break;
            case 1: --x_mass; break;
            case 2: ++y_mass; break;
            case 3: --y_mass; break;
        }
    }
    return {x_mass, y_mass};
}

std::uint16_t packed_mass_score(const PackedState& state) {
    const auto first = packed_exponent_sums(state.first);
    const auto second = packed_exponent_sums(state.second);
    const auto& scores = mass_score_table();
    std::uint16_t best = MASS_SCORE_FALLBACK;
    for (const int first_sign : {-1, 1}) {
        for (const int second_sign : {-1, 1}) {
            const MassMatrix direct{
                first_sign * first.first,
                first_sign * first.second,
                second_sign * second.first,
                second_sign * second.second,
            };
            const MassMatrix swapped{direct.c, direct.d, direct.a, direct.b};
            const auto direct_score = scores.find(direct);
            if (direct_score != scores.end()) best = std::min<std::uint16_t>(best, direct_score->second);
            const auto swapped_score = scores.find(swapped);
            if (swapped_score != scores.end()) best = std::min<std::uint16_t>(best, swapped_score->second);
        }
    }
    return best;
}

enum class SubQueueScoreMode : std::uint8_t {
    Length,
    LengthThenMatrix,
    LengthPlusMatrix,
};

SubQueueScoreMode parse_sub_queue_score_mode(const std::string& value) {
    if (value == "length") return SubQueueScoreMode::Length;
    if (value == "length_then_matrix") return SubQueueScoreMode::LengthThenMatrix;
    if (value == "length_plus_matrix") return SubQueueScoreMode::LengthPlusMatrix;
    throw std::invalid_argument(
        "sub_queue_score_mode must be length, length_then_matrix, or length_plus_matrix."
    );
}

std::uint16_t packed_sub_queue_score(
    const PackedState& state,
    SubQueueScoreMode mode
) {
    const auto length = packed_length(state);
    if (mode == SubQueueScoreMode::Length) return length;
    const auto matrix_score = packed_mass_score(state);
    if (mode == SubQueueScoreMode::LengthThenMatrix) {
        return static_cast<std::uint16_t>(length * MASS_LEXICOGRAPHIC_BASE + matrix_score);
    }
    return static_cast<std::uint16_t>(length + matrix_score);
}

bool packed_solver_less(const PackedState& left, const PackedState& right) {
    const auto left_words = unpack_state(left);
    const auto right_words = unpack_state(right);
    if (left.first.length != right.first.length) {
        return left.first.length < right.first.length;
    }
    if (left_words.first != right_words.first) {
        return solver_less(left_words.first, right_words.first);
    }
    if (left.second.length != right.second.length) {
        return left.second.length < right.second.length;
    }
    return solver_less(left_words.second, right_words.second);
}

std::string apply_signed_generator_automorphism(
    const std::string& word,
    char x_image,
    char y_image
) {
    std::string transformed;
    transformed.reserve(word.size());
    for (const char letter : word) {
        switch (letter) {
            case 'x': transformed.push_back(x_image); break;
            case 'X': transformed.push_back(inverse_letter(x_image)); break;
            case 'y': transformed.push_back(y_image); break;
            case 'Y': transformed.push_back(inverse_letter(y_image)); break;
            default: throw std::invalid_argument("Words may contain only x, X, y, Y.");
        }
    }
    return transformed;
}

py::tuple automorphic_equivalence_key_native(
    const std::string& first,
    const std::string& second
) {
    static constexpr std::array<char, 4> images = {'x', 'X', 'y', 'Y'};
    bool have_best = false;
    PackedState best;
    for (const char x_image : images) {
        const std::array<char, 2> y_images =
            (x_image == 'x' || x_image == 'X')
            ? std::array<char, 2>{'y', 'Y'}
            : std::array<char, 2>{'x', 'X'};
        for (const char y_image : y_images) {
            const auto canonical = canonical_pair(
                apply_signed_generator_automorphism(first, x_image, y_image),
                apply_signed_generator_automorphism(second, x_image, y_image)
            );
            const PackedState candidate = pack_state(canonical.first, canonical.second);
            if (!have_best || packed_solver_less(candidate, best)) {
                best = candidate;
                have_best = true;
            }
        }
    }
    const auto words = unpack_state(best);
    return py::make_tuple(words.first, words.second);
}

using StateId = std::uint32_t;
constexpr StateId NO_STATE = std::numeric_limits<StateId>::max();

enum class MoveKind : std::uint8_t {
    Start = 0,
    Standard = 1,
    Cov = 2,
    CompleteCov = 3,
};

struct StateRecord {
    PackedState state;
    StateId parent = NO_STATE;
    std::uint32_t depth = 0;
    std::uint32_t standard_gap = 0;
    MoveKind move_kind = MoveKind::Start;
    bool alive = true;
    bool sub_expanded = false;
    bool cov_queued = false;
    bool cov_expanded = false;
    bool complete_cov_queued = false;
    bool complete_cov_expanded = false;
};

struct Priority {
    std::uint16_t score = 0;
    std::int64_t depth_priority = 0;
    std::uint64_t insertion = 0;
};

bool priority_less(const Priority& left, const Priority& right) noexcept {
    if (left.score != right.score) return left.score < right.score;
    if (left.depth_priority != right.depth_priority) {
        return left.depth_priority < right.depth_priority;
    }
    return left.insertion < right.insertion;
}

struct QueueEntry {
    Priority priority;
    StateId state_id = NO_STATE;
};

struct QueueGreater {
    bool operator()(const QueueEntry& left, const QueueEntry& right) const noexcept {
        return priority_less(right.priority, left.priority);
    }
};

class MinQueue
    : public std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueGreater> {
public:
    using Base = std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueGreater>;
    using Base::Base;

    const std::vector<QueueEntry>& entries() const noexcept {
        return this->c;
    }

    void restore_entries(std::vector<QueueEntry> entries) {
        this->c = std::move(entries);
        std::make_heap(this->c.begin(), this->c.end(), this->comp);
    }
};

py::tuple state_tuple(const PackedState& state) {
    const auto words = unpack_state(state);
    return py::make_tuple(words.first, words.second);
}

bool is_composite_count(int value) noexcept {
    if (value < 4) return false;
    for (int divisor = 2; divisor * divisor <= value; ++divisor) {
        if (value % divisor == 0) return true;
    }
    return false;
}

bool complete_cov_composite_eligible(const PackedState& state) {
    const auto words = unpack_state(state);
    for (const auto& word : {words.first, words.second}) {
        int x_count = 0;
        int y_count = 0;
        for (const char letter : word) {
            x_count += letter == 'x' || letter == 'X';
            y_count += letter == 'y' || letter == 'Y';
        }
        if (is_composite_count(x_count - 1) || is_composite_count(y_count - 1)) {
            return true;
        }
    }
    return false;
}

struct TripleSearchCounters {
    std::uint64_t standard_generated_count = 0;
    std::uint64_t standard_enqueued_count = 0;
    std::uint64_t auto_generated_count = 0;
    std::uint64_t auto_valid_count = 0;
    std::uint64_t auto_enqueued_count = 0;
    std::uint64_t auto_generation_call_count = 0;
    std::uint64_t initial_auto_seed_enqueued_count = 0;
    std::uint64_t initial_complete_seed_enqueued_count = 0;
    std::uint64_t complete_generation_call_count = 0;
    std::uint64_t complete_neighbor_count = 0;
    std::uint64_t complete_seen_neighbor_count = 0;
    std::uint64_t complete_direct_enqueued_count = 0;
    std::uint64_t complete_prefilter_accepted_count = 0;
    std::uint64_t complete_prefilter_rejected_count = 0;
    std::uint64_t complete_ineligible_pop_count = 0;
    std::uint64_t auto_queued_count = 0;
    std::uint64_t auto_queue_eligible_count = 0;
    std::uint64_t auto_queue_distance_blocked_count = 0;
    std::uint64_t complete_queued_count = 0;
    std::uint64_t sub_pop_count = 0;
    std::uint64_t auto_pop_count = 0;
    std::uint64_t complete_pop_count = 0;
    std::uint64_t nodes_used = 0;
    std::uint64_t frontier_prune_count = 0;
    std::uint64_t frontier_states_pruned = 0;
    std::uint64_t frontier_peak = 0;
    std::uint64_t checkpoint_count = 0;
    std::uint64_t last_checkpoint_nodes = 0;
};

struct TripleCheckpointConfig {
    PackedState start_state;
    std::uint64_t total_node_budget = 0;
    std::int32_t max_total_length = 0;
    std::int32_t auto_preference_threshold = 0;
    std::int32_t complete_preference_threshold = 0;
    std::int32_t min_standard_moves_between_auto = 0;
    std::int64_t max_live_frontier_states = 0;
    bool seed_initial_auto_neighbors = false;
    bool seed_initial_complete_neighbors = false;
    bool deepest = false;
};

constexpr std::array<char, 8> TRIPLE_CHECKPOINT_MAGIC = {
    'T', 'R', 'P', 'G', 'S', 'C', 'P', '1'
};
constexpr std::uint32_t TRIPLE_CHECKPOINT_VERSION = 1;

template <typename Value>
void checkpoint_write(std::ostream& stream, const Value& value) {
    static_assert(std::is_trivially_copyable<Value>::value, "POD required");
    stream.write(reinterpret_cast<const char*>(&value), sizeof(Value));
    if (!stream) throw std::runtime_error("Failed while writing Triple GS checkpoint.");
}

template <typename Value>
Value checkpoint_read(std::istream& stream) {
    static_assert(std::is_trivially_copyable<Value>::value, "POD required");
    Value value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(Value));
    if (!stream) throw std::runtime_error("Truncated Triple GS checkpoint.");
    return value;
}

void checkpoint_write_bytes(std::ostream& stream, const std::string& bytes) {
    checkpoint_write(stream, static_cast<std::uint64_t>(bytes.size()));
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("Failed while writing Triple GS checkpoint bytes.");
}

std::string checkpoint_read_bytes(std::istream& stream) {
    const auto size = checkpoint_read<std::uint64_t>(stream);
    if (size > (1ULL << 34)) {
        throw std::runtime_error("Triple GS checkpoint contains an invalid byte count.");
    }
    std::string bytes(static_cast<std::size_t>(size), '\0');
    stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("Truncated Triple GS checkpoint bytes.");
    return bytes;
}

void checkpoint_write_state(std::ostream& stream, const PackedState& state) {
    for (const PackedWord* word : {&state.first, &state.second}) {
        checkpoint_write(stream, word->low);
        checkpoint_write(stream, word->high);
        checkpoint_write(stream, word->length);
    }
}

PackedState checkpoint_read_state(std::istream& stream) {
    PackedState state;
    for (PackedWord* word : {&state.first, &state.second}) {
        word->low = checkpoint_read<std::uint64_t>(stream);
        word->high = checkpoint_read<std::uint64_t>(stream);
        word->length = checkpoint_read<std::uint16_t>(stream);
        if (word->length > 64) {
            throw std::runtime_error("Triple GS checkpoint contains an invalid word length.");
        }
    }
    return state;
}

void checkpoint_write_config(
    std::ostream& stream,
    const TripleCheckpointConfig& config
) {
    checkpoint_write_state(stream, config.start_state);
    checkpoint_write(stream, config.total_node_budget);
    checkpoint_write(stream, config.max_total_length);
    checkpoint_write(stream, config.auto_preference_threshold);
    checkpoint_write(stream, config.complete_preference_threshold);
    checkpoint_write(stream, config.min_standard_moves_between_auto);
    checkpoint_write(stream, config.max_live_frontier_states);
    checkpoint_write(stream, static_cast<std::uint8_t>(config.seed_initial_auto_neighbors));
    checkpoint_write(stream, static_cast<std::uint8_t>(config.seed_initial_complete_neighbors));
    checkpoint_write(stream, static_cast<std::uint8_t>(config.deepest));
}

TripleCheckpointConfig checkpoint_read_config(std::istream& stream) {
    TripleCheckpointConfig config;
    config.start_state = checkpoint_read_state(stream);
    config.total_node_budget = checkpoint_read<std::uint64_t>(stream);
    config.max_total_length = checkpoint_read<std::int32_t>(stream);
    config.auto_preference_threshold = checkpoint_read<std::int32_t>(stream);
    config.complete_preference_threshold = checkpoint_read<std::int32_t>(stream);
    config.min_standard_moves_between_auto = checkpoint_read<std::int32_t>(stream);
    config.max_live_frontier_states = checkpoint_read<std::int64_t>(stream);
    config.seed_initial_auto_neighbors = checkpoint_read<std::uint8_t>(stream) != 0;
    config.seed_initial_complete_neighbors = checkpoint_read<std::uint8_t>(stream) != 0;
    config.deepest = checkpoint_read<std::uint8_t>(stream) != 0;
    return config;
}

bool same_checkpoint_config(
    const TripleCheckpointConfig& left,
    const TripleCheckpointConfig& right
) {
    return left.start_state == right.start_state
        && left.total_node_budget == right.total_node_budget
        && left.max_total_length == right.max_total_length
        && left.auto_preference_threshold == right.auto_preference_threshold
        && left.complete_preference_threshold == right.complete_preference_threshold
        && left.min_standard_moves_between_auto == right.min_standard_moves_between_auto
        && left.max_live_frontier_states == right.max_live_frontier_states
        && left.seed_initial_auto_neighbors == right.seed_initial_auto_neighbors
        && left.seed_initial_complete_neighbors == right.seed_initial_complete_neighbors
        && left.deepest == right.deepest;
}

void checkpoint_write_counters(
    std::ostream& stream,
    const TripleSearchCounters& counters
) {
#define WRITE_TRIPLE_COUNTER(name) checkpoint_write(stream, counters.name)
    WRITE_TRIPLE_COUNTER(standard_generated_count);
    WRITE_TRIPLE_COUNTER(standard_enqueued_count);
    WRITE_TRIPLE_COUNTER(auto_generated_count);
    WRITE_TRIPLE_COUNTER(auto_valid_count);
    WRITE_TRIPLE_COUNTER(auto_enqueued_count);
    WRITE_TRIPLE_COUNTER(auto_generation_call_count);
    WRITE_TRIPLE_COUNTER(initial_auto_seed_enqueued_count);
    WRITE_TRIPLE_COUNTER(initial_complete_seed_enqueued_count);
    WRITE_TRIPLE_COUNTER(complete_generation_call_count);
    WRITE_TRIPLE_COUNTER(complete_neighbor_count);
    WRITE_TRIPLE_COUNTER(complete_seen_neighbor_count);
    WRITE_TRIPLE_COUNTER(complete_direct_enqueued_count);
    WRITE_TRIPLE_COUNTER(complete_prefilter_accepted_count);
    WRITE_TRIPLE_COUNTER(complete_prefilter_rejected_count);
    WRITE_TRIPLE_COUNTER(complete_ineligible_pop_count);
    WRITE_TRIPLE_COUNTER(auto_queued_count);
    WRITE_TRIPLE_COUNTER(auto_queue_eligible_count);
    WRITE_TRIPLE_COUNTER(auto_queue_distance_blocked_count);
    WRITE_TRIPLE_COUNTER(complete_queued_count);
    WRITE_TRIPLE_COUNTER(sub_pop_count);
    WRITE_TRIPLE_COUNTER(auto_pop_count);
    WRITE_TRIPLE_COUNTER(complete_pop_count);
    WRITE_TRIPLE_COUNTER(nodes_used);
    WRITE_TRIPLE_COUNTER(frontier_prune_count);
    WRITE_TRIPLE_COUNTER(frontier_states_pruned);
    WRITE_TRIPLE_COUNTER(frontier_peak);
    WRITE_TRIPLE_COUNTER(checkpoint_count);
    WRITE_TRIPLE_COUNTER(last_checkpoint_nodes);
#undef WRITE_TRIPLE_COUNTER
}

TripleSearchCounters checkpoint_read_counters(std::istream& stream) {
    TripleSearchCounters counters;
#define READ_TRIPLE_COUNTER(name) counters.name = checkpoint_read<std::uint64_t>(stream)
    READ_TRIPLE_COUNTER(standard_generated_count);
    READ_TRIPLE_COUNTER(standard_enqueued_count);
    READ_TRIPLE_COUNTER(auto_generated_count);
    READ_TRIPLE_COUNTER(auto_valid_count);
    READ_TRIPLE_COUNTER(auto_enqueued_count);
    READ_TRIPLE_COUNTER(auto_generation_call_count);
    READ_TRIPLE_COUNTER(initial_auto_seed_enqueued_count);
    READ_TRIPLE_COUNTER(initial_complete_seed_enqueued_count);
    READ_TRIPLE_COUNTER(complete_generation_call_count);
    READ_TRIPLE_COUNTER(complete_neighbor_count);
    READ_TRIPLE_COUNTER(complete_seen_neighbor_count);
    READ_TRIPLE_COUNTER(complete_direct_enqueued_count);
    READ_TRIPLE_COUNTER(complete_prefilter_accepted_count);
    READ_TRIPLE_COUNTER(complete_prefilter_rejected_count);
    READ_TRIPLE_COUNTER(complete_ineligible_pop_count);
    READ_TRIPLE_COUNTER(auto_queued_count);
    READ_TRIPLE_COUNTER(auto_queue_eligible_count);
    READ_TRIPLE_COUNTER(auto_queue_distance_blocked_count);
    READ_TRIPLE_COUNTER(complete_queued_count);
    READ_TRIPLE_COUNTER(sub_pop_count);
    READ_TRIPLE_COUNTER(auto_pop_count);
    READ_TRIPLE_COUNTER(complete_pop_count);
    READ_TRIPLE_COUNTER(nodes_used);
    READ_TRIPLE_COUNTER(frontier_prune_count);
    READ_TRIPLE_COUNTER(frontier_states_pruned);
    READ_TRIPLE_COUNTER(frontier_peak);
    READ_TRIPLE_COUNTER(checkpoint_count);
    READ_TRIPLE_COUNTER(last_checkpoint_nodes);
#undef READ_TRIPLE_COUNTER
    return counters;
}

void checkpoint_write_queue(std::ostream& stream, const MinQueue& queue) {
    const auto& entries = queue.entries();
    checkpoint_write(stream, static_cast<std::uint64_t>(entries.size()));
    for (const auto& entry : entries) {
        checkpoint_write(stream, entry.priority.score);
        checkpoint_write(stream, entry.priority.depth_priority);
        checkpoint_write(stream, entry.priority.insertion);
        checkpoint_write(stream, entry.state_id);
    }
}

std::vector<QueueEntry> checkpoint_read_queue(
    std::istream& stream,
    std::size_t record_count
) {
    const auto count = checkpoint_read<std::uint64_t>(stream);
    if (count > (1ULL << 33)) {
        throw std::runtime_error("Triple GS checkpoint contains an invalid queue size.");
    }
    std::vector<QueueEntry> entries;
    entries.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        QueueEntry entry;
        entry.priority.score = checkpoint_read<std::uint16_t>(stream);
        entry.priority.depth_priority = checkpoint_read<std::int64_t>(stream);
        entry.priority.insertion = checkpoint_read<std::uint64_t>(stream);
        entry.state_id = checkpoint_read<StateId>(stream);
        if (entry.state_id >= record_count) {
            throw std::runtime_error("Triple GS checkpoint queue references an invalid state.");
        }
        entries.push_back(entry);
    }
    return entries;
}

void checkpoint_write_header(
    std::ostream& stream,
    const TripleCheckpointConfig& config,
    const std::string& metadata_bytes
) {
    stream.write(TRIPLE_CHECKPOINT_MAGIC.data(), TRIPLE_CHECKPOINT_MAGIC.size());
    checkpoint_write(stream, TRIPLE_CHECKPOINT_VERSION);
    checkpoint_write_config(stream, config);
    checkpoint_write_bytes(stream, metadata_bytes);
}

std::pair<TripleCheckpointConfig, std::string> checkpoint_read_header(
    std::istream& stream
) {
    std::array<char, TRIPLE_CHECKPOINT_MAGIC.size()> magic{};
    stream.read(magic.data(), magic.size());
    if (!stream || magic != TRIPLE_CHECKPOINT_MAGIC) {
        throw std::runtime_error("Not a supported Triple GS active checkpoint.");
    }
    const auto version = checkpoint_read<std::uint32_t>(stream);
    if (version != TRIPLE_CHECKPOINT_VERSION) {
        throw std::runtime_error("Unsupported Triple GS active checkpoint version.");
    }
    auto config = checkpoint_read_config(stream);
    auto metadata = checkpoint_read_bytes(stream);
    return {config, metadata};
}

std::string pickle_checkpoint_object(const py::object& value) {
    const py::object pickle = py::module_::import("pickle");
    return pickle.attr("dumps")(value, py::int_(5)).cast<std::string>();
}

py::object unpickle_checkpoint_object(const std::string& bytes) {
    if (bytes.empty()) return py::none();
    const py::object pickle = py::module_::import("pickle");
    return pickle.attr("loads")(py::bytes(bytes));
}

std::uint64_t save_triple_active_checkpoint(
    const std::string& checkpoint_path,
    const TripleCheckpointConfig& config,
    const py::object& metadata_provider,
    const std::vector<StateRecord>& records,
    const std::unordered_map<StateId, py::object>& special_moves,
    const MinQueue& sub_queue,
    const MinQueue& auto_queue,
    const MinQueue& complete_queue,
    std::uint64_t insertion_counter,
    StateId best_id,
    const TripleSearchCounters& counters
) {
    if (checkpoint_path.empty()) return 0;
    const std::filesystem::path target(checkpoint_path);
    if (!target.parent_path().empty()) {
        std::filesystem::create_directories(target.parent_path());
    }
    const std::filesystem::path temporary = target.string() + ".tmp";
    py::object metadata = py::none();
    if (!metadata_provider.is_none()) {
        metadata = metadata_provider();
    }
    const std::string metadata_bytes = pickle_checkpoint_object(metadata);

    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error(
                "Unable to open Triple GS active checkpoint for writing: "
                + temporary.string()
            );
        }
        checkpoint_write_header(stream, config, metadata_bytes);
        checkpoint_write(stream, insertion_counter);
        checkpoint_write(stream, best_id);
        checkpoint_write_counters(stream, counters);

        checkpoint_write(stream, static_cast<std::uint64_t>(records.size()));
        for (const auto& record : records) {
            checkpoint_write_state(stream, record.state);
            checkpoint_write(stream, record.parent);
            checkpoint_write(stream, record.depth);
            checkpoint_write(stream, record.standard_gap);
            checkpoint_write(stream, static_cast<std::uint8_t>(record.move_kind));
            std::uint8_t flags = 0;
            flags |= static_cast<std::uint8_t>(record.alive) << 0;
            flags |= static_cast<std::uint8_t>(record.sub_expanded) << 1;
            flags |= static_cast<std::uint8_t>(record.cov_queued) << 2;
            flags |= static_cast<std::uint8_t>(record.cov_expanded) << 3;
            flags |= static_cast<std::uint8_t>(record.complete_cov_queued) << 4;
            flags |= static_cast<std::uint8_t>(record.complete_cov_expanded) << 5;
            checkpoint_write(stream, flags);
        }

        std::vector<StateId> special_ids;
        special_ids.reserve(special_moves.size());
        for (const auto& item : special_moves) special_ids.push_back(item.first);
        std::sort(special_ids.begin(), special_ids.end());
        checkpoint_write(stream, static_cast<std::uint64_t>(special_ids.size()));
        for (const StateId state_id : special_ids) {
            checkpoint_write(stream, state_id);
            checkpoint_write_bytes(
                stream,
                pickle_checkpoint_object(special_moves.at(state_id))
            );
        }

        checkpoint_write_queue(stream, sub_queue);
        checkpoint_write_queue(stream, auto_queue);
        checkpoint_write_queue(stream, complete_queue);
        stream.flush();
        if (!stream) {
            throw std::runtime_error("Failed to flush Triple GS active checkpoint.");
        }
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary, target, rename_error);
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(target, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary, target, rename_error);
        if (rename_error) {
            throw std::runtime_error(
                "Unable to publish Triple GS active checkpoint: "
                + rename_error.message()
            );
        }
    }
    return std::filesystem::file_size(target);
}

void load_triple_active_checkpoint(
    const std::string& checkpoint_path,
    const TripleCheckpointConfig& expected_config,
    std::vector<StateRecord>& records,
    std::unordered_map<PackedState, StateId, PackedStateHash>& seen,
    std::unordered_map<StateId, py::object>& special_moves,
    std::unordered_set<PackedState, PackedStateHash>& minimum_states,
    MinQueue& sub_queue,
    MinQueue& auto_queue,
    MinQueue& complete_queue,
    std::uint64_t& insertion_counter,
    StateId& best_id,
    std::uint16_t& minimum_length,
    TripleSearchCounters& counters
) {
    std::ifstream stream(checkpoint_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "Unable to open Triple GS active checkpoint: " + checkpoint_path
        );
    }
    const auto header = checkpoint_read_header(stream);
    if (!same_checkpoint_config(header.first, expected_config)) {
        throw std::runtime_error(
            "Triple GS active checkpoint does not match the requested presentation or settings."
        );
    }
    insertion_counter = checkpoint_read<std::uint64_t>(stream);
    best_id = checkpoint_read<StateId>(stream);
    counters = checkpoint_read_counters(stream);

    const auto record_count = checkpoint_read<std::uint64_t>(stream);
    if (record_count == 0 || record_count >= static_cast<std::uint64_t>(NO_STATE)) {
        throw std::runtime_error("Triple GS checkpoint contains an invalid record count.");
    }
    records.clear();
    records.reserve(std::max<std::size_t>(1'000'000, static_cast<std::size_t>(record_count)));
    seen.clear();
    seen.reserve(std::max<std::size_t>(1'000'000, static_cast<std::size_t>(record_count)));
    minimum_states.clear();
    minimum_length = std::numeric_limits<std::uint16_t>::max();
    for (std::uint64_t index = 0; index < record_count; ++index) {
        StateRecord record;
        record.state = checkpoint_read_state(stream);
        record.parent = checkpoint_read<StateId>(stream);
        record.depth = checkpoint_read<std::uint32_t>(stream);
        record.standard_gap = checkpoint_read<std::uint32_t>(stream);
        const auto move_kind = checkpoint_read<std::uint8_t>(stream);
        if (move_kind > static_cast<std::uint8_t>(MoveKind::CompleteCov)) {
            throw std::runtime_error("Triple GS checkpoint contains an invalid move kind.");
        }
        record.move_kind = static_cast<MoveKind>(move_kind);
        const auto flags = checkpoint_read<std::uint8_t>(stream);
        record.alive = (flags & (1U << 0)) != 0;
        record.sub_expanded = (flags & (1U << 1)) != 0;
        record.cov_queued = (flags & (1U << 2)) != 0;
        record.cov_expanded = (flags & (1U << 3)) != 0;
        record.complete_cov_queued = (flags & (1U << 4)) != 0;
        record.complete_cov_expanded = (flags & (1U << 5)) != 0;
        if (record.parent != NO_STATE && record.parent >= index) {
            throw std::runtime_error("Triple GS checkpoint contains an invalid parent link.");
        }
        const StateId state_id = static_cast<StateId>(records.size());
        records.push_back(std::move(record));
        if (!seen.emplace(records.back().state, state_id).second) {
            throw std::runtime_error("Triple GS checkpoint contains duplicate state records.");
        }
        const auto length = packed_length(records.back().state);
        if (length < minimum_length) {
            minimum_length = length;
            minimum_states.clear();
            minimum_states.insert(records.back().state);
        } else if (length == minimum_length) {
            minimum_states.insert(records.back().state);
        }
    }
    if (
        best_id >= records.size()
        || !(records.front().state == expected_config.start_state)
    ) {
        throw std::runtime_error("Triple GS checkpoint contains an invalid best or start state.");
    }

    special_moves.clear();
    const auto special_count = checkpoint_read<std::uint64_t>(stream);
    if (special_count > record_count) {
        throw std::runtime_error("Triple GS checkpoint contains too many special moves.");
    }
    for (std::uint64_t index = 0; index < special_count; ++index) {
        const StateId state_id = checkpoint_read<StateId>(stream);
        if (state_id >= records.size()) {
            throw std::runtime_error("Triple GS checkpoint special move has an invalid state.");
        }
        special_moves.emplace(
            state_id,
            unpickle_checkpoint_object(checkpoint_read_bytes(stream))
        );
    }

    sub_queue.restore_entries(checkpoint_read_queue(stream, records.size()));
    auto_queue.restore_entries(checkpoint_read_queue(stream, records.size()));
    complete_queue.restore_entries(checkpoint_read_queue(stream, records.size()));
}

py::object read_triple_active_checkpoint_metadata(const std::string& checkpoint_path) {
    std::ifstream stream(checkpoint_path, std::ios::binary);
    if (!stream) return py::none();
    const auto header = checkpoint_read_header(stream);
    return unpickle_checkpoint_object(header.second);
}

py::dict run_compact_dual_gs(
    const std::string& start_first,
    const std::string& start_second,
    py::function cov_expander,
    std::uint64_t total_node_budget,
    int max_total_length,
    int cov_preference_threshold,
    bool seed_initial_cov_neighbors,
    int min_standard_moves_between_cov,
    const std::string& depth_tie_strategy,
    std::int64_t max_live_frontier_states,
    std::uint64_t progress_interval,
    bool expand_cov_on_every_sub_pop,
    const std::string& sub_queue_score_mode_name
) {
    if (depth_tie_strategy != "deepest" && depth_tie_strategy != "shallowest") {
        throw std::invalid_argument(
            "Compact Dual GS supports deterministic deepest or shallowest tie-breaking."
        );
    }
    if (max_total_length <= 0 || max_total_length > 65) {
        throw std::invalid_argument("max_total_length must be between 1 and 65.");
    }
    if (expand_cov_on_every_sub_pop && seed_initial_cov_neighbors) {
        throw std::invalid_argument(
            "Every-pop callback expansion already expands the initial state; "
            "seed_initial_cov_neighbors must be false."
        );
    }
    const auto sub_queue_score_mode = parse_sub_queue_score_mode(
        sub_queue_score_mode_name
    );
    const bool separate_sub_activation_index =
        sub_queue_score_mode != SubQueueScoreMode::Length;

    std::vector<StateRecord> records;
    records.reserve(1'000'000);
    std::unordered_map<PackedState, StateId, PackedStateHash> seen;
    seen.reserve(1'000'000);
    std::unordered_map<StateId, py::object> cov_moves;
    std::unordered_set<PackedState, PackedStateHash> minimum_states;
    MinQueue sub_queue;
    MinQueue sub_activation_queue;
    MinQueue cov_queue;
    std::uint64_t insertion_counter = 0;

    const bool deepest = depth_tie_strategy == "deepest";
    auto priority_for_score = [&](std::uint16_t score, std::uint32_t depth) {
        return Priority{
            score,
            deepest ? -static_cast<std::int64_t>(depth) : static_cast<std::int64_t>(depth),
            insertion_counter++,
        };
    };
    auto sub_priority_for = [&](const PackedState& state, std::uint32_t depth) {
        return priority_for_score(
            packed_sub_queue_score(state, sub_queue_score_mode), depth
        );
    };
    auto activation_priority_for = [&](const PackedState& state, std::uint32_t depth) {
        return priority_for_score(packed_length(state), depth);
    };

    std::uint16_t minimum_length = static_cast<std::uint16_t>(
        start_first.size() + start_second.size()
    );
    auto track_minimum = [&](const PackedState& state) {
        const auto length = packed_length(state);
        if (length < minimum_length) {
            minimum_length = length;
            minimum_states.clear();
            minimum_states.insert(state);
        } else if (length == minimum_length) {
            minimum_states.insert(state);
        }
    };

    auto allocate_state = [&](StateRecord record) {
        if (records.size() >= static_cast<std::size_t>(NO_STATE)) {
            throw std::overflow_error("Compact Dual GS exhausted 32-bit state IDs.");
        }
        const StateId state_id = static_cast<StateId>(records.size());
        records.push_back(std::move(record));
        return state_id;
    };

    std::uint64_t standard_generated_count = 0;
    std::uint64_t standard_enqueued_count = 0;
    std::uint64_t cov_candidate_count = 0;
    std::uint64_t cov_valid_candidate_count = 0;
    std::uint64_t cov_enqueued_count = 0;
    std::uint64_t cov_generation_call_count = 0;
    std::uint64_t cov_queued_count = 0;
    std::uint64_t initial_cov_seed_enqueued_count = 0;
    std::uint64_t cov_queue_eligible_count = 0;
    std::uint64_t cov_queue_distance_blocked_count = 0;
    std::uint64_t sub_pop_count = 0;
    std::uint64_t cov_pop_count = 0;
    std::uint64_t nodes_used = 0;
    std::uint64_t frontier_prune_count = 0;
    std::uint64_t frontier_states_pruned = 0;
    std::uint64_t frontier_peak_live_state_count = 0;

    auto enqueue_new_state = [&]
    (
        const PackedState& state,
        StateId parent,
        std::uint32_t depth,
        std::uint32_t gap,
        MoveKind move_kind,
        py::object cov_move
    ) -> std::pair<bool, StateId> {
        const auto found = seen.find(state);
        if (found != seen.end()) {
            return {false, found->second};
        }
        StateRecord record;
        record.state = state;
        record.parent = parent;
        record.depth = depth;
        record.standard_gap = gap;
        record.move_kind = move_kind;
        const StateId state_id = allocate_state(std::move(record));
        seen.emplace(state, state_id);
        if (move_kind == MoveKind::Cov && !cov_move.is_none()) {
            cov_moves.emplace(state_id, std::move(cov_move));
        }
        sub_queue.push({sub_priority_for(state, depth), state_id});
        if (separate_sub_activation_index) {
            sub_activation_queue.push({activation_priority_for(state, depth), state_id});
        }
        track_minimum(state);
        return {true, state_id};
    };

    const PackedState start_state = pack_state(start_first, start_second);
    StateRecord start_record;
    start_record.state = start_state;
    const StateId start_id = allocate_state(start_record);
    seen.emplace(start_state, start_id);
    minimum_states.insert(start_state);
    sub_queue.push({sub_priority_for(start_state, 0), start_id});
    if (separate_sub_activation_index) {
        sub_activation_queue.push({activation_priority_for(start_state, 0), start_id});
    }
    frontier_peak_live_state_count = 1;

    StateId best_id = start_id;
    auto candidate_is_better = [&](StateId candidate_id, StateId current_id) {
        const auto& candidate = records[candidate_id];
        const auto& current = records[current_id];
        const auto candidate_length = packed_length(candidate.state);
        const auto current_length = packed_length(current.state);
        if (candidate_length != current_length) return candidate_length < current_length;
        if (candidate.depth != current.depth) return candidate.depth < current.depth;
        return packed_solver_less(candidate.state, current.state);
    };
    auto update_best = [&](StateId state_id) {
        if (candidate_is_better(state_id, best_id)) best_id = state_id;
    };

    auto discard_stale = [&]() {
        while (!sub_queue.empty()) {
            const auto id = sub_queue.top().state_id;
            if (records[id].alive && !records[id].sub_expanded) break;
            sub_queue.pop();
        }
        while (!sub_activation_queue.empty()) {
            const auto id = sub_activation_queue.top().state_id;
            if (records[id].alive && !records[id].sub_expanded) break;
            sub_activation_queue.pop();
        }
        while (!cov_queue.empty()) {
            const auto id = cov_queue.top().state_id;
            if (records[id].alive && !records[id].cov_expanded) break;
            cov_queue.pop();
        }
    };

    auto expand_cov = [&](StateId source_id, const char* source_queue) {
        ++cov_generation_call_count;
        const auto source_words = unpack_state(records[source_id].state);
        const py::tuple callback_result = cov_expander(
            py::make_tuple(source_words.first, source_words.second),
            py::str(source_queue)
        ).cast<py::tuple>();
        const py::list neighbors = callback_result[0].cast<py::list>();
        cov_candidate_count += callback_result[1].cast<std::uint64_t>();
        cov_valid_candidate_count += callback_result[2].cast<std::uint64_t>();
        std::uint64_t enqueued = 0;
        for (const py::handle item_handle : neighbors) {
            const py::tuple item = py::reinterpret_borrow<py::tuple>(item_handle);
            const py::tuple next_state = item[0].cast<py::tuple>();
            const auto next_first = next_state[0].cast<std::string>();
            const auto next_second = next_state[1].cast<std::string>();
            const PackedState packed = pack_state(next_first, next_second);
            const auto inserted = enqueue_new_state(
                packed,
                source_id,
                records[source_id].depth + 1,
                0,
                MoveKind::Cov,
                py::reinterpret_borrow<py::object>(item[1])
            );
            if (inserted.first) {
                ++enqueued;
                update_best(inserted.second);
            }
        }
        cov_enqueued_count += enqueued;
        return enqueued;
    };

    if (seed_initial_cov_neighbors) {
        records[start_id].cov_expanded = true;
        records[start_id].cov_queued = true;
        const auto enqueued = expand_cov(start_id, "initial_cov_seed");
        initial_cov_seed_enqueued_count += enqueued;
    }

    auto prune_frontier = [&]() {
        const auto live_count = sub_queue.size() + cov_queue.size();
        if (
            max_live_frontier_states <= 0
            || live_count <= static_cast<std::size_t>(max_live_frontier_states)
        ) {
            return;
        }
        struct TaggedEntry {
            QueueEntry entry;
            Priority pruning_priority;
            bool is_cov = false;
        };
        std::vector<TaggedEntry> entries;
        entries.reserve(live_count);
        while (!sub_queue.empty()) {
            const auto entry = sub_queue.top();
            entries.push_back(
                {
                    entry,
                    Priority{
                        packed_length(records[entry.state_id].state),
                        entry.priority.depth_priority,
                        entry.priority.insertion,
                    },
                    false,
                }
            );
            sub_queue.pop();
        }
        while (!cov_queue.empty()) {
            const auto entry = cov_queue.top();
            entries.push_back({entry, entry.priority, true});
            cov_queue.pop();
        }
        std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
            if (priority_less(left.pruning_priority, right.pruning_priority)) return true;
            if (priority_less(right.pruning_priority, left.pruning_priority)) return false;
            return static_cast<int>(left.is_cov) < static_cast<int>(right.is_cov);
        });
        const auto retain_count = std::max<std::size_t>(
            1,
            static_cast<std::size_t>(max_live_frontier_states) / 2
        );
        for (std::size_t index = 0; index < entries.size(); ++index) {
            const auto& tagged = entries[index];
            if (index < retain_count) {
                if (tagged.is_cov) cov_queue.push(tagged.entry);
                else sub_queue.push(tagged.entry);
                continue;
            }
            if (!tagged.is_cov) {
                const StateId id = tagged.entry.state_id;
                auto& record = records[id];
                if (record.alive && !record.sub_expanded && !record.cov_expanded) {
                    const auto seen_entry = seen.find(record.state);
                    if (seen_entry != seen.end() && seen_entry->second == id) {
                        seen.erase(seen_entry);
                    }
                    cov_moves.erase(id);
                    record.alive = false;
                }
            }
        }
        if (separate_sub_activation_index) {
            sub_activation_queue = MinQueue{};
            for (const auto& entry : sub_queue.entries()) {
                const auto id = entry.state_id;
                sub_activation_queue.push(
                    {activation_priority_for(records[id].state, records[id].depth), id}
                );
            }
        }
        ++frontier_prune_count;
        frontier_states_pruned += entries.size() - retain_count;
    };

    bool solved = false;
    StateId final_id = best_id;
    while (nodes_used < total_node_budget) {
        discard_stale();
        if (sub_queue.empty() && cov_queue.empty()) break;
        bool skip_post_iteration = false;

        bool use_cov = false;
        if (sub_queue.empty()) {
            use_cov = true;
        } else if (!cov_queue.empty()) {
            const auto& sub_activation_index = separate_sub_activation_index
                ? sub_activation_queue
                : sub_queue;
            const auto sub_score = sub_activation_index.top().priority.score;
            const auto cov_score = cov_queue.top().priority.score;
            use_cov = static_cast<int>(sub_score) - static_cast<int>(cov_score)
                >= cov_preference_threshold + 1;
        }

        if (!use_cov) {
            const QueueEntry item = sub_queue.top();
            sub_queue.pop();
            StateRecord& current = records[item.state_id];
            if (!current.alive || current.sub_expanded) continue;
            current.sub_expanded = true;
            ++nodes_used;
            ++sub_pop_count;
            update_best(item.state_id);

            const PackedState current_state = current.state;
            const auto current_depth = current.depth;
            const auto current_gap = current.standard_gap;
            if (current_state.first.length == 1 && current_state.second.length == 1) {
                solved = true;
                final_id = item.state_id;
                break;
            }

            const auto words = unpack_state(current_state);
            const auto successors = standard_successors(
                words.first,
                words.second,
                max_total_length
            );
            const auto next_depth = current_depth + 1;
            const auto next_gap = current_gap + 1;
            for (const auto& successor : successors) {
                ++standard_generated_count;
                const auto inserted = enqueue_new_state(
                    pack_state(successor.first, successor.second),
                    item.state_id,
                    next_depth,
                    next_gap,
                    MoveKind::Standard,
                    py::none()
                );
                if (inserted.first) {
                    ++standard_enqueued_count;
                    update_best(inserted.second);
                }
            }

            if (expand_cov_on_every_sub_pop) {
                if (
                    current_gap
                    >= static_cast<std::uint32_t>(min_standard_moves_between_cov)
                ) {
                    ++cov_queue_eligible_count;
                    expand_cov(item.state_id, "standard_pop");
                } else {
                    ++cov_queue_distance_blocked_count;
                }
            } else {
                StateRecord& refreshed_current = records[item.state_id];
                if (!refreshed_current.cov_queued) {
                    if (current_gap < static_cast<std::uint32_t>(min_standard_moves_between_cov)) {
                        ++cov_queue_distance_blocked_count;
                        skip_post_iteration = true;
                    } else {
                        ++cov_queue_eligible_count;
                        refreshed_current.cov_queued = true;
                        cov_queue.push(
                            {activation_priority_for(current_state, current_depth), item.state_id}
                        );
                        ++cov_queued_count;
                    }
                }
            }
        } else {
            const QueueEntry item = cov_queue.top();
            cov_queue.pop();
            StateRecord& current = records[item.state_id];
            if (!current.alive || current.cov_expanded) continue;
            current.cov_expanded = true;
            ++nodes_used;
            ++cov_pop_count;
            update_best(item.state_id);
            expand_cov(item.state_id, "cov");
        }

        if (skip_post_iteration) continue;

        const auto live_frontier = sub_queue.size() + cov_queue.size();
        frontier_peak_live_state_count = std::max<std::uint64_t>(
            frontier_peak_live_state_count,
            live_frontier
        );
        prune_frontier();
        if (progress_interval > 0 && nodes_used % progress_interval == 0) {
            std::cout
                << "nodes=" << nodes_used
                << " sub_q=" << sub_queue.size()
                << " cov_q=" << cov_queue.size()
                << " seen=" << seen.size()
                << " best_len=" << packed_length(records[best_id].state)
                << std::endl;
        }
    }

    discard_stale();
    if (!solved) final_id = best_id;

    std::vector<StateId> path_ids;
    for (StateId id = final_id; id != NO_STATE; id = records[id].parent) {
        path_ids.push_back(id);
    }
    std::reverse(path_ids.begin(), path_ids.end());

    py::list path;
    py::list moves;
    for (const StateId id : path_ids) path.append(state_tuple(records[id].state));
    for (std::size_t index = 1; index < path_ids.size(); ++index) {
        const StateId id = path_ids[index];
        const StateId parent = path_ids[index - 1];
        if (records[id].move_kind == MoveKind::Cov) {
            moves.append(cov_moves.at(id));
        } else {
            py::dict move;
            move["move_type"] = "standard_substitution";
            move["from_state"] = state_tuple(records[parent].state);
            move["to_state"] = state_tuple(records[id].state);
            move["source_queue"] = "sub";
            moves.append(std::move(move));
        }
    }

    std::vector<PackedState> ordered_minimum_states(
        minimum_states.begin(), minimum_states.end()
    );
    std::sort(
        ordered_minimum_states.begin(),
        ordered_minimum_states.end(),
        packed_solver_less
    );
    py::list minimum_state_list;
    for (const auto& state : ordered_minimum_states) {
        minimum_state_list.append(state_tuple(state));
    }

    std::uint64_t sub_expanded_count = 0;
    std::uint64_t cov_expanded_count = 0;
    for (const auto& record : records) {
        if (!record.alive) continue;
        sub_expanded_count += record.sub_expanded;
        cov_expanded_count += record.cov_expanded;
    }

    py::dict result;
    result["canonical_start"] = state_tuple(start_state);
    result["solved"] = solved;
    result["solved_trivial"] = solved;
    result["termination_reason"] = solved
        ? "solved"
        : (nodes_used >= total_node_budget ? "node_budget_exhausted" : "queues_exhausted");
    result["total_nodes_used"] = nodes_used;
    result["path_length"] = path_ids.size() - 1;
    result["path"] = path;
    result["moves"] = moves;
    result["final_state"] = state_tuple(records[final_id].state);
    result["final_total_length"] = packed_length(records[final_id].state);
    result["minimum_total_length"] = packed_length(records[final_id].state);
    result["minimum_state_count"] = ordered_minimum_states.size();
    result["minimum_length_states"] = minimum_state_list;
    result["outer_iterations"] = nodes_used;
    result["queue_size"] = sub_queue.size() + cov_queue.size();
    result["sub_queue_size"] = sub_queue.size();
    result["sub_activation_queue_size"] = separate_sub_activation_index
        ? sub_activation_queue.size()
        : sub_queue.size();
    result["cov_queue_size"] = cov_queue.size();
    result["enqueued_state_count"] = seen.size();
    result["expanded_state_count"] = sub_expanded_count + cov_expanded_count;
    result["sub_expanded_count"] = sub_expanded_count;
    result["cov_expansion_count"] = cov_expanded_count;
    result["sub_pop_count"] = sub_pop_count;
    result["cov_pop_count"] = cov_pop_count;
    result["cov_queued_count"] = cov_queued_count;
    result["initial_cov_seed_enqueued_count"] = initial_cov_seed_enqueued_count;
    result["cov_queue_eligible_count"] = cov_queue_eligible_count;
    result["cov_queue_distance_blocked_count"] = cov_queue_distance_blocked_count;
    result["min_standard_moves_between_cov"] = min_standard_moves_between_cov;
    result["outer_tie_strategy"] = depth_tie_strategy;
    result["cov_preference_threshold"] = cov_preference_threshold;
    result["cov_required_length_advantage"] = cov_preference_threshold + 1;
    result["seed_initial_cov_neighbors"] = seed_initial_cov_neighbors;
    result["seed_initial_additional_cov_moves"] = false;
    result["standard_generated_count"] = standard_generated_count;
    result["standard_enqueued_count"] = standard_enqueued_count;
    result["cov_candidate_count"] = cov_candidate_count;
    result["cov_valid_candidate_count"] = cov_valid_candidate_count;
    result["cov_enqueued_count"] = cov_enqueued_count;
    result["cov_generation_call_count"] = cov_generation_call_count;
    result["additional_cov_integration_mode"] = "full";
    result["additional_cov_generated_count"] = 0;
    result["additional_cov_valid_novel_count"] = 0;
    result["additional_cov_duplicate_or_seen_count"] = 0;
    result["additional_cov_filtered_length_count"] = 0;
    result["additional_cov_seen_reserved_count"] = 0;
    result["additional_cov_enqueued_count"] = 0;
    result["additional_cov_activation_index_enqueued_count"] = 0;
    result["additional_cov_sub_pop_count"] = 0;
    result["standard_blocked_by_additional_cov_seen_count"] = 0;
    result["cov_blocked_by_additional_cov_seen_count"] = 0;
    result["max_live_frontier_states"] = max_live_frontier_states;
    result["frontier_prune_count"] = frontier_prune_count;
    result["frontier_states_pruned"] = frontier_states_pruned;
    result["frontier_peak_live_state_count"] = frontier_peak_live_state_count;
    result["cov_variant"] = "single_cov";
    result["queue_policy"] = "dual_sub_then_cov_strict_length_cov";
    result["final_depth"] = records[final_id].depth;
    result["storage_backend"] = "compact_cpp";
    result["packed_state_bytes"] = sizeof(PackedState);
    result["allocated_state_slots"] = records.size();
    result["expand_cov_on_every_sub_pop"] = expand_cov_on_every_sub_pop;
    result["sub_queue_score_mode"] = sub_queue_score_mode_name;
    result["cov_activation_score_name"] = "total_length";
    result["matrix_score_bound"] = MASS_SCORE_BOUND;
    result["matrix_score_fallback"] = MASS_SCORE_FALLBACK;
    return result;
}

py::dict run_compact_triple_gs(
    const std::string& start_first,
    const std::string& start_second,
    py::function auto_expander,
    py::function complete_expander,
    std::uint64_t total_node_budget,
    int max_total_length,
    int auto_preference_threshold,
    int complete_preference_threshold,
    bool seed_initial_auto_neighbors,
    bool seed_initial_complete_neighbors,
    int min_standard_moves_between_auto,
    const std::string& depth_tie_strategy,
    std::int64_t max_live_frontier_states,
    std::uint64_t progress_interval,
    const std::string& active_checkpoint_path,
    double checkpoint_interval_seconds,
    bool resume_active_checkpoint,
    std::uint64_t checkpoint_interval_nodes,
    bool stop_after_checkpoint,
    py::object checkpoint_metadata_provider,
    py::object checkpoint_callback,
    double progress_heartbeat_interval_seconds,
    std::uint64_t progress_heartbeat_interval_nodes,
    py::object progress_heartbeat_callback,
    std::uint64_t max_resident_memory_bytes,
    std::uint64_t resource_check_interval_nodes,
    const std::string& sub_queue_score_mode_name,
    const std::string& cov_activation_score_mode_name
) {
    if (depth_tie_strategy != "deepest" && depth_tie_strategy != "shallowest") {
        throw std::invalid_argument(
            "Compact Triple GS supports deterministic deepest or shallowest tie-breaking."
        );
    }
    if (max_total_length <= 0 || max_total_length > 65) {
        throw std::invalid_argument("max_total_length must be between 1 and 65.");
    }
    if (auto_preference_threshold < 0 || complete_preference_threshold < 0) {
        throw std::invalid_argument("Triple GS queue thresholds must be nonnegative.");
    }
    if (min_standard_moves_between_auto < 0) {
        throw std::invalid_argument("Triple GS Auto COV gap must be nonnegative.");
    }
    if (checkpoint_interval_seconds < 0.0) {
        throw std::invalid_argument("Triple GS checkpoint interval must be nonnegative.");
    }
    if (progress_heartbeat_interval_seconds < 0.0) {
        throw std::invalid_argument(
            "Triple GS progress heartbeat interval must be nonnegative."
        );
    }
    if (max_resident_memory_bytes > 0 && resource_check_interval_nodes == 0) {
        throw std::invalid_argument(
            "Triple GS resource check interval must be positive when a memory limit is set."
        );
    }
    const auto sub_queue_score_mode = parse_sub_queue_score_mode(
        sub_queue_score_mode_name
    );
    if (
        cov_activation_score_mode_name != "total_length"
        && cov_activation_score_mode_name != "queue_score"
    ) {
        throw std::invalid_argument(
            "cov_activation_score_mode must be total_length or queue_score."
        );
    }
    const bool activation_uses_queue_score =
        cov_activation_score_mode_name == "queue_score";
    const bool separate_sub_activation_index =
        !activation_uses_queue_score
        && sub_queue_score_mode != SubQueueScoreMode::Length;

    std::vector<StateRecord> records;
    records.reserve(1'000'000);
    std::unordered_map<PackedState, StateId, PackedStateHash> seen;
    seen.reserve(1'000'000);
    std::unordered_map<StateId, py::object> special_moves;
    std::unordered_set<PackedState, PackedStateHash> minimum_states;
    MinQueue sub_queue;
    MinQueue sub_activation_queue;
    MinQueue auto_queue;
    MinQueue complete_queue;
    std::uint64_t insertion_counter = 0;

    const bool deepest = depth_tie_strategy == "deepest";
    auto priority_for_score = [&](std::uint16_t score, std::uint32_t depth) {
        return Priority{
            score,
            deepest ? -static_cast<std::int64_t>(depth) : static_cast<std::int64_t>(depth),
            insertion_counter++,
        };
    };
    auto sub_priority_for = [&](const PackedState& state, std::uint32_t depth) {
        return priority_for_score(
            packed_sub_queue_score(state, sub_queue_score_mode), depth
        );
    };
    auto activation_score_for = [&](const PackedState& state) {
        return activation_uses_queue_score
            ? packed_sub_queue_score(state, sub_queue_score_mode)
            : packed_length(state);
    };
    auto activation_priority_for = [&](const PackedState& state, std::uint32_t depth) {
        return priority_for_score(activation_score_for(state), depth);
    };
    auto activation_index_priority_for = [&](const QueueEntry& entry) {
        return Priority{
            activation_score_for(records[entry.state_id].state),
            entry.priority.depth_priority,
            entry.priority.insertion,
        };
    };

    std::uint16_t minimum_length = static_cast<std::uint16_t>(
        start_first.size() + start_second.size()
    );
    auto track_minimum = [&](const PackedState& state) {
        const auto length = packed_length(state);
        if (length < minimum_length) {
            minimum_length = length;
            minimum_states.clear();
            minimum_states.insert(state);
        } else if (length == minimum_length) {
            minimum_states.insert(state);
        }
    };

    auto allocate_state = [&](StateRecord record) {
        if (records.size() >= static_cast<std::size_t>(NO_STATE)) {
            throw std::overflow_error("Compact Triple GS exhausted 32-bit state IDs.");
        }
        const StateId state_id = static_cast<StateId>(records.size());
        records.push_back(std::move(record));
        return state_id;
    };

    TripleSearchCounters counters;
    auto& standard_generated_count = counters.standard_generated_count;
    auto& standard_enqueued_count = counters.standard_enqueued_count;
    auto& auto_generated_count = counters.auto_generated_count;
    auto& auto_valid_count = counters.auto_valid_count;
    auto& auto_enqueued_count = counters.auto_enqueued_count;
    auto& auto_generation_call_count = counters.auto_generation_call_count;
    auto& initial_auto_seed_enqueued_count = counters.initial_auto_seed_enqueued_count;
    auto& initial_complete_seed_enqueued_count = counters.initial_complete_seed_enqueued_count;
    auto& complete_generation_call_count = counters.complete_generation_call_count;
    auto& complete_neighbor_count = counters.complete_neighbor_count;
    auto& complete_seen_neighbor_count = counters.complete_seen_neighbor_count;
    auto& complete_direct_enqueued_count = counters.complete_direct_enqueued_count;
    auto& complete_prefilter_accepted_count = counters.complete_prefilter_accepted_count;
    auto& complete_prefilter_rejected_count = counters.complete_prefilter_rejected_count;
    auto& complete_ineligible_pop_count = counters.complete_ineligible_pop_count;
    auto& auto_queued_count = counters.auto_queued_count;
    auto& auto_queue_eligible_count = counters.auto_queue_eligible_count;
    auto& auto_queue_distance_blocked_count = counters.auto_queue_distance_blocked_count;
    auto& complete_queued_count = counters.complete_queued_count;
    auto& sub_pop_count = counters.sub_pop_count;
    auto& auto_pop_count = counters.auto_pop_count;
    auto& complete_pop_count = counters.complete_pop_count;
    auto& nodes_used = counters.nodes_used;
    auto& frontier_prune_count = counters.frontier_prune_count;
    auto& frontier_states_pruned = counters.frontier_states_pruned;
    auto& frontier_peak = counters.frontier_peak;

    auto enqueue_new_state = [&]
    (
        const PackedState& state,
        StateId parent,
        std::uint32_t depth,
        std::uint32_t standard_gap,
        MoveKind move_kind,
        py::object special_move
    ) -> std::pair<bool, StateId> {
        const auto found = seen.find(state);
        if (found != seen.end()) return {false, found->second};
        StateRecord record;
        record.state = state;
        record.parent = parent;
        record.depth = depth;
        record.standard_gap = standard_gap;
        record.move_kind = move_kind;
        const StateId state_id = allocate_state(std::move(record));
        seen.emplace(state, state_id);
        if (
            (move_kind == MoveKind::Cov || move_kind == MoveKind::CompleteCov)
            && !special_move.is_none()
        ) {
            special_moves.emplace(state_id, std::move(special_move));
        }
        const QueueEntry sub_entry{sub_priority_for(state, depth), state_id};
        sub_queue.push(sub_entry);
        if (separate_sub_activation_index) {
            sub_activation_queue.push(
                {activation_index_priority_for(sub_entry), state_id}
            );
        }
        track_minimum(state);
        return {true, state_id};
    };

    const PackedState start_state = pack_state(start_first, start_second);
    StateRecord start_record;
    start_record.state = start_state;
    const StateId start_id = allocate_state(start_record);
    seen.emplace(start_state, start_id);
    minimum_states.insert(start_state);
    const QueueEntry start_entry{sub_priority_for(start_state, 0), start_id};
    sub_queue.push(start_entry);
    if (separate_sub_activation_index) {
        sub_activation_queue.push(
            {activation_index_priority_for(start_entry), start_id}
        );
    }
    frontier_peak = 1;

    StateId best_id = start_id;
    const TripleCheckpointConfig checkpoint_config{
        start_state,
        total_node_budget,
        static_cast<std::int32_t>(max_total_length),
        static_cast<std::int32_t>(auto_preference_threshold),
        static_cast<std::int32_t>(complete_preference_threshold),
        static_cast<std::int32_t>(min_standard_moves_between_auto),
        max_live_frontier_states,
        seed_initial_auto_neighbors,
        seed_initial_complete_neighbors,
        deepest,
    };
    bool resumed_from_active_checkpoint = false;
    std::uint64_t last_checkpoint_bytes = 0;
    auto candidate_is_better = [&](StateId candidate_id, StateId current_id) {
        const auto& candidate = records[candidate_id];
        const auto& current = records[current_id];
        const auto candidate_length = packed_length(candidate.state);
        const auto current_length = packed_length(current.state);
        if (candidate_length != current_length) return candidate_length < current_length;
        if (candidate.depth != current.depth) return candidate.depth < current.depth;
        return packed_solver_less(candidate.state, current.state);
    };
    auto update_best = [&](StateId state_id) {
        if (candidate_is_better(state_id, best_id)) best_id = state_id;
    };

    auto discard_stale = [&]() {
        while (!sub_queue.empty()) {
            const auto id = sub_queue.top().state_id;
            if (records[id].alive && !records[id].sub_expanded) break;
            sub_queue.pop();
        }
        while (!sub_activation_queue.empty()) {
            const auto id = sub_activation_queue.top().state_id;
            if (records[id].alive && !records[id].sub_expanded) break;
            sub_activation_queue.pop();
        }
        while (!auto_queue.empty()) {
            const auto id = auto_queue.top().state_id;
            if (records[id].alive && !records[id].cov_expanded) break;
            auto_queue.pop();
        }
        while (!complete_queue.empty()) {
            const auto id = complete_queue.top().state_id;
            if (records[id].alive && !records[id].complete_cov_expanded) break;
            complete_queue.pop();
        }
    };

    auto invoke_auto = [&](StateId source_id, const char* source_queue) {
        ++auto_generation_call_count;
        const auto words = unpack_state(records[source_id].state);
        const py::tuple callback_result = auto_expander(
            py::make_tuple(words.first, words.second),
            py::str(source_queue)
        ).cast<py::tuple>();
        const py::list neighbors = callback_result[0].cast<py::list>();
        auto_generated_count += callback_result[1].cast<std::uint64_t>();
        auto_valid_count += callback_result[2].cast<std::uint64_t>();
        std::uint64_t enqueued = 0;
        for (const py::handle item_handle : neighbors) {
            const py::tuple item = py::reinterpret_borrow<py::tuple>(item_handle);
            const py::tuple next_state = item[0].cast<py::tuple>();
            const PackedState packed = pack_state(
                next_state[0].cast<std::string>(),
                next_state[1].cast<std::string>()
            );
            const auto inserted = enqueue_new_state(
                packed,
                source_id,
                records[source_id].depth + 1,
                0,
                MoveKind::Cov,
                py::reinterpret_borrow<py::object>(item[1])
            );
            if (inserted.first) {
                ++enqueued;
                update_best(inserted.second);
            }
        }
        auto_enqueued_count += enqueued;
        return enqueued;
    };

    if (
        resume_active_checkpoint
        && !active_checkpoint_path.empty()
        && std::filesystem::exists(active_checkpoint_path)
    ) {
        load_triple_active_checkpoint(
            active_checkpoint_path,
            checkpoint_config,
            records,
            seen,
            special_moves,
            minimum_states,
            sub_queue,
            auto_queue,
            complete_queue,
            insertion_counter,
            best_id,
            minimum_length,
            counters
        );
        resumed_from_active_checkpoint = true;
        last_checkpoint_bytes = std::filesystem::file_size(active_checkpoint_path);
        if (separate_sub_activation_index) {
            for (const auto& entry : sub_queue.entries()) {
                sub_activation_queue.push(
                    {activation_index_priority_for(entry), entry.state_id}
                );
            }
        }
    }

    if (!resumed_from_active_checkpoint && seed_initial_auto_neighbors) {
        records[start_id].cov_queued = true;
        records[start_id].cov_expanded = true;
        ++auto_queued_count;
        initial_auto_seed_enqueued_count += invoke_auto(start_id, "initial_auto_cov_seed");
    }

    if (!resumed_from_active_checkpoint && seed_initial_complete_neighbors) {
        records[start_id].complete_cov_queued = true;
        records[start_id].complete_cov_expanded = true;
        ++complete_queued_count;
        ++complete_generation_call_count;
        const auto words = unpack_state(start_state);
        const py::tuple callback_result = complete_expander(
            py::make_tuple(words.first, words.second),
            py::str("initial_complete_cov_seed")
        ).cast<py::tuple>();
        const py::list neighbors = callback_result[0].cast<py::list>();
        complete_neighbor_count += neighbors.size();
        for (const py::handle item_handle : neighbors) {
            const py::tuple item = py::reinterpret_borrow<py::tuple>(item_handle);
            const py::tuple next_state = item[0].cast<py::tuple>();
            const PackedState packed = pack_state(
                next_state[0].cast<std::string>(),
                next_state[1].cast<std::string>()
            );
            const auto inserted = enqueue_new_state(
                packed,
                start_id,
                1,
                0,
                MoveKind::CompleteCov,
                py::reinterpret_borrow<py::object>(item[1])
            );
            if (inserted.first) {
                ++initial_complete_seed_enqueued_count;
                ++complete_direct_enqueued_count;
                update_best(inserted.second);
            } else {
                ++complete_seen_neighbor_count;
            }
        }
    }

    auto prune_frontier = [&]() {
        const auto live_count = sub_queue.size() + auto_queue.size() + complete_queue.size();
        if (
            max_live_frontier_states <= 0
            || live_count <= static_cast<std::size_t>(max_live_frontier_states)
        ) return;

        struct TaggedEntry {
            QueueEntry entry;
            Priority pruning_priority;
            std::uint8_t queue_rank = 0;
        };
        std::vector<TaggedEntry> entries;
        entries.reserve(live_count);
        while (!sub_queue.empty()) {
            const auto entry = sub_queue.top();
            entries.push_back(
                {entry, activation_index_priority_for(entry), 0}
            );
            sub_queue.pop();
        }
        while (!auto_queue.empty()) {
            const auto entry = auto_queue.top();
            entries.push_back({entry, entry.priority, 1});
            auto_queue.pop();
        }
        while (!complete_queue.empty()) {
            const auto entry = complete_queue.top();
            entries.push_back({entry, entry.priority, 2});
            complete_queue.pop();
        }
        std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
            if (priority_less(left.pruning_priority, right.pruning_priority)) return true;
            if (priority_less(right.pruning_priority, left.pruning_priority)) return false;
            return left.queue_rank < right.queue_rank;
        });
        const auto retain_count = std::max<std::size_t>(
            1,
            static_cast<std::size_t>(max_live_frontier_states) / 2
        );
        for (std::size_t index = 0; index < std::min(retain_count, entries.size()); ++index) {
            const auto& tagged = entries[index];
            if (tagged.queue_rank == 0) sub_queue.push(tagged.entry);
            else if (tagged.queue_rank == 1) auto_queue.push(tagged.entry);
            else complete_queue.push(tagged.entry);
        }
        if (separate_sub_activation_index) {
            sub_activation_queue = MinQueue{};
            for (const auto& entry : sub_queue.entries()) {
                sub_activation_queue.push(
                    {activation_index_priority_for(entry), entry.state_id}
                );
            }
        }
        ++frontier_prune_count;
        frontier_states_pruned += entries.size() - std::min(retain_count, entries.size());
    };

    bool solved = false;
    bool checkpoint_requested_stop = false;
    bool memory_limit_reached = false;
    StateId final_id = best_id;
    using CheckpointClock = std::chrono::steady_clock;
    const auto search_started_at = CheckpointClock::now();
    auto next_checkpoint_time = CheckpointClock::now()
        + std::chrono::duration_cast<CheckpointClock::duration>(
            std::chrono::duration<double>(checkpoint_interval_seconds)
        );
    auto maybe_checkpoint = [&]() {
        if (active_checkpoint_path.empty()) return false;
        const auto now = CheckpointClock::now();
        const bool time_due = checkpoint_interval_seconds > 0.0
            && now >= next_checkpoint_time;
        const bool nodes_due = checkpoint_interval_nodes > 0
            && nodes_used >= counters.last_checkpoint_nodes + checkpoint_interval_nodes;
        if (!time_due && !nodes_due) return false;

        discard_stale();
        ++counters.checkpoint_count;
        counters.last_checkpoint_nodes = nodes_used;
        last_checkpoint_bytes = save_triple_active_checkpoint(
            active_checkpoint_path,
            checkpoint_config,
            checkpoint_metadata_provider,
            records,
            special_moves,
            sub_queue,
            auto_queue,
            complete_queue,
            insertion_counter,
            best_id,
            counters
        );
        if (!checkpoint_callback.is_none()) {
            py::dict heartbeat;
            heartbeat["checkpoint_path"] = active_checkpoint_path;
            heartbeat["checkpoint_count"] = counters.checkpoint_count;
            heartbeat["checkpoint_bytes"] = last_checkpoint_bytes;
            heartbeat["nodes_used"] = nodes_used;
            heartbeat["seen_state_count"] = seen.size();
            heartbeat["sub_queue_size"] = sub_queue.size();
            heartbeat["auto_queue_size"] = auto_queue.size();
            heartbeat["complete_queue_size"] = complete_queue.size();
            heartbeat["best_total_length"] = packed_length(records[best_id].state);
            heartbeat["minimum_total_length"] = minimum_length;
            checkpoint_callback(std::move(heartbeat));
        }
        next_checkpoint_time = CheckpointClock::now()
            + std::chrono::duration_cast<CheckpointClock::duration>(
                std::chrono::duration<double>(checkpoint_interval_seconds)
            );
        return stop_after_checkpoint;
    };

    auto next_progress_heartbeat_time = CheckpointClock::now()
        + std::chrono::duration_cast<CheckpointClock::duration>(
            std::chrono::duration<double>(progress_heartbeat_interval_seconds)
        );
    std::uint64_t last_progress_heartbeat_nodes = nodes_used;
    std::uint64_t last_resource_check_nodes = nodes_used;
    std::uint64_t last_resident_memory_bytes = current_resident_memory_bytes();
    std::uint64_t peak_observed_resident_memory_bytes = last_resident_memory_bytes;

    auto emit_progress_heartbeat = [&](const char* event_type) {
        if (progress_heartbeat_callback.is_none()) return;
        py::dict heartbeat;
        heartbeat["event_type"] = event_type;
        heartbeat["nodes_used"] = nodes_used;
        heartbeat["seen_state_count"] = seen.size();
        heartbeat["sub_queue_size"] = sub_queue.size();
        heartbeat["auto_queue_size"] = auto_queue.size();
        heartbeat["complete_queue_size"] = complete_queue.size();
        heartbeat["best_total_length"] = packed_length(records[best_id].state);
        heartbeat["minimum_total_length"] = minimum_length;
        heartbeat["frontier_prune_count"] = frontier_prune_count;
        heartbeat["frontier_states_pruned"] = frontier_states_pruned;
        heartbeat["resident_memory_bytes"] = last_resident_memory_bytes;
        heartbeat["peak_observed_resident_memory_bytes"] = (
            peak_observed_resident_memory_bytes
        );
        heartbeat["max_resident_memory_bytes"] = max_resident_memory_bytes;
        heartbeat["checkpoint_path"] = active_checkpoint_path;
        heartbeat["checkpoint_count"] = counters.checkpoint_count;
        heartbeat["checkpoint_bytes"] = last_checkpoint_bytes;
        heartbeat["elapsed_seconds"] = std::chrono::duration<double>(
            CheckpointClock::now() - search_started_at
        ).count();
        progress_heartbeat_callback(std::move(heartbeat));
    };

    auto maybe_progress_heartbeat = [&]() {
        if (progress_heartbeat_callback.is_none()) return;
        const auto now = CheckpointClock::now();
        const bool time_due = progress_heartbeat_interval_seconds > 0.0
            && now >= next_progress_heartbeat_time;
        const bool nodes_due = progress_heartbeat_interval_nodes > 0
            && nodes_used >= (
                last_progress_heartbeat_nodes + progress_heartbeat_interval_nodes
            );
        if (!time_due && !nodes_due) return;
        emit_progress_heartbeat("active");
        last_progress_heartbeat_nodes = nodes_used;
        next_progress_heartbeat_time = CheckpointClock::now()
            + std::chrono::duration_cast<CheckpointClock::duration>(
                std::chrono::duration<double>(progress_heartbeat_interval_seconds)
            );
    };

    emit_progress_heartbeat("started");

    while (nodes_used < total_node_budget) {
        maybe_progress_heartbeat();
        if (maybe_checkpoint()) {
            checkpoint_requested_stop = true;
            break;
        }
        discard_stale();
        if (sub_queue.empty() && auto_queue.empty() && complete_queue.empty()) break;

        int selected_queue = -1;
        int selected_score = std::numeric_limits<int>::max();
        int selected_rank = std::numeric_limits<int>::max();
        auto consider_queue = [&](const MinQueue& queue, int offset, int rank) {
            if (queue.empty()) return;
            const int score = static_cast<int>(queue.top().priority.score) + offset;
            if (score < selected_score || (score == selected_score && rank < selected_rank)) {
                selected_queue = rank;
                selected_score = score;
                selected_rank = rank;
            }
        };
        consider_queue(
            separate_sub_activation_index ? sub_activation_queue : sub_queue,
            0,
            0
        );
        consider_queue(auto_queue, auto_preference_threshold, 1);
        consider_queue(complete_queue, complete_preference_threshold, 2);

        if (selected_queue == 0) {
            const QueueEntry item = sub_queue.top();
            sub_queue.pop();
            StateRecord& current = records[item.state_id];
            if (!current.alive || current.sub_expanded) continue;
            current.sub_expanded = true;
            ++nodes_used;
            ++sub_pop_count;
            update_best(item.state_id);

            const PackedState current_state = current.state;
            const auto current_depth = current.depth;
            const auto current_gap = current.standard_gap;
            if (current_state.first.length == 1 && current_state.second.length == 1) {
                solved = true;
                final_id = item.state_id;
                break;
            }

            const auto words = unpack_state(current_state);
            const auto successors = standard_successors(
                words.first,
                words.second,
                max_total_length
            );
            for (const auto& successor : successors) {
                ++standard_generated_count;
                const auto inserted = enqueue_new_state(
                    pack_state(successor.first, successor.second),
                    item.state_id,
                    current_depth + 1,
                    current_gap + 1,
                    MoveKind::Standard,
                    py::none()
                );
                if (inserted.first) {
                    ++standard_enqueued_count;
                    update_best(inserted.second);
                }
            }

            StateRecord& refreshed = records[item.state_id];
            if (!refreshed.cov_queued) {
                if (
                    current_gap
                    >= static_cast<std::uint32_t>(min_standard_moves_between_auto)
                ) {
                    refreshed.cov_queued = true;
                    auto_queue.push(
                        {activation_priority_for(current_state, current_depth), item.state_id}
                    );
                    ++auto_queue_eligible_count;
                    ++auto_queued_count;
                } else {
                    ++auto_queue_distance_blocked_count;
                }
            }
            if (!refreshed.complete_cov_queued) {
                if (complete_cov_composite_eligible(current_state)) {
                    refreshed.complete_cov_queued = true;
                    complete_queue.push(
                        {activation_priority_for(current_state, current_depth), item.state_id}
                    );
                    ++complete_prefilter_accepted_count;
                    ++complete_queued_count;
                } else {
                    ++complete_prefilter_rejected_count;
                }
            }
        } else if (selected_queue == 1) {
            const QueueEntry item = auto_queue.top();
            auto_queue.pop();
            StateRecord& current = records[item.state_id];
            if (!current.alive || current.cov_expanded) continue;
            current.cov_expanded = true;
            ++nodes_used;
            ++auto_pop_count;
            update_best(item.state_id);
            invoke_auto(item.state_id, "auto_cov_activation");
        } else {
            const QueueEntry item = complete_queue.top();
            complete_queue.pop();
            StateRecord& current = records[item.state_id];
            if (!current.alive || current.complete_cov_expanded) continue;
            current.complete_cov_expanded = true;
            const auto current_depth = current.depth;
            ++complete_generation_call_count;

            const auto words = unpack_state(current.state);
            const py::tuple callback_result = complete_expander(
                py::make_tuple(words.first, words.second),
                py::str("complete_cov_activation")
            ).cast<py::tuple>();
            const py::list neighbors = callback_result[0].cast<py::list>();
            complete_neighbor_count += neighbors.size();
            if (neighbors.empty()) {
                ++complete_ineligible_pop_count;
                continue;
            }

            ++nodes_used;
            ++complete_pop_count;
            update_best(item.state_id);
            if (nodes_used < total_node_budget) {
                for (const py::handle item_handle : neighbors) {
                    const py::tuple neighbor_item = py::reinterpret_borrow<py::tuple>(item_handle);
                    const py::tuple next_state = neighbor_item[0].cast<py::tuple>();
                    const PackedState packed = pack_state(
                        next_state[0].cast<std::string>(),
                        next_state[1].cast<std::string>()
                    );
                    const auto inserted = enqueue_new_state(
                        packed,
                        item.state_id,
                        current_depth + 1,
                        0,
                        MoveKind::CompleteCov,
                        py::reinterpret_borrow<py::object>(neighbor_item[1])
                    );
                    if (inserted.first) {
                        ++complete_direct_enqueued_count;
                        update_best(inserted.second);
                    } else {
                        ++complete_seen_neighbor_count;
                    }
                }
            }
        }

        const auto live_frontier = sub_queue.size() + auto_queue.size() + complete_queue.size();
        frontier_peak = std::max<std::uint64_t>(frontier_peak, live_frontier);
        if (
            max_resident_memory_bytes > 0
            && nodes_used >= last_resource_check_nodes + resource_check_interval_nodes
        ) {
            last_resource_check_nodes = nodes_used;
            last_resident_memory_bytes = current_resident_memory_bytes();
            peak_observed_resident_memory_bytes = std::max(
                peak_observed_resident_memory_bytes,
                last_resident_memory_bytes
            );
            if (
                last_resident_memory_bytes > 0
                && last_resident_memory_bytes >= max_resident_memory_bytes
            ) {
                memory_limit_reached = true;
                emit_progress_heartbeat("memory_limit");
                break;
            }
        }
        prune_frontier();
        if (progress_interval > 0 && nodes_used > 0 && nodes_used % progress_interval == 0) {
            std::cout
                << "nodes=" << nodes_used
                << " sub_q=" << sub_queue.size()
                << " auto_q=" << auto_queue.size()
                << " complete_q=" << complete_queue.size()
                << " seen=" << seen.size()
                << " best_len=" << packed_length(records[best_id].state)
                << std::endl;
        }
    }

    if (!memory_limit_reached) {
        last_resident_memory_bytes = current_resident_memory_bytes();
        peak_observed_resident_memory_bytes = std::max(
            peak_observed_resident_memory_bytes,
            last_resident_memory_bytes
        );
        emit_progress_heartbeat(solved ? "solved" : "finished");
    }

    discard_stale();
    if (!solved) final_id = best_id;

    std::vector<StateId> path_ids;
    for (StateId id = final_id; id != NO_STATE; id = records[id].parent) {
        path_ids.push_back(id);
    }
    std::reverse(path_ids.begin(), path_ids.end());

    py::list path;
    py::list moves;
    std::uint64_t auto_moves_in_path = 0;
    std::uint64_t complete_moves_in_path = 0;
    for (const StateId id : path_ids) path.append(state_tuple(records[id].state));
    for (std::size_t index = 1; index < path_ids.size(); ++index) {
        const StateId id = path_ids[index];
        const StateId parent = path_ids[index - 1];
        if (
            records[id].move_kind == MoveKind::Cov
            || records[id].move_kind == MoveKind::CompleteCov
        ) {
            moves.append(special_moves.at(id));
            auto_moves_in_path += records[id].move_kind == MoveKind::Cov;
            complete_moves_in_path += records[id].move_kind == MoveKind::CompleteCov;
        } else {
            py::dict move;
            move["move_type"] = "standard_substitution";
            move["from_state"] = state_tuple(records[parent].state);
            move["to_state"] = state_tuple(records[id].state);
            move["source_queue"] = "sub";
            moves.append(std::move(move));
        }
    }

    std::vector<PackedState> ordered_minimum_states(
        minimum_states.begin(), minimum_states.end()
    );
    std::sort(ordered_minimum_states.begin(), ordered_minimum_states.end(), packed_solver_less);
    py::list minimum_state_list;
    for (const auto& state : ordered_minimum_states) {
        minimum_state_list.append(state_tuple(state));
    }

    py::dict result;
    result["canonical_start"] = state_tuple(start_state);
    result["solved"] = solved;
    result["solved_trivial"] = solved;
    result["termination_reason"] = solved
        ? "solved"
        : (
            memory_limit_reached
            ? "memory_limit"
            : (
                checkpoint_requested_stop
                ? "checkpoint_requested_stop"
                : (
                    nodes_used >= total_node_budget
                    ? "node_budget_exhausted"
                    : "queues_exhausted"
                )
            )
        );
    result["total_nodes_used"] = nodes_used;
    result["path_length"] = path_ids.size() - 1;
    result["path"] = path;
    result["moves"] = moves;
    result["final_state"] = state_tuple(records[final_id].state);
    result["final_total_length"] = packed_length(records[final_id].state);
    result["final_depth"] = records[final_id].depth;
    result["minimum_total_length"] = minimum_length;
    result["minimum_state_count"] = ordered_minimum_states.size();
    result["minimum_length_states"] = minimum_state_list;
    result["seen_state_count"] = seen.size();
    result["sub_queue_size"] = sub_queue.size();
    result["auto_cov_queue_size"] = auto_queue.size();
    result["complete_cov_queue_size"] = complete_queue.size();
    result["frontier_peak"] = frontier_peak;
    result["sub_pop_count"] = sub_pop_count;
    result["auto_cov_queue_pop_count"] = auto_pop_count;
    result["auto_cov_queued_count"] = auto_queued_count;
    result["auto_cov_queue_eligible_count"] = auto_queue_eligible_count;
    result["auto_cov_queue_distance_blocked_count"] = auto_queue_distance_blocked_count;
    result["complete_cov_queue_pop_count"] = complete_pop_count;
    result["complete_cov_queued_count"] = complete_queued_count;
    result["complete_cov_ineligible_pop_count"] = complete_ineligible_pop_count;
    result["standard_generated_count"] = standard_generated_count;
    result["standard_enqueued_count"] = standard_enqueued_count;
    result["auto_cov_generated_count"] = auto_generated_count;
    result["auto_cov_valid_count"] = auto_valid_count;
    result["auto_cov_enqueued_count"] = auto_enqueued_count;
    result["auto_cov_generation_call_count"] = auto_generation_call_count;
    result["initial_auto_cov_seed_enqueued_count"] = initial_auto_seed_enqueued_count;
    result["initial_complete_cov_seed_enqueued_count"] = initial_complete_seed_enqueued_count;
    result["auto_cov_moves_in_path"] = auto_moves_in_path;
    result["complete_cov_generation_call_count"] = complete_generation_call_count;
    result["complete_cov_unique_neighbor_count"] = complete_neighbor_count;
    result["complete_cov_seen_neighbor_count"] = complete_seen_neighbor_count;
    result["complete_cov_direct_enqueued_count"] = complete_direct_enqueued_count;
    result["complete_cov_moves_in_path"] = complete_moves_in_path;
    result["complete_cov_prefilter_accepted_count"] = complete_prefilter_accepted_count;
    result["complete_cov_prefilter_rejected_count"] = complete_prefilter_rejected_count;
    result["auto_cov_preference_threshold"] = auto_preference_threshold;
    result["complete_cov_preference_threshold"] = complete_preference_threshold;
    result["auto_cov_required_length_advantage"] = auto_preference_threshold + 1;
    result["complete_cov_required_length_advantage"] = complete_preference_threshold + 1;
    result["seed_initial_auto_cov_neighbors"] = seed_initial_auto_neighbors;
    result["seed_initial_complete_cov_neighbors"] = seed_initial_complete_neighbors;
    result["min_standard_moves_between_auto_cov"] = min_standard_moves_between_auto;
    result["depth_tie_strategy"] = depth_tie_strategy;
    result["queue_policy"] = "sub_auto_complete_adjusted_length";
    result["queue_tie_order"] = "sub_auto_complete";
    result["sub_queue_score_mode"] = sub_queue_score_mode_name;
    result["cov_activation_score_mode"] = cov_activation_score_mode_name;
    result["cov_activation_score_name"] = activation_uses_queue_score
        ? sub_queue_score_mode_name
        : "total_length";
    result["matrix_score_bound"] = MASS_SCORE_BOUND;
    result["matrix_score_fallback"] = MASS_SCORE_FALLBACK;
    result["max_live_frontier_states"] = max_live_frontier_states;
    result["frontier_prune_count"] = frontier_prune_count;
    result["frontier_states_pruned"] = frontier_states_pruned;
    result["active_checkpoint_path"] = active_checkpoint_path;
    result["active_checkpoint_interval_seconds"] = checkpoint_interval_seconds;
    result["active_checkpoint_count"] = counters.checkpoint_count;
    result["active_checkpoint_last_nodes"] = counters.last_checkpoint_nodes;
    result["active_checkpoint_bytes"] = last_checkpoint_bytes;
    result["resumed_from_active_checkpoint"] = resumed_from_active_checkpoint;
    result["progress_heartbeat_interval_seconds"] = (
        progress_heartbeat_interval_seconds
    );
    result["progress_heartbeat_interval_nodes"] = progress_heartbeat_interval_nodes;
    result["max_resident_memory_bytes"] = max_resident_memory_bytes;
    result["resource_check_interval_nodes"] = resource_check_interval_nodes;
    result["last_resident_memory_bytes"] = last_resident_memory_bytes;
    result["peak_observed_resident_memory_bytes"] = (
        peak_observed_resident_memory_bytes
    );
    result["memory_limit_reached"] = memory_limit_reached;
    result["storage_backend"] = "compact_cpp_triple";
    result["packed_state_bytes"] = sizeof(PackedState);
    result["allocated_state_slots"] = records.size();
    return result;
}

}  // namespace

#ifndef DUAL_GS_MODULE
#define DUAL_GS_MODULE native_standard_successors
#endif

PYBIND11_MODULE(DUAL_GS_MODULE, module) {
    module.doc() = "Fused C++ standard-substitution successor generation for Dual GS.";
    module.def(
        "standard_successors",
        &standard_successors,
        py::arg("r1"),
        py::arg("r2"),
        py::arg("max_total_length")
    );
    module.def("reduce_word", &reduce_word, py::arg("word"));
    module.def("canonical_word", &canonical_word, py::arg("word"));
    module.def("canonical_pair", &canonical_pair, py::arg("r1"), py::arg("r2"));
    module.def(
        "automorphic_equivalence_key",
        &automorphic_equivalence_key_native,
        py::arg("r1"),
        py::arg("r2")
    );
    module.def(
        "read_triple_active_checkpoint_metadata",
        &read_triple_active_checkpoint_metadata,
        py::arg("checkpoint_path")
    );
    module.def(
        "run_compact_dual_gs",
        &run_compact_dual_gs,
        py::arg("start_r1"),
        py::arg("start_r2"),
        py::arg("cov_expander"),
        py::arg("total_node_budget") = 1'000'000,
        py::arg("max_total_length") = 50,
        py::arg("cov_preference_threshold") = 2,
        py::arg("seed_initial_cov_neighbors") = true,
        py::arg("min_standard_moves_between_cov") = 10,
        py::arg("depth_tie_strategy") = "deepest",
        py::arg("max_live_frontier_states") = 5'000'000,
        py::arg("progress_interval") = 0,
        py::arg("expand_cov_on_every_sub_pop") = false,
        py::arg("sub_queue_score_mode") = "length"
    );
    module.def(
        "run_compact_triple_gs",
        &run_compact_triple_gs,
        py::arg("start_r1"),
        py::arg("start_r2"),
        py::arg("auto_expander"),
        py::arg("complete_expander"),
        py::arg("total_node_budget") = 1'000'000,
        py::arg("max_total_length") = 50,
        py::arg("auto_preference_threshold") = 2,
        py::arg("complete_preference_threshold") = 2,
        py::arg("seed_initial_auto_neighbors") = true,
        py::arg("seed_initial_complete_neighbors") = false,
        py::arg("min_standard_moves_between_auto") = 0,
        py::arg("depth_tie_strategy") = "deepest",
        py::arg("max_live_frontier_states") = 5'000'000,
        py::arg("progress_interval") = 0,
        py::arg("active_checkpoint_path") = "",
        py::arg("checkpoint_interval_seconds") = 0.0,
        py::arg("resume_active_checkpoint") = true,
        py::arg("checkpoint_interval_nodes") = 0,
        py::arg("stop_after_checkpoint") = false,
        py::arg("checkpoint_metadata_provider") = py::none(),
        py::arg("checkpoint_callback") = py::none(),
        py::arg("progress_heartbeat_interval_seconds") = 0.0,
        py::arg("progress_heartbeat_interval_nodes") = 0,
        py::arg("progress_heartbeat_callback") = py::none(),
        py::arg("max_resident_memory_bytes") = 0,
        py::arg("resource_check_interval_nodes") = 10'000,
        py::arg("sub_queue_score_mode") = "length",
        py::arg("cov_activation_score_mode") = "total_length"
    );
}
