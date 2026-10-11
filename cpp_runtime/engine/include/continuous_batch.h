#pragma once
#include "generation_step.h"
#include <deque>
#include <memory>
#include <unordered_map>

namespace mfq::engine {

// Physical row metadata only. Request lifecycle and output belong to the executor.
struct BatchRequest {
    BatchRequest(std::string id, const InferenceRequest& input)
        : id(std::move(id)), prompt(input.prompt), sampling(input.sampling),
          token_constraint(input.token_constraint), generation_limit(input.sampling.max_tokens) {}
    std::string id;
    std::vector<int64_t> prompt;
    MfqSamplingParams sampling;
    MfqTokenConstraintPtr token_constraint;
    int32_t generation_limit = 0, produced = 0;
    int64_t pending_token = 0;
    bool eligible = false;
    int64_t prefill_offset = 0, cache_length = 0;
};

template <class Request> struct BatchState {
    std::deque<std::shared_ptr<Request>> prefilling;
    std::vector<std::shared_ptr<Request>> active;
};

// Groups pending numerical operations. All rows run generate_sequence(), also
// used by B=1 execution; this component never accepts tokens or publishes events.
template <class Ops> class ContinuousBatch {
    using Request = typename Ops::Request;
    using Sample = typename Ops::Sample;
    using State = BatchState<Request>;
    struct Work {
        std::shared_ptr<Request> request;
        std::optional<PrefillChunk> prefill;
        std::optional<Sample> result;
        std::exception_ptr failure;
        bool decode = false, active = false;
    };
    int64_t prefill_chunk_size;
    Ops operations;
    State state;
    std::unordered_map<std::string, std::shared_ptr<Work>> requests;
    bool decode_next = true;
    int64_t admissions = 0, prefill_chunks = 0, prefill_yields = 0;

    int64_t cache_position() const {
        int64_t position = 0;
        for (const auto& request : state.active)
            position = std::max(position, request->cache_length);
        return position;
    }
    void recover(std::exception_ptr error) {
        State failed;
        for (const auto& [id, work] : requests) {
            work->failure = error;
            work->prefill.reset();
            work->decode = work->active = false;
            failed.prefilling.push_back(work->request);
        }
        state = {};
        operations.recover(failed);
    }
    void release(Work& work) {
        if (!requests.contains(work.request->id)) return;
        try {
            if (work.active) {
                std::erase(state.active, work.request);
                operations.retire({work.request}, cache_position());
            } else if (!work.failure) {
                operations.suspend_decode();
                operations.discard_prefill(work.request);
                operations.resume_decode(cache_position());
            }
        } catch (...) {
            auto error = std::current_exception();
            try { recover(error); } catch (...) { error = std::current_exception(); }
            requests.erase(work.request->id);
            std::rethrow_exception(error);
        }
        requests.erase(work.request->id);
    }
    struct SequenceOps {
        ContinuousBatch& batch;
        Work& work;
        PrefillChunk prefill_chunk(PrefillChunk chunk) {
            if constexpr (requires { batch.operations.prefill_chunk(work.request, chunk); })
                return batch.operations.prefill_chunk(work.request, chunk);
            return chunk;
        }
        void schedule_prefill(PrefillChunk chunk) { work.result.reset(); work.prefill = chunk; }
        void schedule_decode() { work.result.reset(); work.decode = true; }
        bool ready() const { return work.result.has_value() || work.failure; }
        const Sample& sample() const {
            if (work.failure) std::rethrow_exception(work.failure);
            return work.result.value();
        }
        double prefill(PrefillChunk) { return sample().timing.llm_ms; }
        int64_t first_token() { return sample().token; }
        int64_t advance() { return sample().token; }
        std::optional<Generation> generate_decode(InferenceOutput& output) {
            if constexpr (requires { batch.operations.generate_decode(work.request, output,
                    std::function<void()>{}, std::function<bool()>{}, std::function<void()>{}); })
                return batch.operations.generate_decode(work.request, output,
                    [this] { schedule_decode(); }, [this] { return ready(); },
                    [this] { (void)sample(); });
            return std::nullopt;
        }
        void accept(int64_t token) {
            auto& request = *work.request;
            // activate() initializes counts for the first sample.
            if (request.produced) batch.operations.accept(work.request, sample());
            request.pending_token = token;
            ++request.produced;
        }
    };

  public:
    bool empty() const noexcept { return requests.empty(); }
    template <class... Args>
    explicit ContinuousBatch(int64_t chunk_size, int64_t token_budget, Args&&... args)
        : prefill_chunk_size(std::min(chunk_size, token_budget)),
          operations(std::forward<Args>(args)...) {
        if (prefill_chunk_size <= 0)
            throw std::invalid_argument("continuous batching requires a positive prefill chunk");
    }

    Generation generate(std::string id, InferenceRequest& input, InferenceOutput& output) {
        if (output.stopped()) co_return;
        auto work = std::make_shared<Work>();
        work->request = std::make_shared<Request>(std::move(id), input);
        if (!requests.emplace(work->request->id, work).second)
            throw std::invalid_argument("duplicate batch request id");
        ExecutionCleanup cleanup{output.cleanup_failure, [&] { release(*work); }};
        SequenceOps ops{*this, *work};
        int64_t reused = 0, prefix = 0;
        if constexpr (requires { operations.prepare(work->request); })
            reused = operations.prepare(work->request);
        if constexpr (requires { operations.prefix_tokens(work->request); })
            prefix = operations.prefix_tokens(work->request);
        auto sequence = generate_sequence(ops, output, input.prompt.size(), reused, prefix, prefill_chunk_size);
        while (auto step = sequence.next()) co_yield std::move(step);
        cleanup.finish();
    }

    void step(const std::vector<std::string>& eligible) {
        for (const auto& request : state.active) {
            const auto& work = *requests.at(request->id);
            request->eligible = work.decode &&
                std::find(eligible.begin(), eligible.end(), request->id) != eligible.end();
        }
        auto prefill = requests.end();
        for (const auto& id : eligible) {
            auto it = requests.find(id);
            if (it != requests.end() && it->second->prefill) { prefill = it; break; }
        }
        const bool rows_ready = std::all_of(state.active.begin(), state.active.end(),
            [&](const auto& request) {
                return std::find(eligible.begin(), eligible.end(), request->id) == eligible.end() ||
                    requests.at(request->id)->decode;
            });
        // Let eligible peers consume their previous sample and schedule the
        // next decode before executing it. Otherwise staggered coroutines
        // alternate single-row decodes and never form a stable batch.
        const bool has_decode = rows_ready && std::any_of(state.active.begin(), state.active.end(),
            [](const auto& request) { return request->eligible; });
        bool run_decode = has_decode && (prefill == requests.end() || decode_next);
        if constexpr (requires { operations.ready_decode(state, eligible); })
            run_decode = run_decode && operations.ready_decode(state, eligible);
        std::vector<std::shared_ptr<Work>> prefills;
        bool prefilling_operation = false;
        bool decoding_operation = false;
        try {
            if constexpr (requires { operations.synchronize(state); }) operations.synchronize(state);
            if (run_decode) {
                decoding_operation = true;
                auto decoded = operations.decode(state);
                decoding_operation = false;
                for (const auto& request : state.active) if (request->eligible) {
                    auto& work = *requests.at(request->id);
                    work.result = operations.sample(request, decoded);
                    if constexpr (requires { operations.cache_length(request); })
                        request->cache_length = operations.cache_length(request);
                    else ++request->cache_length;
                    work.decode = false;
                }
            } else if (prefill != requests.end()) {
                std::vector<std::pair<std::shared_ptr<Request>, PrefillChunk>> chunks;
                int64_t remaining = prefill_chunk_size;
                for (const auto& id : eligible) {
                    const auto it = requests.find(id);
                    if (it == requests.end() || !it->second->prefill || it->second->prefill->count > remaining) continue;
                    auto work = it->second;
                    chunks.emplace_back(work->request, *work->prefill);
                    prefills.push_back(work);
                    remaining -= work->prefill->count;
                    if constexpr (!requires { operations.prefill_batch(chunks); }) break;
                }
                operations.suspend_decode();
                PrefillActivity prefill_activity;
                prefilling_operation = true;
                auto samples = [&] {
                    if constexpr (requires { operations.prefill_batch(chunks); }) return operations.prefill_batch(chunks);
                    else return std::vector<Sample>{operations.prefill(chunks.front().first, chunks.front().second)};
                }();
                prefilling_operation = false;
                if (samples.size() != prefills.size()) throw std::logic_error("prefill batch returned an invalid row count");
                for (std::size_t row = 0; row < prefills.size(); ++row) {
                    auto& work = *prefills[row];
                    auto& request = work.request;
                    const auto chunk = *work.prefill;
                    request->prefill_offset = chunk.offset + chunk.count;
                    ++prefill_chunks;
                    if (request->prefill_offset == static_cast<int64_t>(request->prompt.size())) {
                        request->cache_length = request->prompt.size();
                        operations.activate(request, samples[row]);
                        state.active.push_back(request);
                        work.active = true;
                        ++admissions;
                    } else ++prefill_yields;
                    work.result = std::move(samples[row]);
                    work.prefill.reset();
                }
                operations.resume_decode(cache_position());
            }
            decode_next = !decode_next;
        } catch (...) {
            if constexpr (requires { Ops::isolate_context_growth_failures; }) {
                bool capacity_failure = false;
                try { throw; }
                catch (const InferenceInputError& error) {
                    capacity_failure = error.code == InferenceInputErrorCode::ContextLength;
                } catch (...) {}
                if (Ops::isolate_context_growth_failures && decoding_operation && capacity_failure) {
                    // Capacity checks run before numerical decode. Release the
                    // newest eligible request while preserving established rows.
                    const auto victim = std::find_if(state.active.rbegin(),state.active.rend(),
                        [](const auto& request) { return request->eligible; });
                    if (victim != state.active.rend()) {
                        auto request = *victim;
                        auto& work = *requests.at(request->id);
                        work.failure = std::current_exception(); work.decode = false;
                        std::erase(state.active,request);
                        try {
                            operations.retire({request},cache_position());
                            work.active = false;
                            return;
                        } catch (...) { recover(std::current_exception()); throw; }
                    }
                }
            }
            if constexpr (requires { Ops::isolate_prefill_failures; }) {
                if (Ops::isolate_prefill_failures && prefilling_operation && prefill != requests.end()) {
                    try {
                        for (const auto& item : prefills) {
                            auto& work = *item;
                            work.failure = std::current_exception();
                            work.prefill.reset();
                            work.decode = work.active = false;
                            std::erase(state.active, work.request);
                            operations.discard_prefill(work.request);
                        }
                        operations.resume_decode(cache_position());
                        return;
                    } catch (...) { recover(std::current_exception()); throw; }
                }
            }
            const auto error = std::current_exception();
            if constexpr (requires { operations.recoverable_failure(error); }) {
                if (operations.recoverable_failure(error)) {
                    if (!run_decode && prefill != requests.end()) {
                        for (const auto& item : prefills) {
                            auto& work = *item;
                            work.failure = error;
                            work.prefill.reset();
                            work.decode = work.active = false;
                            std::erase(state.active, work.request);
                            operations.discard_prefill(work.request);
                        }
                        operations.resume_decode(cache_position());
                    } else recover(error);
                    return;
                }
            }
            recover(error);
            throw;
        }
    }

    Metrics metrics() const {
        auto result = operations.metrics();
        result.emplace_back("continuous_batching_active", state.active.size());
        result.emplace_back("continuous_batching_prefilling", requests.size() - state.active.size());
        result.emplace_back("continuous_batching_requests", admissions);
        result.emplace_back("continuous_batching_admissions", admissions);
        result.emplace_back("continuous_batching_prefill_chunks", prefill_chunks);
        result.emplace_back("continuous_batching_prefill_yields", prefill_yields);
        return result;
    }
};
} // namespace mfq::engine
