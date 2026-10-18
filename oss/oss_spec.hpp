// oss/oss_spec.hpp - sampling and speculative decoding for strata-oss.
//
// Drafts come from prompt lookup (the continuation of an earlier occurrence of the last few tokens, as Strata's
// suffix drafter does for code edits and quoted text).  One decode step checks the last token plus the drafts; the
// target keeps the drafts it agrees with and adds its own next token, so every token is still the target's:
//   * greedy: a draft is kept when it is the argmax;
//   * sampling: a draft x (a point-mass proposal) is kept with probability p(x), else the token is drawn from p with
//     x removed - the standard speculative-sampling rule, an exact sample of p.
// How many tokens to draft comes from the measured step costs (a step that checks k tokens costs cost[k] steps of
// one) and the drafter's running acceptance rate: the k with the most expected tokens per unit of cost, 0 = no
// draft.
#pragma once

#include "oss_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace oss {

struct Sampler {
    double temp = 0, top_p = 1, min_p = 0;
    int top_k = 0;
    std::mt19937_64 rng{1};

    struct Cand { int id; double p; };
    // the sampling distribution of the logits (candidates with p > 0, descending)
    void dist(const float* lg, int V, std::vector<Cand>& out) const {
        out.clear();
        const float mx = *std::max_element(lg, lg + V);
        const float cut = mx - 30.f * (float) temp;
        std::vector<std::pair<float, int>> c;
        for (int i = 0; i < V; ++i)
            if (lg[i] >= cut) c.push_back({lg[i], i});
        std::sort(c.begin(), c.end(), [](auto& a, auto& b) { return a.first > b.first; });
        if (top_k > 0 && (int) c.size() > top_k) c.resize(top_k);
        double s = 0;
        for (auto& x : c) {
            out.push_back({x.second, std::exp((x.first - mx) / temp)});
            s += out.back().p;
        }
        for (auto& x : out) x.p /= s;
        size_t n = out.size();
        if (top_p < 1) {
            double acc = 0;
            for (size_t i = 0; i < out.size(); ++i) { acc += out[i].p; if (acc >= top_p) { n = i + 1; break; } }
        }
        if (min_p > 0)
            for (size_t i = 1; i < n; ++i) if (out[i].p < min_p * out[0].p) { n = i; break; }
        out.resize(n);
        double t = 0;
        for (auto& x : out) t += x.p;
        for (auto& x : out) x.p /= t;
    }
    int draw(const std::vector<Cand>& d) {
        std::uniform_real_distribution<double> u(0, 1);
        double r = u(rng), c = 0;
        for (auto& x : d) { c += x.p; if (r <= c) return x.id; }
        return d.back().id;
    }
    int pick(const float* lg, int V) {
        if (temp <= 0) return (int) (std::max_element(lg, lg + V) - lg);
        std::vector<Cand> d;
        dist(lg, V, d);
        return draw(d);
    }
    // speculative check of `draft` against these logits: true = keep it; false = `out` is the replacement
    bool check(const float* lg, int V, int draft, int& out) {
        if (temp <= 0) {
            out = (int) (std::max_element(lg, lg + V) - lg);
            return out == draft;
        }
        std::vector<Cand> d;
        dist(lg, V, d);
        double px = 0;
        for (auto& x : d) if (x.id == draft) px = x.p;
        std::uniform_real_distribution<double> u(0, 1);
        if (px > 0 && u(rng) < px) { out = draft; return true; }
        // the residual max(p - q, 0) with q a point mass on the draft: p without the draft, renormalized
        std::vector<Cand> r;
        for (auto& x : d) if (x.id != draft) r.push_back(x);
        double t = 0;
        for (auto& x : r) t += x.p;
        if (r.empty() || t <= 0) { out = draft; return true; }
        for (auto& x : r) x.p /= t;
        out = draw(r);
        return false;
    }
};

/// Prompt lookup over the conversation so far.
struct LookupDrafter {
    int min_ngram = 2, max_ngram = 4;
    // drafts up to `k` tokens continuing the latest earlier match of the history's last n tokens (longest n first)
    int draft(const std::vector<int32_t>& h, int k, std::vector<int32_t>& out) const {
        out.clear();
        const int n = (int) h.size();
        for (int g = std::min(max_ngram, n - 1); g >= min_ngram; --g) {
            const int32_t* tail = h.data() + n - g;
            // newest match first, but not the tail itself
            for (int i = n - g - 1; i >= 0; --i) {
                if (h[i] != tail[0] || !std::equal(tail, tail + g, h.data() + i)) continue;
                const int start = i + g;
                for (int j = start; j < n && (int) out.size() < k; ++j) out.push_back(h[j]);
                if (!out.empty()) return g;
            }
        }
        return 0;
    }
};

struct SpecStats {
    uint64_t steps = 0, drafted = 0, accepted = 0, emitted = 0, eagle_drafted = 0, eagle_kept = 0;
};

/// Generates with optional speculation.  `cost[k]` = the cost of a step checking k + 1 tokens, in single steps.
class Generator {
public:
    Generator(Engine& e, Sampler& s) : eng_(e), smp_(s) {}
    bool spec = true;
    int max_draft = 7;
    double cost[8] = {1.0, 1.29, 1.54, 1.79, 2.05, 2.29, 2.55, 2.87};  // 120b, measured (FINDINGS.md section 13)
    SpecStats st;
    int eagle_max = 6;         // EAGLE3 drafts a step at most (it stops earlier on its own confidence)
    double draft_cost = 0.09;  // one EAGLE3 draft token, in target steps: 1.07 ms vs ~12 ms (FINDINGS.md section 15)

    /// Starts an answer: `last` = the logits after the prompt.
    void begin(const float* last) {
        const int V = eng_.n_vocab();
        pending_ = smp_.pick(last, V);
    }
    /// The next tokens of the answer (at least one).  The first call returns the token sampled by begin().
    /// `stop`: tokens that end the answer (nothing is evaluated past them).
    std::vector<int32_t> next(const std::vector<int32_t>& stop) {
        std::vector<int32_t> out;
        const int V = eng_.n_vocab();
        const int32_t t = pending_;
        out.push_back(t);
        if (std::find(stop.begin(), stop.end(), t) != stop.end()) return out;
        // drafts for what follows t: prompt lookup first (free), else the EAGLE3 draft model
        std::vector<int32_t> h = eng_.history();
        h.push_back(t);
        std::vector<int32_t> d;
        src_ = 0;
        if (spec) {
            const int room = std::max(0, eng_.n_ctx() - eng_.n_past() - 2);
            const int kl = std::min(choose_k(0, 0.0), room);
            if (kl > 0) lookup_.draft(h, kl, d);
            if (d.empty() && eng_.has_draft()) {
                // the draft model stops itself when it is not confident; the cost model only caps the length
                const int ke = std::min(std::max(choose_k(1, draft_cost), eagle_max), room);
                if (ke > 0) { eng_.draft(t, ke, d); src_ = 1; }
            }
        }
        std::vector<int32_t> win{t};
        win.insert(win.end(), d.begin(), d.end());
        const int base = eng_.n_past();
        const float* lg = eng_.eval(win, true);
        ++st.steps;
        st.drafted += d.size();
        // keep the drafts the target agrees with
        size_t j = 0;
        int next_tok = 0;
        for (; j < d.size(); ++j) {
            int o;
            if (!smp_.check(lg + j * (size_t) V, V, d[j], o)) { next_tok = o; break; }
            out.push_back(d[j]);
            if (std::find(stop.begin(), stop.end(), d[j]) != stop.end()) {
                // a kept stop token ends the answer: evaluate nothing more
                eng_.truncate(base + 1 + (int) j);
                st.accepted += j + 1;
                update(src_, d.size(), j + 1);
                st.emitted += out.size();
                pending_ = -1;
                return out;
            }
        }
        if (j == d.size()) next_tok = smp_.pick(lg + j * (size_t) V, V);
        st.accepted += j;
        if (!d.empty()) update(src_, d.size(), j);
        eng_.truncate(base + 1 + (int) j);
        pending_ = next_tok;
        st.emitted += out.size();
        return out;
    }

private:
    // The draft length with the most expected tokens per unit of cost.  Acceptance is tracked per depth as a
    // conditional rate (the i-th draft kept given the ones before it were): drafts come in runs, so a deeper draft
    // is often more likely kept than the first.  `per_draft`: the cost of making one draft, in target steps.
    int choose_k(int src, double per_draft) const {
        double best = 1.0 / cost[0];
        int bk = 0;
        double e = 1, p = 1;
        for (int k = 1; k <= std::min(max_draft, 7); ++k) {
            p *= cond(src, k - 1);
            e += p;
            const double v = e / (cost[k] + per_draft * k);
            if (v > best * 1.02) { best = v; bk = k; }
        }
        // probe now and then so a drafter that stopped paying can come back
        if (bk == 0 && (st.steps % 16) == 0) bk = 2;
        return bk;
    }
    double cond(int src, int i) const {
        // a prior of 0.6 worth 2 trials, so an untried depth is neither ruled out nor trusted
        return (kept_[src][i] + 1.2) / (tried_[src][i] + 2.0);
    }
    void update(int src, size_t drafted, size_t kept) {
        // depth i was tried when the drafts before it were kept
        for (size_t i = 0; i < drafted && i <= kept; ++i) {
            tried_[src][i] = 0.98 * tried_[src][i] + 1.0;
            kept_[src][i] = 0.98 * kept_[src][i] + (i < kept ? 1.0 : 0.0);
        }
        if (src == 1) { st.eagle_drafted += drafted; st.eagle_kept += kept; }
    }
    Engine& eng_;
    Sampler& smp_;
    LookupDrafter lookup_;
    int32_t pending_ = -1;
    double kept_[2][8] = {}, tried_[2][8] = {};
    int src_ = 0;
};

} // namespace oss
