// oss/oss_main.cpp - strata-oss: the GPT-OSS engine's command line.
//
//   strata-oss --model gpt-oss-120b-MXFP4.gguf [--gpus 0,1] [--ctx 32768] ...
//     --tokens FILE [--n-predict N]     greedy generation from token ids (whitespace separated), prints ids + speed
//     --dump-logits OUT --tokens FILE   teacher-forced logits of every position (float32 [n][vocab]) to OUT
//     --serve                           JSON lines on stdin/stdout (oss/server.py speaks it)
#include "oss_engine.hpp"
#include "oss_spec.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using oss::Sampler;

namespace {

// ---------------------------------------------------------------- a small JSON reader / writer
struct J {
    enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<J> a;
    std::map<std::string, J> o;
    const J* get(const std::string& k) const {
        auto it = o.find(k);
        return it == o.end() ? nullptr : &it->second;
    }
    double num(const std::string& k, double def) const {
        const J* v = get(k);
        return v && v->t == Num ? v->n : def;
    }
    std::string str(const std::string& k, const std::string& def = "") const {
        const J* v = get(k);
        return v && v->t == Str ? v->s : def;
    }
};

struct Parser {
    const char* p;
    const char* end;
    void ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }
    bool lit(const char* w) {
        const size_t n = std::strlen(w);
        if ((size_t) (end - p) >= n && !std::strncmp(p, w, n)) { p += n; return true; }
        return false;
    }
    J value() {
        ws();
        J v;
        if (p >= end) throw std::runtime_error("json: end");
        if (*p == '{') {
            v.t = J::Obj; ++p; ws();
            if (*p == '}') { ++p; return v; }
            for (;;) {
                ws();
                J k = value();
                ws();
                if (*p != ':') throw std::runtime_error("json: ':'");
                ++p;
                v.o[k.s] = value();
                ws();
                if (*p == ',') { ++p; continue; }
                if (*p == '}') { ++p; break; }
                throw std::runtime_error("json: object");
            }
        } else if (*p == '[') {
            v.t = J::Arr; ++p; ws();
            if (*p == ']') { ++p; return v; }
            for (;;) {
                v.a.push_back(value());
                ws();
                if (*p == ',') { ++p; continue; }
                if (*p == ']') { ++p; break; }
                throw std::runtime_error("json: array");
            }
        } else if (*p == '"') {
            v.t = J::Str; ++p;
            while (p < end && *p != '"') {
                if (*p == '\\' && p + 1 < end) {
                    ++p;
                    switch (*p) {
                        case 'n': v.s += '\n'; break;
                        case 't': v.s += '\t'; break;
                        case 'r': v.s += '\r'; break;
                        case 'u': {
                            unsigned cp = std::stoul(std::string(p + 1, p + 5), nullptr, 16);
                            p += 4;
                            if (cp < 0x80) v.s += (char) cp;
                            else if (cp < 0x800) { v.s += (char) (0xc0 | cp >> 6); v.s += (char) (0x80 | (cp & 0x3f)); }
                            else { v.s += (char) (0xe0 | cp >> 12); v.s += (char) (0x80 | (cp >> 6 & 0x3f)); v.s += (char) (0x80 | (cp & 0x3f)); }
                            break;
                        }
                        default: v.s += *p;
                    }
                    ++p;
                } else v.s += *p++;
            }
            ++p;
        } else if (lit("true")) { v.t = J::Bool; v.b = true; }
        else if (lit("false")) { v.t = J::Bool; }
        else if (lit("null")) {}
        else {
            char* e = nullptr;
            v.t = J::Num;
            v.n = std::strtod(p, &e);
            if (e == p) throw std::runtime_error("json: value");
            p = e;
        }
        return v;
    }
};

J parse(const std::string& s) {
    Parser ps{s.data(), s.data() + s.size()};
    return ps.value();
}

std::vector<int32_t> ints(const J* v) {
    std::vector<int32_t> r;
    if (v && v->t == J::Arr)
        for (auto& x : v->a) r.push_back((int32_t) x.n);
    return r;
}

std::mutex out_mu;
void emit(const std::string& line) {
    std::lock_guard<std::mutex> lk(out_mu);
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

std::vector<int32_t> read_ids(const std::string& path) {
    std::ifstream f(path);
    std::vector<int32_t> v;
    long x;
    while (f >> x) v.push_back((int32_t) x);
    return v;
}

std::vector<int> csv_ints(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
    return v;
}

void usage() {
    std::fprintf(stderr,
                 "strata-oss --model FILE.gguf [--gpus 0,1] [--split 12,24] [--ctx N] [--chunk N] [--threads N]\n"
                 "           [--reserve-mib N] [--max-slots N] [--profile FILE] [--adapt-every N] [--adapt-swaps N]\n"
                 "           (--serve | --tokens FILE [--n-predict N] | --dump-logits OUT --tokens FILE)\n");
}

} // namespace

int main(int argc, char** argv) {
    oss::Options o;
    std::string tokens_file, dump, mode = "gen";
    int n_predict = 128;
    bool spec = true;
    int draft_max = 7;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model" || a == "-m") o.model = next();
        else if (a == "--gpus") o.gpus = csv_ints(next());
        else if (a == "--split") o.split = csv_ints(next());
        else if (a == "--ctx") o.ctx = std::atoi(next().c_str());
        else if (a == "--chunk") o.chunk = std::atoi(next().c_str());
        else if (a == "--swa-slack") o.swa_slack = std::atoi(next().c_str());
        else if (a == "--threads") o.threads = std::atoi(next().c_str());
        else if (a == "--reserve-mib") o.reserve_mib = std::atof(next().c_str());
        else if (a == "--max-slots") o.max_slots = std::atoll(next().c_str());
        else if (a == "--profile") { o.profile_in = next(); o.profile_out = o.profile_in; }
        else if (a == "--profile-in") o.profile_in = next();
        else if (a == "--profile-out") o.profile_out = next();
        else if (a == "--adapt-every") o.adapt_every = std::atoi(next().c_str());
        else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(next().c_str());
        else if (a == "--tokens") tokens_file = next();
        else if (a == "--n-predict") n_predict = std::atoi(next().c_str());
        else if (a == "--dump-logits") { dump = next(); mode = "dump"; }
        else if (a == "--serve") mode = "serve";
        else if (a == "--bench-verify") mode = "verify";
        else if (a == "--bench-parallel") mode = "parallel";
        else if (a == "--parallel") o.parallel = std::atoi(next().c_str());
        else if (a == "--no-spec") spec = false;
        else if (a == "--draft-model") o.draft_model = next();
        else if (a == "--draft-vocab") o.draft_vocab = next();
        else if (a == "--draft-window") o.draft_window = std::atoi(next().c_str());
        else if (a == "--draft-max") draft_max = std::atoi(next().c_str());
        else if (a == "-v") o.verbose = true;
        else { usage(); return 2; }
    }
    if (o.model.empty()) { usage(); return 2; }
    try {
        oss::Engine eng(o);
        const int V = eng.n_vocab();
        if (mode == "dump") {
            auto ids = read_ids(tokens_file);
            eng.set_busy(true);
            const float* lg = eng.eval(ids, true);
            FILE* f = std::fopen(dump.c_str(), "wb");
            std::fwrite(lg, sizeof(float), ids.size() * (size_t) V, f);
            std::fclose(f);
            eng.report(stderr);
            return 0;
        }
        if (mode == "parallel") {
            // --tokens a,b,c: generate n-predict greedy tokens for every prompt at once, then each alone; compare
            std::vector<std::vector<int32_t>> prompts;
            {
                std::stringstream ss(tokens_file);
                std::string f;
                while (std::getline(ss, f, ',')) prompts.push_back(read_ids(f));
            }
            const int n = (int) prompts.size();
            if (n > eng.n_seqs()) { std::fprintf(stderr, "--parallel %d < %d prompts\n", eng.n_seqs(), n); return 2; }
            eng.set_busy(true);
            std::vector<std::vector<int32_t>> together(n), alone(n);
            std::vector<int32_t> pend(n);
            for (int i = 0; i < n; ++i) {
                eng.select(i);
                const float* lg = eng.eval(prompts[i]);
                pend[i] = (int32_t) (std::max_element(lg, lg + V) - lg);
            }
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<int> slots(n);
            for (int i = 0; i < n; ++i) slots[i] = i;
            for (int k = 0; k < n_predict; ++k) {
                for (int i = 0; i < n; ++i) together[i].push_back(pend[i]);
                const float* lg = eng.step_multi(slots, pend);
                for (int i = 0; i < n; ++i)
                    pend[i] = (int32_t) (std::max_element(lg + (size_t) i * V, lg + (size_t) (i + 1) * V) - (lg + (size_t) i * V));
            }
            const double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "strata-oss: %d conversations x %d tokens together: %.2f s, %.1f tok/s in total (%.1f each)\n",
                         n, n_predict, tt, n * n_predict / tt, n_predict / tt);
            double ta = 0;
            for (int i = 0; i < n; ++i) {
                eng.select(i);
                eng.reuse_prefix({});  // read the prompt again the same way (the prompt path rounds differently)
                const float* lg = eng.eval(prompts[i]);
                const auto t1 = std::chrono::steady_clock::now();
                int32_t t = (int32_t) (std::max_element(lg, lg + V) - lg);
                for (int k = 0; k < n_predict; ++k) {
                    alone[i].push_back(t);
                    lg = eng.eval({t});
                    t = (int32_t) (std::max_element(lg, lg + V) - lg);
                }
                ta += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            }
            std::fprintf(stderr, "strata-oss: the same one at a time: %.2f s, %.1f tok/s\n", ta, n * n_predict / ta);
            for (int i = 0; i < n; ++i) {
                int same = 0;
                while (same < n_predict && together[i][same] == alone[i][same]) ++same;
                std::fprintf(stderr, "strata-oss: conversation %d: %s (first %d of %d tokens equal)\n", i,
                             same == n_predict ? "same tokens" : "DIFFERENT", same, n_predict);
            }
            eng.report(stderr);
            return 0;
        }
        if (mode == "verify") {
            // the cost of a decode step that checks nc tokens at once (a speculative verify window)
            auto ids = read_ids(tokens_file);
            eng.set_busy(true);
            eng.eval(ids);
            std::vector<int32_t> probe(ids.begin(), ids.end());
            probe.push_back(-1);
            const std::vector<int32_t> words = {13, 290, 326, 316, 382, 261, 11, 328};
            for (int nc : {1, 2, 3, 4, 6, 8}) {
                double best = 1e9, sum = 0;
                const int reps = 24;
                for (int r = 0; r < reps; ++r) {
                    eng.reuse_prefix(probe);
                    std::vector<int32_t> w(words.begin(), words.begin() + nc);
                    for (auto& x : w) x += r;  // vary the tokens a little
                    const auto t0 = std::chrono::steady_clock::now();
                    eng.eval(w);
                    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (r >= 4) { best = std::min(best, dt); sum += dt; }
                }
                std::fprintf(stderr, "verify window %d tokens: %.2f ms a step (best %.2f), %.2f ms a token\n", nc,
                             1e3 * sum / (reps - 4), 1e3 * best, 1e3 * sum / (reps - 4) / nc);
            }
            eng.report(stderr);
            return 0;
        }
        if (mode == "gen") {
            auto ids = read_ids(tokens_file);
            if (ids.empty()) { std::fprintf(stderr, "no tokens\n"); return 2; }
            eng.set_busy(true);
            const auto t0 = std::chrono::steady_clock::now();
            const float* lg = eng.eval(ids);
            const auto t1 = std::chrono::steady_clock::now();
            std::vector<int32_t> out;
            Sampler smp;
            oss::Generator gen(eng, smp);
            gen.spec = spec;
            gen.max_draft = draft_max;
            gen.begin(lg);
            const std::vector<int32_t> stop = {eng.eos(), 200012, 199999};
            while ((int) out.size() < n_predict) {
                auto toks = gen.next(stop);
                bool end = false;
                for (int t : toks) {
                    out.push_back(t);
                    std::printf("%d ", t);
                    if (std::find(stop.begin(), stop.end(), t) != stop.end()) end = true;
                }
                std::fflush(stdout);
                if (end) break;
            }
            std::fprintf(stderr, "strata-oss: %llu steps, %llu drafted, %llu kept (%.2f tokens a step; EAGLE3 %llu / %llu)\n",
                         (unsigned long long) gen.st.steps, (unsigned long long) gen.st.drafted,
                         (unsigned long long) gen.st.accepted,
                         gen.st.steps ? (double) gen.st.emitted / gen.st.steps : 0.0,
                         (unsigned long long) gen.st.eagle_kept, (unsigned long long) gen.st.eagle_drafted);
            const auto t2 = std::chrono::steady_clock::now();
            std::printf("\n");
            const double tp = std::chrono::duration<double>(t1 - t0).count();
            const double tg = std::chrono::duration<double>(t2 - t1).count();
            std::fprintf(stderr, "strata-oss: prompt %zu tokens %.2f s (%.1f tok/s), %zu generated %.2f s (%.2f tok/s)\n",
                         ids.size(), tp, ids.size() / tp, out.size(), tg, out.size() / tg);
            eng.report(stderr);
            return 0;
        }
        // ---- serve: up to --parallel answers at once.  A reader thread queues the requests; the loop reads one new
        // prompt at a time, then advances every answer: alone with speculation, together in one batched step.
        std::deque<std::string> q;
        std::mutex mu;
        std::condition_variable cv;
        std::atomic<bool> eof{false};
        std::thread reader([&] {
            std::string line;
            while (std::getline(std::cin, line)) {
                std::lock_guard<std::mutex> lk(mu);
                q.push_back(line);
                cv.notify_one();
            }
            eof.store(true);
            cv.notify_one();
        });
        reader.detach();
        {
            std::ostringstream ss;
            ss << "{\"type\":\"ready\",\"n_vocab\":" << V << ",\"n_ctx\":" << eng.n_ctx()
               << ",\"parallel\":" << eng.n_seqs() << ",\"expert_slots\":" << eng.expert_slots()
               << ",\"expert_mib\":" << eng.expert_mib() << ",\"draft\":" << (eng.has_draft() ? "true" : "false")
               << ",\"spec\":" << (spec ? "true" : "false") << "}";
            emit(ss.str());
        }
        struct Req {
            std::string rid;
            std::vector<int32_t> toks, stops;
            Sampler smp;
            std::unique_ptr<oss::Generator> gen;
            int slot = -1, max_tokens = 512, n_gen = 0, reused = 0;
            bool started = false, cancel = false, spec = true;
            std::chrono::steady_clock::time_point t0;
            double t_prompt = 0;
            uint64_t routed0 = 0, hits0 = 0;  // the engine's routing counters when the answer started
        };
        std::vector<std::unique_ptr<Req>> active, waiting;
        std::vector<char> slot_busy((size_t) eng.n_seqs(), 0);
        auto jstr = [](const std::string& x) {
            std::string r = "\"";
            for (char c : x) { if (c == '"' || c == '\\') r += '\\'; r += c; }
            return r + "\"";
        };
        auto finish = [&](Req& r, const std::string& reason) {
            const double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - r.t0).count();
            std::ostringstream ss;
            ss << "{\"type\":\"done\",\"rid\":" << jstr(r.rid) << ",\"reason\":\"" << reason << "\",\"n_gen\":"
               << r.n_gen << ",\"n_prompt\":" << r.toks.size() << ",\"n_reused\":" << r.reused
               << ",\"t_prompt\":" << r.t_prompt << ",\"t_gen\":" << (tt - r.t_prompt)
               << ",\"steps\":" << (r.gen ? r.gen->st.steps : 0)
               << ",\"drafted\":" << (r.gen ? r.gen->st.drafted : 0) << ",\"kept\":" << (r.gen ? r.gen->st.accepted : 0)
               << ",\"routed\":" << (eng.stats().routed - r.routed0) << ",\"hits\":" << (eng.stats().gpu_hits - r.hits0)
               << "}";
            emit(ss.str());
            if (r.slot >= 0) slot_busy[(size_t) r.slot] = 0;
            r.slot = -1;
        };
        // emits t; true when the answer ends with it
        auto give = [&](Req& r, int32_t t, std::string& reason) {
            ++r.n_gen;
            std::ostringstream ss;
            ss << "{\"type\":\"token\",\"rid\":" << jstr(r.rid) << ",\"id\":" << t << "}";
            emit(ss.str());
            if (std::find(r.stops.begin(), r.stops.end(), t) != r.stops.end()) { reason = "stop"; return true; }
            if (r.n_gen >= r.max_tokens) { reason = "length"; return true; }
            return false;
        };
        for (;;) {
            // 1. take requests
            {
                std::unique_lock<std::mutex> lk(mu);
                if (active.empty() && waiting.empty())
                    cv.wait(lk, [&] { return !q.empty() || eof.load(); });
                if (q.empty() && eof.load() && active.empty() && waiting.empty()) break;
                while (!q.empty()) {
                    const std::string line = q.front();
                    q.pop_front();
                    lk.unlock();
                    J req;
                    bool ok = true;
                    try { req = parse(line); } catch (std::exception&) { ok = false; }
                    if (!ok) { emit("{\"type\":\"error\",\"error\":\"bad json\"}"); lk.lock(); continue; }
                    const std::string op = req.str("op", "generate");
                    const std::string rid = req.str("rid", "0");
                    if (op == "cancel") {
                        for (auto& r : active) if (r->rid == rid) r->cancel = true;
                        for (auto& r : waiting) if (r->rid == rid) r->cancel = true;
                    } else if (op == "stats") {
                        eng.report(stderr);
                        emit("{\"type\":\"stats\"}");
                    } else if (op == "save_profile") {
                        eng.save_profile();
                        emit("{\"type\":\"saved\"}");
                    } else {
                        auto r = std::make_unique<Req>();
                        r->rid = rid;
                        r->toks = ints(req.get("tokens"));
                        r->stops = ints(req.get("stop"));
                        r->max_tokens = (int) req.num("max_tokens", 512);
                        r->smp.temp = req.num("temperature", 0);
                        r->smp.top_p = req.num("top_p", 1);
                        r->smp.top_k = (int) req.num("top_k", 0);
                        r->smp.min_p = req.num("min_p", 0);
                        r->smp.rng.seed((uint64_t) req.num("seed", (double) std::random_device{}()));
                        r->spec = spec && !req.get("no_spec");
                        r->t0 = std::chrono::steady_clock::now();
                        r->routed0 = eng.stats().routed;
                        r->hits0 = eng.stats().gpu_hits;
                        if (r->toks.empty() || (int) r->toks.size() >= eng.n_ctx()) {
                            emit("{\"type\":\"error\",\"rid\":" + jstr(rid) + ",\"error\":\"" +
                                 (r->toks.empty() ? "no tokens" : "prompt longer than the context") + "\"}");
                        } else {
                            waiting.push_back(std::move(r));
                        }
                    }
                    lk.lock();
                }
            }
            eng.set_busy(true);
            // 2. cancelled answers
            for (auto& r : waiting) if (r->cancel) finish(*r, "cancel");
            waiting.erase(std::remove_if(waiting.begin(), waiting.end(), [](auto& r) { return r->cancel; }), waiting.end());
            for (auto& r : active) if (r->cancel && r->slot >= 0) finish(*r, "cancel");
            active.erase(std::remove_if(active.begin(), active.end(), [](auto& r) { return r->slot < 0; }), active.end());
            // 3. one new prompt: the free slot whose conversation shares the longest prefix with it
            if (!waiting.empty()) {
                int best = -1;
                size_t best_n = 0;
                for (int sl = 0; sl < eng.n_seqs(); ++sl) {
                    if (slot_busy[(size_t) sl]) continue;
                    eng.select(sl);
                    const auto& h = eng.history();
                    size_t n = 0;
                    while (n < h.size() && n < waiting.front()->toks.size() && h[n] == waiting.front()->toks[n]) ++n;
                    if (best < 0 || n > best_n) { best = sl; best_n = n; }
                }
                if (best >= 0) {
                    auto r = std::move(waiting.front());
                    waiting.erase(waiting.begin());
                    r->slot = best;
                    slot_busy[(size_t) best] = 1;
                    eng.select(best);
                    try {
                        r->reused = eng.reuse_prefix(r->toks);
                        std::vector<int32_t> rest(r->toks.begin() + r->reused, r->toks.end());
                        {
                            std::ostringstream ss;
                            ss << "{\"type\":\"start\",\"rid\":" << jstr(r->rid) << ",\"n_prompt\":" << r->toks.size()
                               << ",\"n_reused\":" << r->reused << ",\"slot\":" << best << "}";
                            emit(ss.str());
                        }
                        const float* lg = eng.eval(rest);
                        r->t_prompt = std::chrono::duration<double>(std::chrono::steady_clock::now() - r->t0).count();
                        r->gen = std::make_unique<oss::Generator>(eng, r->smp);
                        r->gen->spec = r->spec;
                        r->gen->max_draft = draft_max;
                        r->gen->begin(lg);
                        active.push_back(std::move(r));
                    } catch (std::exception& e) {
                        emit("{\"type\":\"error\",\"rid\":" + jstr(r->rid) + ",\"error\":" + jstr(e.what()) + "}");
                        eng.reuse_prefix({});
                        slot_busy[(size_t) best] = 0;
                    }
                }
            }
            // 4. advance the answers
            try {
                if (active.size() == 1) {
                    Req& r = *active[0];
                    eng.select(r.slot);
                    std::string reason;
                    bool end = false;
                    if (eng.n_past() + 2 >= eng.n_ctx()) { reason = "context"; end = true; }
                    else
                        for (int t : r.gen->next(r.stops))
                            if (give(r, t, reason)) { end = true; break; }
                    if (end) finish(r, reason);
                } else if (active.size() > 1) {
                    std::vector<int> slots;
                    std::vector<int32_t> toks;
                    std::vector<Req*> who;
                    for (auto& rp : active) {
                        Req& r = *rp;
                        std::string reason;
                        eng.select(r.slot);
                        if (give(r, r.gen->pending(), reason)) { finish(r, reason); continue; }
                        if (eng.n_past() + 2 >= eng.n_ctx()) { finish(r, "context"); continue; }
                        if ((int) slots.size() < 8) {  // the engine's columns a step
                            slots.push_back(r.slot);
                            toks.push_back(r.gen->pending());
                            who.push_back(&r);
                        }
                    }
                    if (!slots.empty()) {
                        const float* lg = eng.step_multi(slots, toks);
                        for (size_t c = 0; c < who.size(); ++c) who[c]->gen->advance(lg + c * (size_t) V);
                    }
                }
            } catch (std::exception& e) {
                for (auto& r : active) {
                    emit("{\"type\":\"error\",\"rid\":" + jstr(r->rid) + ",\"error\":" + jstr(e.what()) + "}");
                    if (r->slot >= 0) { eng.select(r->slot); eng.reuse_prefix({}); slot_busy[(size_t) r->slot] = 0; }
                    r->slot = -1;
                }
            }
            active.erase(std::remove_if(active.begin(), active.end(), [](auto& r) { return r->slot < 0; }), active.end());
            if (active.empty() && waiting.empty()) {
                eng.set_busy(false);
                eng.report(stderr);
            }
        }
        return 0;
    } catch (std::exception& e) {
        std::fprintf(stderr, "strata-oss: %s\n", e.what());
        return 1;
    }
}
