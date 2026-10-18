// oss/oss_main.cpp - strata-oss: the GPT-OSS engine's command line.
//
//   strata-oss --model gpt-oss-120b-MXFP4.gguf [--gpus 0,1] [--ctx 32768] ...
//     --tokens FILE [--n-predict N]     greedy generation from token ids (whitespace separated), prints ids + speed
//     --dump-logits OUT --tokens FILE   teacher-forced logits of every position (float32 [n][vocab]) to OUT
//     --serve                           JSON lines on stdin/stdout (oss/server.py speaks it)
#include "oss_engine.hpp"

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

// ---------------------------------------------------------------- sampling
struct Sampler {
    double temp = 0, top_p = 1, min_p = 0;
    int top_k = 0;
    std::mt19937_64 rng{1};
    std::vector<std::pair<float, int>> cand;
    int pick(const float* lg, int V) {
        if (temp <= 0) return (int) (std::max_element(lg, lg + V) - lg);
        cand.clear();
        const float mx = *std::max_element(lg, lg + V);
        // keep what can matter: logits within 30 / temp of the best
        const float cut = mx - 30.f * (float) temp;
        for (int i = 0; i < V; ++i)
            if (lg[i] >= cut) cand.push_back({lg[i], i});
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        if (top_k > 0 && (int) cand.size() > top_k) cand.resize(top_k);
        std::vector<double> p(cand.size());
        double s = 0;
        for (size_t i = 0; i < cand.size(); ++i) { p[i] = std::exp((cand[i].first - mx) / temp); s += p[i]; }
        for (auto& x : p) x /= s;
        size_t n = p.size();
        if (top_p < 1) {
            double c = 0;
            for (size_t i = 0; i < p.size(); ++i) { c += p[i]; if (c >= top_p) { n = i + 1; break; } }
        }
        if (min_p > 0)
            for (size_t i = 1; i < n; ++i) if (p[i] < min_p * p[0]) { n = i; break; }
        double tot = 0;
        for (size_t i = 0; i < n; ++i) tot += p[i];
        std::uniform_real_distribution<double> u(0, tot);
        double r = u(rng), c = 0;
        for (size_t i = 0; i < n; ++i) { c += p[i]; if (r <= c) return cand[i].second; }
        return cand[n - 1].second;
    }
};

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
        if (mode == "gen") {
            auto ids = read_ids(tokens_file);
            if (ids.empty()) { std::fprintf(stderr, "no tokens\n"); return 2; }
            eng.set_busy(true);
            const auto t0 = std::chrono::steady_clock::now();
            const float* lg = eng.eval(ids);
            const auto t1 = std::chrono::steady_clock::now();
            std::vector<int32_t> out;
            Sampler smp;
            for (int i = 0; i < n_predict; ++i) {
                const int t = smp.pick(lg, V);
                out.push_back(t);
                std::printf("%d ", t);
                std::fflush(stdout);
                if (t == eng.eos() || t == 200012 || t == 199999) break;
                lg = eng.eval({t});
            }
            const auto t2 = std::chrono::steady_clock::now();
            std::printf("\n");
            const double tp = std::chrono::duration<double>(t1 - t0).count();
            const double tg = std::chrono::duration<double>(t2 - t1).count();
            std::fprintf(stderr, "strata-oss: prompt %zu tokens %.2f s (%.1f tok/s), %zu generated %.2f s (%.2f tok/s)\n",
                         ids.size(), tp, ids.size() / tp, out.size(), tg, out.size() / tg);
            eng.report(stderr);
            return 0;
        }
        // ---- serve: one request at a time; a reader thread so a "cancel" can arrive mid-answer
        std::deque<std::string> q;
        std::mutex mu;
        std::condition_variable cv;
        std::atomic<bool> cancel{false}, eof{false};
        std::thread reader([&] {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.find("\"cancel\"") != std::string::npos) { cancel.store(true); continue; }
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
            ss << "{\"type\":\"ready\",\"n_vocab\":" << V << ",\"n_ctx\":" << eng.n_ctx() << "}";
            emit(ss.str());
        }
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return !q.empty() || eof.load(); });
                if (q.empty()) break;
                line = q.front();
                q.pop_front();
            }
            J req;
            try { req = parse(line); } catch (std::exception& e) {
                emit(std::string("{\"type\":\"error\",\"error\":\"bad json\"}"));
                continue;
            }
            const std::string op = req.str("op", "generate");
            if (op == "stats") {
                eng.report(stderr);
                emit("{\"type\":\"stats\"}");
                continue;
            }
            if (op == "save_profile") { eng.save_profile(); emit("{\"type\":\"saved\"}"); continue; }
            auto toks = ints(req.get("tokens"));
            if (toks.empty()) { emit("{\"type\":\"error\",\"error\":\"no tokens\"}"); continue; }
            if ((int) toks.size() >= eng.n_ctx()) {
                emit("{\"type\":\"error\",\"error\":\"prompt longer than the context\"}");
                continue;
            }
            cancel.store(false);
            Sampler smp;
            smp.temp = req.num("temperature", 0);
            smp.top_p = req.num("top_p", 1);
            smp.top_k = (int) req.num("top_k", 0);
            smp.min_p = req.num("min_p", 0);
            smp.rng.seed((uint64_t) req.num("seed", (double) std::random_device{}()));
            const int max_tokens = (int) req.num("max_tokens", 512);
            auto stops = ints(req.get("stop"));
            eng.set_busy(true);
            const auto t0 = std::chrono::steady_clock::now();
            const int reused = eng.reuse_prefix(toks);
            std::vector<int32_t> rest(toks.begin() + reused, toks.end());
            {
                std::ostringstream ss;
                ss << "{\"type\":\"start\",\"n_prompt\":" << toks.size() << ",\"n_reused\":" << reused << "}";
                emit(ss.str());
            }
            const float* lg = nullptr;
            std::string reason = "length";
            int n_gen = 0;
            double t_prompt = 0;
            try {
                lg = eng.eval(rest);
                const auto t1 = std::chrono::steady_clock::now();
                t_prompt = std::chrono::duration<double>(t1 - t0).count();
                for (; n_gen < max_tokens;) {
                    if (cancel.load()) { reason = "cancel"; break; }
                    const int t = smp.pick(lg, V);
                    ++n_gen;
                    {
                        std::ostringstream ss;
                        ss << "{\"type\":\"token\",\"id\":" << t << "}";
                        emit(ss.str());
                    }
                    if (std::find(stops.begin(), stops.end(), t) != stops.end()) { reason = "stop"; break; }
                    if (eng.n_past() + 1 >= eng.n_ctx()) { reason = "context"; break; }
                    lg = eng.eval({t});
                }
            } catch (std::exception& e) {
                std::ostringstream ss;
                ss << "{\"type\":\"error\",\"error\":\"" << e.what() << "\"}";
                emit(ss.str());
                eng.reuse_prefix({});
                eng.set_busy(false);
                continue;
            }
            const auto t2 = std::chrono::steady_clock::now();
            const double tt = std::chrono::duration<double>(t2 - t0).count();
            eng.set_busy(false);
            std::ostringstream ss;
            ss << "{\"type\":\"done\",\"reason\":\"" << reason << "\",\"n_gen\":" << n_gen
               << ",\"n_prompt\":" << toks.size() << ",\"n_reused\":" << reused << ",\"t_prompt\":" << t_prompt
               << ",\"t_gen\":" << (tt - t_prompt) << "}";
            emit(ss.str());
            eng.report(stderr);
        }
        return 0;
    } catch (std::exception& e) {
        std::fprintf(stderr, "strata-oss: %s\n", e.what());
        return 1;
    }
}
