"""The EAGLE3 draft's reduced vocabulary: the N most frequent tokens of a corpus, every special token (ids >= 199998),
then the lowest ids not yet in, up to N.  The draft only proposes these; its head is N rows instead of 201,088.

    python oss/tools/draft_vocab.py MODEL.gguf OUT.txt [N] [files or globs...]

Measured (oss/FINDINGS.md section 15): the repo's docs + code and llama.cpp's (4.0M tokens), N = 49,152 covers 96.4%
of gpt-oss-120b's tokens in four held-out answers (prose, a code edit, a summary).
"""
import collections
import glob
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import oss_tokenizer  # noqa: E402

DEFAULT = ["docs/**/*.md", "*.md", "serve/*.py", "tools/*.py", "src/**/*.cpp", "src/**/*.cu", "include/**/*.hpp"]


def main():
    model, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 49152
    pats = sys.argv[4:] or DEFAULT
    tk = oss_tokenizer.load(model)
    cnt = collections.Counter()
    for pat in pats:
        for f in glob.glob(pat, recursive=True):
            try:
                cnt.update(tk.encode(open(f, errors="ignore").read()[:400000]))
            except OSError:
                pass
    keep = [i for i, _ in cnt.most_common(n)]
    s = set(keep) | set(range(199998, 201088))
    for i in range(201088):
        if len(s) >= n:
            break
        s.add(i)
    pathlib.Path(out).write_text(" ".join(map(str, sorted(s))) + "\n")
    print(f"{len(s)} ids from {sum(cnt.values())} corpus tokens ({len(cnt)} distinct) -> {out}")


if __name__ == "__main__":
    main()
