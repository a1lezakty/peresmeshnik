"""Speaker-verification threshold bench (stream S3, stage 8).

Run where the data is ($SPK_BENCH: data/LibriSpeech dev-clean, data/dev + dev.tsv of FLEURS ru, model/ the ONNX):
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python bench/speaker_threshold.py <stage>` (onnx-asr 0.12.0, soundfile).
Stages: embed (LibriSpeech + FLEURS-ru embeddings -> work/*.npz), report (tables -> stdout + work/report.json),
timing (1 thread, 5 s and 40 s), padded + report_padded (8 s live with silence, raw vs trimmed).
The model is run exactly as onnx-asr 0.12.0 does it: Manager().create_se()
-> KaldiPreprocessorNumpy('wespeaker') -> mean subtraction over time -> ONNX 'feats' -> 'embs'.
"""

import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt
import soundfile as sf
from onnx_asr.loader import Manager

ROOT = Path(os.environ.get("SPK_BENCH", "speaker-bench"))  # data/, model/, work/
MODEL_DIR = ROOT / "model"
LIBRI = ROOT / "data/LibriSpeech"
FLEURS = ROOT / "data/dev"
FLEURS_TSV = ROOT / "data/dev.tsv"
WORK = ROOT / "work"
SR = 16000
FILE_SEC = 40
LIVE_SECS = (3, 5, 8)
N_LIVE = 4  # live segments per chapter and duration
FILE_CONDS = ("clean", "opus24k", "tel8k")
LIVE_CONDS = ("clean", "laptop_white20", "laptop_white10", "laptop_pink20", "laptop_pink10")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
CATS = ("own_other_session", "own_same_session", "other_same_gender", "other_diff_gender")


def make_se(threads: int):
    so = rt.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    cfg = {"sess_options": so, "providers": ["CPUExecutionProvider"]}
    return Manager(sess_options=so, providers=["CPUExecutionProvider"]).create_se(
        local_dir=MODEL_DIR, offline=True, config=cfg
    )


def embed(se, x: np.ndarray) -> np.ndarray:
    e = se.embedding(np.ascontiguousarray(x, dtype=np.float32), sample_rate=SR)
    return e / np.linalg.norm(e)


# ---------- degradations ----------
def ffmpeg_roundtrip(x: np.ndarray, enc_args: list[str]) -> np.ndarray:
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1"]
    raw_in = ["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:"]
    enc = subprocess.run(base + raw_in + enc_args + ["pipe:"], input=x.astype(np.float32).tobytes(),
                         capture_output=True, check=True).stdout
    dec = subprocess.run(base + ["-i", "pipe:", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:"],
                         input=enc, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, dtype=np.float32)
    y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    return y.copy()


def file_cond(x: np.ndarray, cond: str) -> np.ndarray:
    if cond == "clean":
        return x
    if cond == "opus24k":
        return ffmpeg_roundtrip(x, ["-c:a", "libopus", "-b:a", "24k", "-f", "ogg"])
    if cond == "tel8k":  # imported file that was 8 kHz telephone audio, upsampled back by the app
        return ffmpeg_roundtrip(x, ["-ar", "8000", "-c:a", "pcm_s16le", "-f", "wav"])
    raise ValueError(cond)


def bandpass(x: np.ndarray, lo=100.0, hi=7000.0) -> np.ndarray:
    # ponytail: FFT brick-wall filter, a real mic has smooth roll-off
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x)).astype(np.float32)


def noise(n: int, color: str, rng) -> np.ndarray:
    w = rng.standard_normal(n)
    if color == "white":
        return w
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / SR)
    W[1:] /= np.sqrt(f[1:])
    W[0] = 0
    return np.fft.irfft(W, n)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def live_cond(x: np.ndarray, cond: str, rng) -> np.ndarray:
    if cond == "clean":
        return x
    color, snr = cond.split("_")[1][:-2], int(cond[-2:])  # laptop_<white|pink><snr>

    s = bandpass(x)
    s *= 10 ** (-30 / 20) / rms(s)  # speech level -30 dBFS RMS
    nz = bandpass(noise(len(x), color, rng).astype(np.float32))
    nz *= rms(s) / 10 ** (snr / 20) / rms(nz)
    return np.clip(s + nz, -1, 1).astype(np.float32)


# ---------- data ----------
def libri_speakers():
    gender = {}
    for line in open(LIBRI / "SPEAKERS.TXT"):
        if line.startswith(";"):
            continue
        p = [c.strip() for c in line.split("|")]
        if len(p) >= 3 and p[2] == "dev-clean":
            gender[p[0]] = p[1]
    return gender


def concat_from(utts: list[Path], start: int, sec: float):
    need, buf, i = int(sec * SR), [], start
    while sum(map(len, buf)) < need and i < len(utts):
        buf.append(sf.read(utts[i], dtype="float32")[0])
        i += 1
    x = np.concatenate(buf) if buf else np.zeros(0, np.float32)
    return (x[:need], i) if len(x) >= need else (None, i)


def stage_embed():
    WORK.mkdir(exist_ok=True)
    se = make_se(2)
    rng = np.random.default_rng(0)
    gender = libri_speakers()
    files, lives = [], []  # (meta, emb)
    t0 = time.time()
    for spk in sorted(os.listdir(LIBRI / "dev-clean"), key=int):
        for ch in sorted(os.listdir(LIBRI / "dev-clean" / spk)):
            utts = sorted(glob.glob(str(LIBRI / "dev-clean" / spk / ch / "*.flac")))
            fx, used = concat_from(utts, 0, FILE_SEC)
            if fx is None:
                print("skip chapter (short)", spk, ch, flush=True)
                continue
            for fc in FILE_CONDS:
                files.append(({"spk": spk, "ch": ch, "g": gender[spk], "cond": fc}, embed(se, file_cond(fx, fc))))
            rest = list(range(used, len(utts)))
            starts = [rest[int(k * len(rest) / N_LIVE)] for k in range(N_LIVE)] if len(rest) >= N_LIVE else rest
            for k, st in enumerate(starts):
                for d in LIVE_SECS:
                    lx, _ = concat_from(utts, st, d)
                    if lx is None:
                        continue
                    for lc in LIVE_CONDS:
                        meta = {"spk": spk, "ch": ch, "g": gender[spk], "cond": lc, "dur": d, "k": k}
                        lives.append((meta, embed(se, live_cond(lx, lc, rng))))
            print(f"{spk}/{ch} files={len(files)} lives={len(lives)} t={time.time() - t0:.0f}s", flush=True)
    save(WORK / "emb_libri.npz", files, lives)

    # FLEURS ru dev: speakers unknown (one reads several sentences, even the same twice) -> only diff_gender is a
    # clean impostor set; file = a whole utterance, live = the first 5 s of another
    fl_files, fl_lives = [], []
    for row in csv.reader(open(FLEURS_TSV), delimiter="\t", quoting=csv.QUOTE_NONE):
        sid, name, g = row[0], row[1], row[-1][0]
        x, sr = sf.read(FLEURS / name, dtype="float32")
        assert sr == SR
        fl_files.append(({"sent": sid, "file": name, "g": g, "cond": "clean", "sec": len(x) / SR}, embed(se, x)))
        if len(x) >= 5 * SR:
            for lc in LIVE_CONDS:
                meta = {"sent": sid, "file": name, "g": g, "cond": lc, "dur": 5}
                fl_lives.append((meta, embed(se, live_cond(x[: 5 * SR], lc, rng))))
    print(f"fleurs files={len(fl_files)} lives={len(fl_lives)} t={time.time() - t0:.0f}s", flush=True)
    save(WORK / "emb_fleurs.npz", fl_files, fl_lives)


def padded_live(utts, start, rng):
    """Real live reading: 8 s = 1.5 s room noise + phrase1 + 0.5 s pause + phrase2 + 1.5 s room noise.

    Speech = 4.5 s of consecutive LibriSpeech speech split in two halves; room noise = pink, -60 dBFS RMS.
    """
    sp, _ = concat_from(utts, start, 4.5)
    if sp is None:
        return None
    room = noise(8 * SR, "pink", rng).astype(np.float32)
    room *= 10 ** (-60 / 20) / rms(room)
    h, a = len(sp) // 2, int(1.5 * SR)
    b = a + h + int(0.5 * SR)
    room[a: a + h] += sp[:h]
    room[b: b + len(sp) - h] += sp[h:]
    return np.clip(room, -1, 1)


def trim_silence(x, frame=480):
    """Keep only 30 ms frames with energy > max(p10_dB + 6, -60) dBFS, concatenated."""
    fr = x[: len(x) // frame * frame].reshape(-1, frame)
    e = 10 * np.log10(np.mean(np.square(fr, dtype=np.float64), axis=1) + 1e-20)
    return fr[e > max(np.percentile(e, 10) + 6, -60)].reshape(-1)


def stage_padded():
    se = make_se(2)
    rng = np.random.default_rng(1)
    gender = libri_speakers()
    lives, kept = [], []
    t0 = time.time()
    for spk in sorted(os.listdir(LIBRI / "dev-clean"), key=int):
        for ch in sorted(os.listdir(LIBRI / "dev-clean" / spk)):
            utts = sorted(glob.glob(str(LIBRI / "dev-clean" / spk / ch / "*.flac")))
            fx, used = concat_from(utts, 0, FILE_SEC)
            if fx is None:
                continue
            rest = list(range(used, len(utts)))
            starts = [rest[int(k * len(rest) / N_LIVE)] for k in range(N_LIVE)] if len(rest) >= N_LIVE else rest
            for k, st in enumerate(starts):
                x = padded_live(utts, st, rng)
                if x is None:
                    continue
                tr = trim_silence(x)
                kept.append(len(tr) / SR)
                for cond, y in (("pad8_raw", x), ("pad8_trim", tr)):
                    lives.append(({"spk": spk, "ch": ch, "g": gender[spk], "cond": cond, "dur": 8, "k": k},
                                  embed(se, y)))
        print(f"{spk} lives={len(lives)} t={time.time() - t0:.0f}s", flush=True)
    print(f"trimmed length s: p5={np.percentile(kept, 5):.2f} p50={np.median(kept):.2f} "
          f"p95={np.percentile(kept, 95):.2f} (speech 4.5 s)", flush=True)
    np.savez(WORK / "emb_padded.npz", le=np.stack([e for _, e in lives]), lm=json.dumps([m for m, _ in lives]))


def stage_report_padded():
    fe, fm, _, _ = load(WORK / "emb_libri.npz")
    z = np.load(WORK / "emb_padded.npz")
    le, lm = z["le"], json.loads(str(z["lm"]))
    out = {}
    def P(d):  # noqa: N802
        return " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())
    for fc in FILE_CONDS:
        for lc in ("pad8_raw", "pad8_trim"):
            sc = libri_scores(fe, fm, le, lm, fc, lc, 8)
            gen, imp = sc["own_other_session"], sc["other_same_gender"]
            ts = np.round(np.arange(0.20, 0.801, 0.01), 2)
            sweep = {float(t): rates(gen, imp, t) for t in ts}
            r = {"gen": pct(gen), "imp": pct(imp), "eer": eer(gen, imp)[0],
                 "far1_from": min((t for t, (_, far) in sweep.items() if far <= 0.01), default=9),
                 "frr5_upto": max((t for t, (frr, _) in sweep.items() if frr <= 0.05), default=-9),
                 "at": {t: rates(gen, imp, t) for t in THRESHOLDS}}
            out[f"{fc}|{lc}"] = r
            print(f"## file={fc} live={lc}: EER={r['eer']:.3%} window {r['far1_from']:.2f}..{r['frr5_upto']:.2f}")
            print(f"  own_other_session {P(r['gen'])}\n  other_same_gender {P(r['imp'])}")
            print("  " + "  ".join(f"t={t}: FRR={a:.3f} FAR={b:.4f}" for t, (a, b) in r["at"].items()))
    json.dump(out, open(WORK / "report_padded.json", "w"), indent=1, default=float)


def save(path, files, lives):
    np.savez(path, fe=np.stack([e for _, e in files]), le=np.stack([e for _, e in lives]),
             fm=json.dumps([m for m, _ in files]), lm=json.dumps([m for m, _ in lives]))


def load(path):
    z = np.load(path)
    return z["fe"], json.loads(str(z["fm"])), z["le"], json.loads(str(z["lm"]))


# ---------- report ----------
def pct(a):
    a = np.asarray(a)
    if not len(a):
        return {"n": 0}
    q = np.percentile(a, [1, 5, 50, 95, 99])
    return {"n": int(len(a)), "p1": q[0], "p5": q[1], "p50": q[2], "p95": q[3], "p99": q[4], "max": float(a.max())}


def eer(gen, imp):
    gen, imp = np.sort(gen), np.sort(imp)
    ts = np.linspace(-0.2, 1.0, 1201)
    frr = np.searchsorted(gen, ts, side="left") / len(gen)
    far = 1 - np.searchsorted(imp, ts, side="left") / len(imp)
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2), float(ts[i])


def rates(gen, imp, t):
    return float(np.mean(np.asarray(gen) < t)), float(np.mean(np.asarray(imp) >= t))


def libri_scores(fe, fm, le, lm, fc, lc, dur, two=False):
    fi = [i for i, m in enumerate(fm) if m["cond"] == fc]
    li = [i for i, m in enumerate(lm) if m["cond"] == lc and m["dur"] == dur]
    F, L = fe[fi], le[li]
    fmeta, lmeta = [fm[i] for i in fi], [lm[i] for i in li]
    if two:  # two phrases: average of normalized embeddings of live segments (k, k+1) of the same chapter
        idx = {(m["spk"], m["ch"], m["k"]): j for j, m in enumerate(lmeta)}
        pairs = [(j, idx[(m["spk"], m["ch"], m["k"] + 1)]) for j, m in enumerate(lmeta)
                 if m["k"] % 2 == 0 and (m["spk"], m["ch"], m["k"] + 1) in idx]
        L = np.stack([L[a] + L[b] for a, b in pairs])
        L /= np.linalg.norm(L, axis=1, keepdims=True)
        lmeta = [lmeta[a] for a, _ in pairs]
    S = F @ L.T
    fs, ls = np.array([m["spk"] for m in fmeta]), np.array([m["spk"] for m in lmeta])
    fcch, lcch = np.array([m["ch"] for m in fmeta]), np.array([m["ch"] for m in lmeta])
    fg, lg = np.array([m["g"] for m in fmeta]), np.array([m["g"] for m in lmeta])
    same = fs[:, None] == ls[None, :]
    samech = fcch[:, None] == lcch[None, :]
    sg = fg[:, None] == lg[None, :]
    return {
        "own_other_session": S[same & ~samech],
        "own_same_session": S[same & samech],
        "other_same_gender": S[~same & sg],
        "other_diff_gender": S[~same & ~sg],
    }


def stage_report():
    fe, fm, le, lm = load(WORK / "emb_libri.npz")
    out = {"libri": {}, "fleurs": {}}
    lines = []
    def P(d):  # noqa: N802
        return " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())
    for dur in list(LIVE_SECS) + ["2x5"]:
        for fc in FILE_CONDS:
            for lc in LIVE_CONDS:
                sc = libri_scores(fe, fm, le, lm, fc, lc, 5 if dur == "2x5" else dur, two=dur == "2x5")
                key = f"{dur}|{fc}|{lc}"
                gen, imp = sc["own_other_session"], sc["other_same_gender"]
                r = {c: pct(sc[c]) for c in CATS}
                r["eer_same_gender"], r["eer_thr"] = eer(gen, imp)
                r["eer_all_impostors"], _ = eer(gen, np.concatenate([imp, sc["other_diff_gender"]]))
                r["at"] = {str(t): dict(zip(("frr", "far_same_gender"), rates(gen, imp, t))) for t in THRESHOLDS}
                ts = np.round(np.arange(0.20, 0.801, 0.01), 2)
                r["sweep"] = {f"{t:.2f}": rates(gen, imp, t) for t in ts}
                out["libri"][key] = r
                lines.append(f"## live={dur}s file={fc} live_cond={lc}  EER(same g)={r['eer_same_gender']:.3%} "
                             f"@{r['eer_thr']:.3f}  EER(all)={r['eer_all_impostors']:.3%}")
                for c in CATS:
                    lines.append(f"  {c:18s} {P(r[c])}")
                lines.append("  " + "  ".join(f"t={t}: FRR={v['frr']:.3f} FAR={v['far_same_gender']:.4f}"
                                              for t, v in r["at"].items()))
    # recommendation window for 5 s and 2x5: one threshold for all conditions
    for dur in ("3", "5", "8", "2x5"):
        keys = [k for k in out["libri"] if k.startswith(dur + "|")]
        lo = max(min((float(t) for t, (frr, far) in out["libri"][k]["sweep"].items() if far <= 0.01), default=9)
                 for k in keys)
        hi = min(max((float(t) for t, (frr, far) in out["libri"][k]["sweep"].items() if frr <= 0.05), default=-9)
                 for k in keys)
        lo_c = max(min((float(t) for t, (frr, far) in out["libri"][k]["sweep"].items() if far <= 0.01), default=9)
                   for k in keys if k.endswith("|clean|clean"))
        hi_c = min(max((float(t) for t, (frr, far) in out["libri"][k]["sweep"].items() if frr <= 0.05), default=-9)
                   for k in keys if k.endswith("|clean|clean"))
        out.setdefault("window", {})[dur] = {"far1_from": lo, "frr5_upto": hi, "clean_far1_from": lo_c,
                                             "clean_frr5_upto": hi_c}
        lines.append(f"WINDOW live={dur}: FAR<=1% from t>={lo:.2f}, FRR<=5% up to t<={hi:.2f} (all conditions);"
                     f" clean/clean: {lo_c:.2f}..{hi_c:.2f}")

    fe2, fm2, le2, lm2 = load(WORK / "emb_fleurs.npz")
    fsent, lsent = np.array([m["sent"] for m in fm2]), np.array([m["sent"] for m in lm2])
    ffile, lfile = np.array([m["file"] for m in fm2]), np.array([m["file"] for m in lm2])
    fg, lg = np.array([m["g"] for m in fm2]), np.array([m["g"] for m in lm2])
    lconds = np.array([m["cond"] for m in lm2])
    for lc in LIVE_CONDS:
        sel = lconds == lc
        S = fe2 @ le2[sel].T
        notself = ffile[:, None] != lfile[sel][None, :]
        sg = fg[:, None] == lg[sel][None, :]
        samesent = fsent[:, None] == lsent[sel][None, :]
        r = {
            "same_gender_same_sentence(other file; speaker may repeat)": pct(S[notself & sg & samesent]),
            "same_gender_any(may include same speaker)": pct(S[notself & sg]),
            "diff_gender": pct(S[~sg]),
        }
        r["far_same_sentence_at"] = {str(t): float(np.mean(S[notself & sg & samesent] >= t)) for t in THRESHOLDS}
        out["fleurs"][lc] = r
        lines.append(f"## FLEURS-ru file=clean(whole utt) live=5s {lc}")
        for c, v in r.items():
            lines.append(f"  {c}: {P(v) if 'n' in v else v}")
    json.dump(out, open(WORK / "report.json", "w"), indent=1, default=float)
    print("\n".join(lines))


def stage_timing():
    se = make_se(1)
    sess = rt.InferenceSession(str(next(MODEL_DIR.glob("*.onnx"))), providers=["CPUExecutionProvider"])
    for io in sess.get_inputs():
        print("input", io.name, io.shape, io.type)
    for io in sess.get_outputs():
        print("output", io.name, io.shape, io.type)
    utts = sorted(glob.glob(str(LIBRI / "dev-clean/84/121123/*.flac")))
    x40, _ = concat_from(utts, 0, 40)
    for sec in (5, 40):
        x = x40[: sec * SR]
        e = se.embedding(x, sample_rate=SR)
        for _ in range(3):
            se.embedding(x, sample_rate=SR)
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            se.embedding(x, sample_rate=SR)
            ts.append(time.perf_counter() - t)
        print(f"{sec}s: dim={e.shape} norm={np.linalg.norm(e):.3f} median={np.median(ts) * 1000:.1f}ms "
              f"p90={np.percentile(ts, 90) * 1000:.1f}ms (1 thread, incl. fbank)")


if __name__ == "__main__":
    {"embed": stage_embed, "report": stage_report, "timing": stage_timing, "padded": stage_padded,
     "report_padded": stage_report_padded}[sys.argv[1]]()
