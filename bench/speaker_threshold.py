"""Speaker-verification threshold bench (stream S3, stage 8).

Run where the data is ($SPK_BENCH: data/LibriSpeech dev-clean, data/dev + dev.tsv of FLEURS ru, model/ the ONNX):
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python bench/speaker_threshold.py <stage>` (onnx-asr 0.12.0, soundfile).
Stages: embed (LibriSpeech + FLEURS-ru embeddings -> work/*.npz), report (tables -> stdout + work/report.json),
timing (1 thread, 5 s and 40 s), padded + report_padded (8 s live with silence, raw vs trimmed);
stage 2 (reference in windows, min of cosines): wlive, wfile <8|10,12>, splice <0,8|10,12> <i/n>, report2, timing2,
trimcheck (live with / without the app's trim),
trim3 <i/n> + report_trim3 (live trimmed: none / edges / all frames).
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
from math import comb
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


# ---------- stage 2: the reference in windows (min over windows) + splices + tries ----------
N_LIVE2 = 10  # 5 s live chunks per chapter (clean); noisy conditions for every third (k % 3 == 0)
LIVE2_CONDS = ("clean", "laptop_white20", "laptop_white10")
REF_SECS = (40, 60)
T2 = (0.35, 0.40, 0.45, 0.50)


def speech_only(x: np.ndarray, frame=480) -> np.ndarray:
    """peresmeshnik.voice._speech: 30 ms frames of x - mean above max(p10 of frames > -100 dB + 6, -60) dBFS."""
    f = (x - x.mean())[: len(x) // frame * frame].reshape(-1, frame).astype(np.float64)
    db = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)
    live = db[db > -100]
    keep = f[db > max(float(np.percentile(live, 10)) + 6 if len(live) else -120.0, -60.0)]
    return keep.reshape(-1).astype(np.float32) if len(keep) else x


def windows(x: np.ndarray, win_s: int) -> list[np.ndarray]:
    """Cut into win_s windows; a last piece shorter than win_s / 2 is glued to the window before; shorter -> one."""
    w = win_s * SR
    parts = [x[i: i + w] for i in range(0, max(len(x), 1), w)]
    if len(parts) > 1 and len(parts[-1]) < w // 2:
        last = parts.pop()
        parts[-1] = np.concatenate([parts[-1], last])
    return parts


def chapters():
    for spk in sorted(os.listdir(LIBRI / "dev-clean"), key=int):
        for ch in sorted(os.listdir(LIBRI / "dev-clean" / spk)):
            yield spk, ch, sorted(glob.glob(str(LIBRI / "dev-clean" / spk / ch / "*.flac")))


def live_chunks(utts, used, sec=5):
    rest = [sf.read(u, dtype="float32")[0] for u in utts[used:]]
    x = np.concatenate(rest) if rest else np.zeros(0, np.float32)
    m = len(x) // (sec * SR)
    ks = [int(k * m / N_LIVE2) for k in range(N_LIVE2)] if m >= N_LIVE2 else list(range(m))
    return [x[k * sec * SR: (k + 1) * sec * SR] for k in ks]


def save2(path, rows):
    np.savez(path, e=np.stack([e for _, e in rows]), m=json.dumps([m for m, _ in rows]))


def load2(path):
    z = np.load(path)
    return z["e"], json.loads(str(z["m"]))


def stage_wlive():
    """Live 5 s (trimmed as the app does) + the old one-vector reference of 60 s (40 s: emb_libri.npz)."""
    se, rng, gender, rows, olds, t0 = make_se(1), np.random.default_rng(2), libri_speakers(), [], [], time.time()
    for spk, ch, utts in chapters():
        _, used = concat_from(utts, 0, FILE_SEC)
        for k, lx in enumerate(live_chunks(utts, used)):
            for lc in LIVE2_CONDS if k % 3 == 0 else ("clean",):
                meta = {"spk": spk, "ch": ch, "g": gender[spk], "cond": lc, "k": k}
                rows.append((meta, embed(se, speech_only(live_cond(lx, lc, rng)))))
        fx, _ = concat_from(utts, 0, 60)
        for fc in FILE_CONDS if fx is not None else ():
            olds.append(({"spk": spk, "ch": ch, "g": gender[spk], "cond": fc, "L": 60}, embed(se, file_cond(fx, fc))))
        print(f"{spk}/{ch} lives={len(rows)} old60={len(olds)} t={time.time() - t0:.0f}s", flush=True)
    save2(WORK / "emb_wlive.npz", rows)
    save2(WORK / "emb_old60.npz", olds)


def stage_wfile():
    """Reference 40 s / 60 s, each file condition: trimmed, cut into windows of argv[2] s ("8" or "10,12")."""
    wins = [int(w) for w in sys.argv[2].split(",")]
    se, gender, rows, t0 = make_se(1), libri_speakers(), {w: [] for w in wins}, time.time()
    for spk, ch, utts in chapters():
        for L in REF_SECS:
            fx, _ = concat_from(utts, 0, L)
            for fc in FILE_CONDS if fx is not None else ():
                t = speech_only(file_cond(fx, fc))
                for w in wins:
                    for i, seg in enumerate(windows(t, w)):
                        meta = {"spk": spk, "ch": ch, "g": gender[spk], "cond": fc, "L": L, "win": w, "i": i,
                                "speech_s": len(t) / SR, "len_s": len(seg) / SR}
                        rows[w].append((meta, embed(se, seg)))
        print(f"{spk}/{ch} windows={sum(map(len, rows.values()))} t={time.time() - t0:.0f}s", flush=True)
    for w in wins:
        save2(WORK / f"emb_win{w}.npz", rows[w])


def splice_pairs():
    """Owner chapter (owner has another chapter) x 1 random speaker of the same sex (a random chapter of theirs)."""
    rng, gender = np.random.default_rng(3), libri_speakers()
    chs = {}
    for spk, ch, _ in chapters():
        chs.setdefault(spk, []).append(ch)
    out = []
    for spk in sorted(chs, key=int):
        for ch in chs[spk] if len(chs[spk]) > 1 else ():
            other = sorted((s for s in chs if s != spk and gender[s] == gender[spk]), key=int)
            imp = other[rng.integers(len(other))]
            out.append((spk, ch, imp, chs[imp][rng.integers(len(chs[imp]))]))
    return out


def stage_splice():
    """argv[2] = windows ("0,8" or "10,12"; 0 = the old one vector), argv[3] = shard "i/n". Reference = L s of another
    person + L s of the owner (order fwd) or the owner first (rev), L = 20 / 30; one vector / trimmed windows."""
    wins = [int(w) for w in sys.argv[2].split(",")]
    i, n = map(int, sys.argv[3].split("/"))
    se, rows, t0 = make_se(1), [], time.time()
    d = LIBRI / "dev-clean"
    for j, (spk, ch, imp, ich) in enumerate(splice_pairs()):
        if j % n != i:
            continue
        for L in (20, 30):
            own, _ = concat_from(sorted(glob.glob(str(d / spk / ch / "*.flac"))), 0, L)
            oth, _ = concat_from(sorted(glob.glob(str(d / imp / ich / "*.flac"))), 0, L)
            if own is None or oth is None:
                continue
            for order, x in (("fwd", np.concatenate([oth, own])), ("rev", np.concatenate([own, oth]))):
                base = {"spk": spk, "ch": ch, "imp": imp, "imp_ch": ich, "L": L, "order": order}
                t = speech_only(x)
                for w in wins:
                    if w == 0:
                        rows.append((base | {"win": 0}, embed(se, x)))
                        continue
                    rows += [(base | {"win": w, "i": k}, embed(se, seg)) for k, seg in enumerate(windows(t, w))]
        print(f"{j} {spk}/{ch}+{imp} rows={len(rows)} t={time.time() - t0:.0f}s", flush=True)
    save2(WORK / f"emb_splice_{'-'.join(map(str, wins))}_{i}.npz", rows)


def masks(fmeta, lmeta):
    fs, ls = np.array([m["spk"] for m in fmeta]), np.array([m["spk"] for m in lmeta])
    fc, lc = np.array([m["ch"] for m in fmeta]), np.array([m["ch"] for m in lmeta])
    fg, lg = np.array([m["g"] for m in fmeta]), np.array([m["g"] for m in lmeta])
    same = fs[:, None] == ls[None, :]
    return same & (fc[:, None] != lc[None, :]), ~same & (fg[:, None] == lg[None, :])


def min_over_windows(we, wm, keys):
    """Rows of `keys` (one per reference) -> score fn(live embs) = min over that reference's windows."""
    pos = {}
    for r, m in enumerate(wm):
        pos.setdefault(tuple(m[k] for k in keys), []).append(r)
    return pos, lambda refs, Lv: np.stack([np.min(we[pos[r]] @ Lv.T, axis=0) for r in refs])


def p_any(scores, n, t=0.45):
    """P(at least one of n tries drawn without replacement from `scores` >= t)."""
    k, m = len(scores), int(np.sum(np.asarray(scores) >= t))
    return 1 - comb(k - m, n) / comb(k, n) if n <= k else float(m > 0)


def stage_report2():
    le, lm = load2(WORK / "emb_wlive.npz")
    fe40, fm40, _, _ = load(WORK / "emb_libri.npz")
    fe60, fm60 = load2(WORK / "emb_old60.npz")
    old = {(m["spk"], m["ch"], m["cond"], 40): e for m, e in zip(fm40, fe40)}
    old |= {(m["spk"], m["ch"], m["cond"], 60): e for m, e in zip(fm60, fe60)}
    wins = sorted(int(p.stem[7:]) for p in WORK.glob("emb_win*.npz"))
    W = {w: load2(WORK / f"emb_win{w}.npz") for w in wins}
    out, lines = {"a": {}, "b": {}, "c": {}}, []
    conds = [("clean", "clean"), ("opus24k", "clean"), ("clean", "laptop_white20"), ("clean", "laptop_white10"),
             ("tel8k", "clean")]
    lines.append("## (a) FRR / FAR(other same sex) per threshold; old = one vector, new = min over windows")
    for w in wins:
        we, wm = W[w]
        for L in REF_SECS:
            for fc, lc in conds:
                refs = sorted({(m["spk"], m["ch"], m["cond"], m["L"]) for m in wm if m["cond"] == fc and m["L"] == L})
                fmeta = [{"spk": r[0], "ch": r[1], "g": next(m["g"] for m in lm if m["spk"] == r[0])} for r in refs]
                li = [j for j, m in enumerate(lm) if m["cond"] == lc]
                Lv, lmeta = le[li], [lm[j] for j in li]
                gen, imp = masks(fmeta, lmeta)
                _, score = min_over_windows(we, wm, ("spk", "ch", "cond", "L"))
                S = {"old": np.stack([old[r] for r in refs]) @ Lv.T, "new": score(refs, Lv)}
                nwin = [sum(1 for m in wm if (m["spk"], m["ch"], m["cond"], m["L"]) == r) for r in refs]
                r = {k: {str(t): rates(v[gen], v[imp], t) for t in T2} for k, v in S.items()}
                r |= {"n_refs": len(refs), "n_gen": int(gen.sum()), "n_imp": int(imp.sum()),
                      "windows_median": float(np.median(nwin)),
                      "speech_s_median": float(np.median([m["speech_s"] for m in wm if m["cond"] == fc and m["L"] == L])),
                      "gen_p5": {k: float(np.percentile(v[gen], 5))
                                                                           for k, v in S.items()}}
                out["a"][f"{w}|{L}|{fc}|{lc}"] = r
                lines.append(f"win={w}s ref={L}s {fc}/{lc} refs={len(refs)} gen={r['n_gen']} imp={r['n_imp']} "
                             f"windows~{r['windows_median']:.0f} speech~{r['speech_s_median']:.1f}s gen_p5 old={r['gen_p5']['old']:.3f} "
                             f"new={r['gen_p5']['new']:.3f}")
                for k in ("old", "new"):
                    lines.append(f"  {k}: " + "  ".join(f"{t}: {a:.2%}/{b:.2%}" for t, (a, b) in r[k].items()))
                if (fc, lc) == ("clean", "clean"):  # (c) tries
                    by_spk = {}
                    for j, m in enumerate(lmeta):
                        by_spk.setdefault(m["spk"], []).append(j)
                    c = {}
                    for k, v in S.items():
                        for t in (0.40, 0.45):
                            ps = {n: [] for n in (1, 3, 6, 10)}
                            for a, ref in enumerate(fmeta):
                                for spk, js in by_spk.items():
                                    if spk != ref["spk"] and lmeta[js[0]]["g"] == ref["g"]:
                                        for n in ps:
                                            ps[n].append(p_any(v[a, js], n, t))
                            c[f"{k}@{t}"] = {n: float(np.mean(p)) for n, p in ps.items()}
                    c["n_pairs"] = len(ps[1])
                    c["tries_per_speaker_min"] = min(map(len, by_spk.values()))
                    out["c"][f"{w}|{L}"] = c
                    lines.append(f"  (c) tries clean/clean pairs={c['n_pairs']} live/speaker>={c['tries_per_speaker_min']}: "
                                 + "  ".join(f"{k} " + " ".join(f"N={n}:{p:.2%}" for n, p in c[k].items())
                                             for k in c if "@" in k))
    lines.append("## (b) splices: live = the owner, another chapter, clean; pass rate at 0.45 (pairs splice x live)")
    sp = [load2(p) for p in sorted(WORK.glob("emb_splice_*.npz"))]
    if sp:
        se_, sm = np.concatenate([e for e, _ in sp]), sum((m for _, m in sp), [])
        key = ("spk", "ch", "imp", "L", "order")
        for L in (20, 30):
            for order in ("fwd", "rev"):
                for w in [0] + sorted({m["win"] for m in sm if m["win"]}):
                    rows = [j for j, m in enumerate(sm) if m["L"] == L and m["order"] == order and m["win"] == w]
                    _, score = min_over_windows(se_[rows], [sm[j] for j in rows], key)
                    refs = sorted({tuple(sm[j][k] for k in key) for j in rows})
                    S = score(refs, le)
                    ok = np.array([[m["cond"] == "clean" and m["spk"] == r[0] and m["ch"] != r[1] for m in lm]
                                   for r in refs])
                    per = [float(np.mean(S[a][ok[a]] >= 0.45)) for a in range(len(refs))]
                    r = {"pass": float(np.mean(S[ok] >= 0.45)), "pass_0.40": float(np.mean(S[ok] >= 0.40)),
                         "n_splices": len(refs), "n_pairs": int(ok.sum()),
                         "splices_any_pass": float(np.mean([p > 0 for p in per])), "median": float(np.median(S[ok]))}
                    out["b"][f"{L}+{L}|{order}|{'old' if w == 0 else f'win{w}'}"] = r
                    lines.append(f"{L}+{L} {order} {'old' if w == 0 else f'win{w}'}: pass={r['pass']:.2%} "
                                 f"pass@0.40={r['pass_0.40']:.2%} "
                                 f"median={r['median']:.3f} splices={r['n_splices']} pairs={r['n_pairs']} "
                                 f"splices with any pass={r['splices_any_pass']:.1%}")
    json.dump(out, open(WORK / "report2.json", "w"), indent=1, default=float)
    print("\n".join(lines))


def stage_timing2():
    """1 thread: reference 60 s -> trim + windows, live 5 s -> trim, the min of cosines; old: one 60 s vector."""
    se = make_se(1)
    x60, _ = concat_from(sorted(glob.glob(str(LIBRI / "dev-clean/84/121123/*.flac"))), 0, 60)
    live = x60[: 5 * SR]

    def run(fn):
        fn()
        ts = []
        for _ in range(10):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return np.median(ts) * 1000, np.percentile(ts, 90) * 1000

    lv = embed(se, speech_only(live))
    print("live 5 s trim+embed: median={:.0f}ms p90={:.0f}ms".format(*run(lambda: embed(se, speech_only(live)))))
    print("old one vector 60 s: median={:.0f}ms p90={:.0f}ms".format(*run(lambda: embed(se, x60) @ lv)))
    for w in (8, 10, 12):
        n = len(windows(speech_only(x60), w))
        ms = run(lambda w=w: min(embed(se, s) @ lv for s in windows(speech_only(x60), w)))
        print(f"windows {w} s ({n} windows) trim+embed+min: median={ms[0]:.0f}ms p90={ms[1]:.0f}ms")


def stage_trimcheck():
    """Live 5 s with and without the app's trim (the same chunks and noise as wlive, k % 3 == 0) against the old
    40 s one-vector reference: does the trim itself cost the noisy live readings?"""
    se, rng, gender, rows, kept = make_se(1), np.random.default_rng(2), libri_speakers(), [], {}
    for spk, ch, utts in chapters():
        _, used = concat_from(utts, 0, FILE_SEC)
        for k, lx in enumerate(live_chunks(utts, used)):
            for lc in LIVE2_CONDS if k % 3 == 0 else ("clean",):
                y = live_cond(lx, lc, rng)  # consumes rng exactly as wlive does
                if k % 3:
                    continue
                t = speech_only(y)
                kept.setdefault(lc, []).append(len(t) / SR)
                for how, z in (("raw", y), ("trim", t)):
                    rows.append(({"spk": spk, "ch": ch, "g": gender[spk], "cond": lc, "how": how}, embed(se, z)))
        print(f"{spk}/{ch} rows={len(rows)}", flush=True)
    save2(WORK / "emb_trimcheck.npz", rows)
    fe, fm, _, _ = load(WORK / "emb_libri.npz")
    fi = [i for i, m in enumerate(fm) if m["cond"] == "clean"]
    F, fmeta = fe[fi], [fm[i] for i in fi]
    le, lm = np.stack([e for _, e in rows]), [m for m, _ in rows]
    for lc in LIVE2_CONDS:
        q = np.percentile(kept[lc], [5, 50, 95])
        print(f"## file=clean 40 s, live={lc}: kept after trim p5/p50/p95 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f} s of 5")
        for how in ("raw", "trim"):
            li = [j for j, m in enumerate(lm) if m["cond"] == lc and m["how"] == how]
            gen, imp = masks(fmeta, [lm[j] for j in li])
            S = F @ le[li].T
            print(f"  {how}: n_gen={int(gen.sum())} gen_p5={np.percentile(S[gen], 5):.3f} gen_p50={np.median(S[gen]):.3f} "
                  + "  ".join(f"{t}: {a:.2%}/{b:.2%}" for t in T2 for a, b in [rates(S[gen], S[imp], t)]))


def speech_edges(x: np.ndarray, frame=480) -> np.ndarray:
    """Only the edges cut: from the first to the last 30 ms frame above the _speech threshold, everything between kept."""
    y = x - x.mean()
    f = y[: len(y) // frame * frame].reshape(-1, frame).astype(np.float64)
    db = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)
    live = db[db > -100]
    on = np.flatnonzero(db > max(float(np.percentile(live, 10)) + 6 if len(live) else -120.0, -60.0))
    return y[on[0] * frame: (on[-1] + 1) * frame].astype(np.float32) if len(on) else x


TRIMS = {"none": lambda x: x, "edges": speech_edges, "all": speech_only}


def stage_trim3():
    """argv[2] = shard "i/n". Live 5 s (clean / white 20 / white 10, 4 per chapter) and the padded 8 s clip (clean /
    white 20, 4 per chapter), each trimmed three ways (TRIMS); rng per chapter, so shards are independent."""
    i, n = map(int, sys.argv[2].split("/"))
    se, gender, rows, kept, t0 = make_se(1), libri_speakers(), [], [], time.time()
    for j, (spk, ch, utts) in enumerate(chapters()):
        if j % n != i:
            continue
        rng = np.random.default_rng([int(spk), int(ch)])
        _, used = concat_from(utts, 0, FILE_SEC)
        clips = [(lc, lx) for k, lx in enumerate(live_chunks(utts, used)) if k % 3 == 0 for lc in LIVE2_CONDS]
        clips = [(lc, live_cond(lx, lc, rng)) for lc, lx in clips]
        rest = list(range(used, len(utts)))
        for st in [rest[int(k * len(rest) / N_LIVE)] for k in range(N_LIVE)] if len(rest) >= N_LIVE else rest:
            x = padded_live(utts, st, rng)
            if x is not None:
                clips += [("pad8_clean", x), ("pad8_white20", live_cond(x, "laptop_white20", rng))]
        for lc, x in clips:
            for how, fn in TRIMS.items():
                y = fn(x)
                kept.append((lc, how, len(y) / SR))
                rows.append(({"spk": spk, "ch": ch, "g": gender[spk], "cond": lc, "how": how, "s": len(y) / SR},
                             embed(se, y)))
        print(f"{spk}/{ch} rows={len(rows)} t={time.time() - t0:.0f}s", flush=True)
    save2(WORK / f"emb_trim3_{i}.npz", rows)


def stage_report_trim3():
    """Reference 40 s in 8 s windows (emb_win8) vs live trimmed three ways: FRR / FAR at 0.45, impostor best of 3."""
    we, wm = load2(WORK / "emb_win8.npz")
    parts = [load2(p) for p in sorted(WORK.glob("emb_trim3_*.npz"))]
    le, lm = np.concatenate([e for e, _ in parts]), sum((m for _, m in parts), [])
    _, score = min_over_windows(we, wm, ("spk", "ch", "cond", "L"))
    out = {}
    for fc, lc in (("clean", "clean"), ("clean", "laptop_white20"), ("clean", "laptop_white10"), ("opus24k", "clean"),
                   ("clean", "pad8_clean"), ("clean", "pad8_white20")):
        refs = sorted({(m["spk"], m["ch"], m["cond"], m["L"]) for m in wm if m["cond"] == fc and m["L"] == 40})
        g = {m["spk"]: m["g"] for m in wm}
        fmeta = [{"spk": r[0], "ch": r[1], "g": g[r[0]]} for r in refs]
        for how in TRIMS:
            li = [j for j, m in enumerate(lm) if m["cond"] == lc and m["how"] == how]
            lmeta = [lm[j] for j in li]
            S = score(refs, le[li])
            gen, imp = masks(fmeta, lmeta)
            frr, far = rates(S[gen], S[imp], 0.45)
            by_spk = {}
            for j, m in enumerate(lmeta):
                by_spk.setdefault(m["spk"], []).append(j)
            b3 = [p_any(S[a, js], 3) for a, ref in enumerate(fmeta) for spk, js in by_spk.items()
                  if spk != ref["spk"] and g[spk] == ref["g"]]
            secs = np.percentile([m["s"] for m in lmeta], [5, 50, 95])
            r = {"frr": frr, "far": far, "far_best3": float(np.mean(b3)), "n_gen": int(gen.sum()), "n_imp": int(imp.sum()),
                 "n_best3": len(b3), "kept_s_p5_p50_p95": secs.tolist()}
            out[f"{fc}|{lc}|{how}"] = r
            print(f"file={fc} live={lc} trim={how}: FRR={frr:.2%} FAR={far:.3%} best-of-3 FAR={r['far_best3']:.3%} "
                  f"gen={r['n_gen']} imp={r['n_imp']} kept s p5/p50/p95={secs[0]:.2f}/{secs[1]:.2f}/{secs[2]:.2f}")
    json.dump(out, open(WORK / "report_trim3.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    {"embed": stage_embed, "report": stage_report, "timing": stage_timing, "padded": stage_padded,
     "report_padded": stage_report_padded, "wlive": stage_wlive, "wfile": stage_wfile, "splice": stage_splice,
     "report2": stage_report2, "timing2": stage_timing2, "trimcheck": stage_trimcheck,
     "trim3": stage_trim3, "report_trim3": stage_report_trim3}[sys.argv[1]]()
