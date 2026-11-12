"""Speaker-verification threshold bench (stream S3, stage 8).

Run where the data is ($SPK_BENCH: data/LibriSpeech dev-clean, data/dev + dev.tsv of FLEURS ru, model/ the ONNX):
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python bench/speaker_threshold.py <stage>` (onnx-asr 0.12.0, soundfile).
Stages: embed (LibriSpeech + FLEURS-ru embeddings -> work/*.npz), report (tables -> stdout + work/report.json),
timing (1 thread, 5 s and 40 s), padded + report_padded (8 s live with silence, raw vs trimmed);
stage 2 (reference in windows, min of cosines): wlive, wfile <8|10,12>, splice <0,8|10,12> <i/n>, report2, timing2,
trimcheck (live with / without the app's trim),
trim3 <i/n> + report_trim3 (live trimmed: none / edges / all frames);
stage 3 (the reference's start in short windows): head <i/n>, report_head, timing_head;
overlap <i/n> + report_overlap + timing_overlap (4 s windows every 2 s over the head, shifted splices);
stage 4 (the raw start, voice._start of 1d79286; VOICE_PY = that voice.py): start4 <i/n>, report_start4,
timing_start4; stage 6 (the file as save() writes it; VOICE_PY = voice.py now): save6 <i/n> <1|2>, report_save6;
stage 8 before 0.3.0 (the head against the rest of the recording): report_headrest (from emb_head / emb_overlap /
emb_save6), prompt8 <i/n> + report_prompt8 (the app's whole rule now: windows, the engines' prompt, head_rest);
FIXES-s3-40 9 (a live start: the room, a breath, a cough, a laugh, an "um"; data/esc50 = ESC-50's master.zip unpacked):
lead9 <i/n> + report_lead9.
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


# ---------- stage 3: the reference's first HEAD_S s of speech in short windows, the rest in 8 s ----------
HEAD_S = 12
SHORT = (4, 6)  # "4+8": the head in 4 s windows; "6+8": in 6 s
HEAD_K = (3, 4, 5, 6)  # splice: the first K s of another person's speech, then the owner up to FILE_SEC
T3 = (0.35, 0.40, 0.45)
LIVE3_S = 8  # the app's live reading, trimmed at the edges only (voice._speech(edges=True))


def head_windows(t: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Trimmed reference -> {kind: windows}: w8 = all in 8 s (the app now), s4 / s6 = the first HEAD_S s in 4 / 6 s,
    r8 = the rest after HEAD_S in 8 s (the same tail rule)."""
    h = HEAD_S * SR
    out = {"w8": windows(t, 8), "r8": windows(t[h:], 8)}
    out |= {f"s{w}": [t[a: a + w * SR] for a in range(0, h, w * SR)] for w in SHORT}
    return out


def stage_head():
    """argv[2] = shard "i/n". live: 8 s chunks after the first 40 s (clean; white 20 / 10 for k % 3 == 0), edges
    trimmed; ref: 40 / 60 s clean / opus24k, trimmed, windows s4 s6 r8 (w8 is emb_win8.npz); splice: the first K s of
    the speech of another person of the same sex (splice_pairs) + the owner's first FILE_SEC - K s, windows w8 s4 s6 r8."""
    i, n = map(int, sys.argv[2].split("/"))
    se, gender, rows, t0 = make_se(1), libri_speakers(), [], time.time()
    imps = {}
    for spk, ch, imp, ich in splice_pairs():
        imps.setdefault((spk, ch), []).append((imp, ich))
    for j, (spk, ch, utts) in enumerate(chapters()):
        if j % n != i:
            continue
        rng, base = np.random.default_rng([int(spk), int(ch), 3]), {"spk": spk, "ch": ch, "g": gender[spk]}
        _, used = concat_from(utts, 0, FILE_SEC)
        for k, lx in enumerate(live_chunks(utts, used, LIVE3_S)):
            for lc in LIVE2_CONDS if k % 3 == 0 else ("clean",):
                y = speech_edges(live_cond(lx, lc, rng))
                rows.append((base | {"set": "live", "cond": lc, "k": k, "s": len(y) / SR}, embed(se, y)))
        for L in REF_SECS:
            fx, _ = concat_from(utts, 0, L)
            for fc in ("clean", "opus24k") if fx is not None else ():
                t = speech_only(file_cond(fx, fc))
                rows += [(base | {"set": "ref", "cond": fc, "L": L, "kind": kind, "i": q}, embed(se, w))
                         for kind, ws in head_windows(t).items() if kind != "w8" for q, w in enumerate(ws)]
        for imp, ich in imps.get((spk, ch), []):
            oth, _ = concat_from(sorted(glob.glob(str(LIBRI / "dev-clean" / imp / ich / "*.flac"))), 0, 20)
            oth = speech_only(oth)
            for K in HEAD_K:
                own, _ = concat_from(utts, 0, FILE_SEC - K)
                t = speech_only(np.concatenate([oth[: K * SR], own]))
                rows += [(base | {"set": "splice", "imp": imp, "imp_ch": ich, "K": K, "kind": kind, "i": q},
                          embed(se, w)) for kind, ws in head_windows(t).items() for q, w in enumerate(ws)]
        print(f"{j} {spk}/{ch} rows={len(rows)} t={time.time() - t0:.0f}s", flush=True)
    save2(WORK / f"emb_head_{i}.npz", rows)


def kind_mins(E, M, keys, refs, Lv):
    """{kind: refs x lives} of the min cosine over that reference's windows of the kind."""
    pos = {}
    for r, m in enumerate(M):
        pos.setdefault((tuple(m[k] for k in keys), m["kind"]), []).append(r)
    kinds = sorted({m["kind"] for m in M})
    return {kd: np.stack([np.min(E[pos[(r, kd)]] @ Lv.T, axis=0) for r in refs]) for kd in kinds}


def schemes(m):
    """kind mins -> {scheme: pass fn(t)}: 8 / 4+8 / 6+8 with one threshold, and 4@ts+8 / 6@ts+8: the short windows
    at their own ts, the 8 s ones at 0.45."""
    out = {"8": lambda t: m["w8"] >= t}
    for w in SHORT:
        out[f"{w}+8"] = lambda t, w=w: np.minimum(m[f"s{w}"], m["r8"]) >= t
        for ts in (0.35, 0.40):
            out[f"{w}@{ts:.2f}+8@0.45"] = lambda t, w=w, ts=ts: (m[f"s{w}"] >= ts) & (m["r8"] >= 0.45)
    return out


def stage_report_head():
    parts = [load2(p) for p in sorted(WORK.glob("emb_head_*.npz"))]
    E, M = np.concatenate([e for e, _ in parts]), sum((m for _, m in parts), [])
    we, wm = load2(WORK / "emb_win8.npz")
    li = [j for j, m in enumerate(M) if m["set"] == "live"]
    le, lm = E[li], [M[j] for j in li]
    ri = [j for j, m in enumerate(M) if m["set"] == "ref"]
    wi = [j for j, m in enumerate(wm) if m["cond"] in ("clean", "opus24k")]
    RE = np.concatenate([E[ri], we[wi]])
    RM = [M[j] for j in ri] + [wm[j] | {"kind": "w8"} for j in wi]
    keys, out, lines = ("spk", "ch", "cond", "L"), {"a": {}, "dist": {}, "b": {}}, []
    g = {m["spk"]: m["g"] for m in lm}
    kept = {lc: np.percentile([m["s"] for m in lm if m["cond"] == lc], [5, 50, 95]) for lc in LIVE2_CONDS}
    lines.append(f"live {LIVE3_S} s edges-trimmed, kept p5/p50/p95: "
                 + "  ".join(f"{lc} {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}" for lc, q in kept.items()))
    lines.append("## (a) FRR (own, another chapter) at 0.35 / 0.40 / 0.45, FAR (other same sex) at 0.45")
    for L in REF_SECS:
        for fc, lc in (("clean", "clean"), ("opus24k", "clean"), ("clean", "laptop_white20"),
                       ("clean", "laptop_white10")):
            refs = sorted({tuple(m[k] for k in keys) for m in RM if m["cond"] == fc and m["L"] == L})
            fmeta = [{"spk": r[0], "ch": r[1], "g": g[r[0]]} for r in refs]
            lj = [j for j, m in enumerate(lm) if m["cond"] == lc]
            gen, imp = masks(fmeta, [lm[j] for j in lj])
            m = kind_mins(RE, RM, keys, refs, le[lj])
            lines.append(f"ref={L}s {fc}/{lc} refs={len(refs)} gen={int(gen.sum())} imp={int(imp.sum())}")
            for name, ok in schemes(m).items():
                ts = T3 if "@" not in name else (0.45,)
                r = {"frr": {str(t): float(1 - ok(t)[gen].mean()) for t in ts},
                     "far": {str(t): float(ok(t)[imp].mean()) for t in ts}, "n_gen": int(gen.sum()),
                     "n_imp": int(imp.sum())}
                out["a"][f"{L}|{fc}|{lc}|{name}"] = r
                lines.append(f"  {name}: " + "  ".join(f"{t}: {r['frr'][str(t)]:.2%}/{r['far'][str(t)]:.3%}" for t in ts))
            if L == 40:  # each window's cosine and the min per kind, own vs other
                d = {}
                for kd in ("s4", "s6", "w8"):
                    rows = [j for j, mm in enumerate(RM) if mm["cond"] == fc and mm["L"] == L and mm["kind"] == kd]
                    S = RE[rows] @ le[lj].T
                    ref_of = {r: a for a, r in enumerate(refs)}
                    a_ = np.array([ref_of[tuple(RM[j][k] for k in keys)] for j in rows])
                    gw, iw = gen[a_], imp[a_]
                    d[kd] = {"win": pct(S[gw]) | {"imp_p99": float(np.percentile(S[iw], 99)),
                                                   "far@0.45": float(np.mean(S[iw] >= 0.45))},
                             "min": pct(m[kd][gen]) | {"imp_p99": float(np.percentile(m[kd][imp], 99))}}
                out["dist"][f"{fc}|{lc}"] = d
                for kd, v in d.items():
                    for what in ("win", "min"):
                        q = v[what]
                        lines.append(f"  dist {kd} {what}: own p1={q['p1']:.3f} p5={q['p5']:.3f} p50={q['p50']:.3f} "
                                     f"other p99={q['imp_p99']:.3f}"
                                     + (f" other far@0.45={q['far@0.45']:.3%}" if what == "win" else ""))
    lines.append("## (b) splice: the first K s of another's speech + the owner to 40 s; live = the owner, another "
                 "chapter, clean; pass rate")
    si = [j for j, m in enumerate(M) if m["set"] == "splice"]
    SE_, SM = E[si], [M[j] for j in si]
    lj = [j for j, m in enumerate(lm) if m["cond"] == "clean"]
    Lv, lmeta = le[lj], [lm[j] for j in lj]
    owners = sorted({(m["spk"], m["ch"]) for m in SM})
    for K in (0,) + HEAD_K:
        if K:
            kk = ("spk", "ch", "imp", "K")
            refs = sorted({tuple(m[k] for k in kk) for m in SM if m["K"] == K})
            sk = [j for j, x in enumerate(SM) if x["K"] == K]
            m = kind_mins(SE_[sk], [SM[j] for j in sk], kk, refs, Lv)
        else:
            refs = [(s, c, "clean", 40) for s, c in owners]
            m = kind_mins(RE, RM, keys, refs, Lv)
        ok_ = np.array([[lv["spk"] == r[0] and lv["ch"] != r[1] for lv in lmeta] for r in refs])
        lines.append(f"K={K}: splices={len(refs)} pairs={int(ok_.sum())}")
        for name, ok in schemes(m).items():
            ts = (0.40, 0.45) if "@" not in name else (0.45,)
            r = {str(t): float(ok(t)[ok_].mean()) for t in ts} | {"n_splices": len(refs), "n_pairs": int(ok_.sum())}
            out["b"][f"{K}|{name}"] = r
            lines.append(f"  {name}: " + "  ".join(f"{t}: {r[str(t)]:.2%}" for t in ts))
        lines.append("  median min: " + "  ".join(f"{kd}={np.median(v[ok_]):.3f}" for kd, v in m.items()))
    json.dump(out, open(WORK / "report_head.json", "w"), indent=1, default=float)
    print("\n".join(lines))


def stage_timing_head():
    """1 thread: reference 60 s -> trim + 4+8 windows (3 x 4 s + 8 s after 12 s), live 8 s -> edges trim, the min."""
    se = make_se(1)
    utts = sorted(glob.glob(str(LIBRI / "dev-clean/84/121123/*.flac")))
    x60, used = concat_from(utts, 0, 60)
    live = live_chunks(utts, used, LIVE3_S)[0]

    def run(fn):
        fn()
        ts = []
        for _ in range(10):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return np.median(ts) * 1000, np.percentile(ts, 90) * 1000

    lv = embed(se, speech_edges(live))
    print("live 8 s edges+embed: median={:.0f}ms p90={:.0f}ms".format(*run(lambda: embed(se, speech_edges(live)))))
    for name, kinds in (("8", ("w8",)), ("4+8", ("s4", "r8")), ("6+8", ("s6", "r8"))):
        n = sum(len(head_windows(speech_only(x60))[k]) for k in kinds)
        ms = run(lambda kinds=kinds: min(embed(se, s) @ lv for k in kinds for s in head_windows(speech_only(x60))[k]))
        print(f"ref 60 s {name} ({n} windows) trim+embed+min: median={ms[0]:.0f}ms p90={ms[1]:.0f}ms")
        ms = run(lambda kinds=kinds: min(embed(se, s) @ lv for k in kinds for s in head_windows(speech_only(x60))[k])
                 + embed(se, speech_edges(live)) @ lv)
        print(f"  + live: median={ms[0]:.0f}ms p90={ms[1]:.0f}ms")


# ---------- stage 3b: the head in 4 s windows every 2 s ("4h2+8"); splices shifted by OFFS s of the owner ----------
OFFS = (1, 2)
X4 = (2, 6)  # the 4 s windows of the head not in s4 (0, 4, 8): together 0-4, 2-6, 4-8, 6-10, 8-12


def head_x4(t: np.ndarray) -> list[np.ndarray]:
    return [t[a * SR: (a + 4) * SR] for a in X4]


def speech_cut(x: np.ndarray, sec: float, frame=480) -> int:
    """The sample of x after which speech_only(x) has kept `sec` s (the same frame threshold)."""
    f = (x - x.mean())[: len(x) // frame * frame].reshape(-1, frame).astype(np.float64)
    db = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)
    live = db[db > -100]
    on = db > max(float(np.percentile(live, 10)) + 6 if len(live) else -120.0, -60.0)
    return int(np.searchsorted(np.cumsum(on), round(sec * SR / frame)) + 1) * frame


def stage_overlap():
    """argv[2] = shard "i/n". ref (40 / 60 s, clean / opus24k) and splice (offset 0): the s4x windows only (the rest is
    emb_head); splice shifted: the owner's first o s of speech, the K s of another person, the owner on (raw, 40 - K s
    in all), windows w8 (only the first: the next ones are the owner's alone, taken from offset 0), s4, s4x (r8: offset 0)."""
    i, n = map(int, sys.argv[2].split("/"))
    se, gender, rows, t0 = make_se(1), libri_speakers(), [], time.time()
    imps = {}
    for spk, ch, imp, ich in splice_pairs():
        imps.setdefault((spk, ch), []).append((imp, ich))
    for j, (spk, ch, utts) in enumerate(chapters()):
        if j % n != i:
            continue
        base = {"spk": spk, "ch": ch, "g": gender[spk]}
        for L in REF_SECS:
            fx, _ = concat_from(utts, 0, L)
            for fc in ("clean", "opus24k") if fx is not None else ():
                t = speech_only(file_cond(fx, fc))
                rows += [(base | {"set": "ref", "cond": fc, "L": L, "kind": "s4x", "i": q}, embed(se, w))
                         for q, w in enumerate(head_x4(t))]
        for imp, ich in imps.get((spk, ch), []):
            oth, _ = concat_from(sorted(glob.glob(str(LIBRI / "dev-clean" / imp / ich / "*.flac"))), 0, 20)
            oth = speech_only(oth)
            for K in HEAD_K:
                own, _ = concat_from(utts, 0, FILE_SEC - K)
                for o in (0,) + OFFS:
                    c = speech_cut(own, o) if o else 0
                    t = speech_only(np.concatenate([own[:c], oth[: K * SR], own[c:]]))
                    ws = {"s4x": head_x4(t)} | ({"w8": windows(t, 8)[:1], "s4": head_windows(t)["s4"]} if o else {})
                    rows += [(base | {"set": "splice", "imp": imp, "imp_ch": ich, "K": K, "o": o, "kind": kind, "i": q},
                              embed(se, w)) for kind, v in ws.items() for q, w in enumerate(v)]
        print(f"{j} {spk}/{ch} rows={len(rows)} t={time.time() - t0:.0f}s", flush=True)
    save2(WORK / f"emb_overlap_{i}.npz", rows)


def schemes_x(m):
    """8 at t; 4+8 / 4h2+8: the 4 s windows at t4 (0.40 or 0.45), the 8 s ones at 0.45 (t is ignored for them)."""
    s4h = np.minimum(m["s4"], m["s4x"])
    out = {"8@0.45": m["w8"] >= 0.45}
    for t4 in (0.40, 0.45):
        out[f"4@{t4:.2f}+8"] = (m["s4"] >= t4) & (m["r8"] >= 0.45)
        out[f"4h2@{t4:.2f}+8"] = (s4h >= t4) & (m["r8"] >= 0.45)
    return out


def stage_report_overlap():
    H = [load2(p) for p in sorted(WORK.glob("emb_head_*.npz"))]
    Ov = [load2(p) for p in sorted(WORK.glob("emb_overlap_*.npz"))]
    HE, HM = np.concatenate([e for e, _ in H]), sum((m for _, m in H), [])
    OE, OM = np.concatenate([e for e, _ in Ov]), sum((m for _, m in Ov), [])
    we, wm = load2(WORK / "emb_win8.npz")
    li = [j for j, m in enumerate(HM) if m["set"] == "live"]
    le, lm = HE[li], [HM[j] for j in li]
    hr = [j for j, m in enumerate(HM) if m["set"] == "ref" and m["kind"] in ("s4", "r8")]
    orf = [j for j, m in enumerate(OM) if m["set"] == "ref"]
    wi = [j for j, m in enumerate(wm) if m["cond"] in ("clean", "opus24k")]
    RE = np.concatenate([HE[hr], OE[orf], we[wi]])
    RM = [HM[j] for j in hr] + [OM[j] for j in orf] + [wm[j] | {"kind": "w8"} for j in wi]
    keys, out, lines = ("spk", "ch", "cond", "L"), {"a": {}, "b": {}}, []
    g = {m["spk"]: m["g"] for m in lm}
    lines.append("## (a) FRR (own, another chapter) / FAR (other same sex)")
    for L in REF_SECS:
        for fc, lc in (("clean", "clean"), ("opus24k", "clean"), ("clean", "laptop_white20"),
                       ("clean", "laptop_white10")):
            refs = sorted({tuple(m[k] for k in keys) for m in RM if m["cond"] == fc and m["L"] == L})
            fmeta = [{"spk": r[0], "ch": r[1], "g": g[r[0]]} for r in refs]
            lj = [j for j, m in enumerate(lm) if m["cond"] == lc]
            gen, imp = masks(fmeta, [lm[j] for j in lj])
            lines.append(f"ref={L}s {fc}/{lc} refs={len(refs)} gen={int(gen.sum())} imp={int(imp.sum())}")
            for name, ok in schemes_x(kind_mins(RE, RM, keys, refs, le[lj])).items():
                r = {"frr": float(1 - ok[gen].mean()), "far": float(ok[imp].mean())}
                out["a"][f"{L}|{fc}|{lc}|{name}"] = r
                lines.append(f"  {name}: FRR={r['frr']:.2%} FAR={r['far']:.3%}")
    lines.append("## (b) splice: o s of the owner, K s of another, the owner to 40 s; live = the owner, clean; pass rate")
    lj = [j for j, m in enumerate(lm) if m["cond"] == "clean"]
    Lv, lmeta = le[lj], [lm[j] for j in lj]
    kk = ("spk", "ch", "imp", "K")
    for K in HEAD_K:
        h0 = [j for j, m in enumerate(HM) if m["set"] == "splice" and m["K"] == K]
        for o in (0,) + OFFS:
            oj = [j for j, m in enumerate(OM) if m["set"] == "splice" and m["K"] == K and m["o"] == o]
            keep = ("w8", "s4", "r8") if o == 0 else ("r8",)
            hj = [j for j in h0 if HM[j]["kind"] in keep or (o and HM[j]["kind"] == "w8" and HM[j]["i"] > 0)]
            E, M = np.concatenate([HE[hj], OE[oj]]), [HM[j] for j in hj] + [OM[j] for j in oj]
            refs = sorted({tuple(m[k] for k in kk) for m in M})
            ok_ = np.array([[lv["spk"] == r[0] and lv["ch"] != r[1] for lv in lmeta] for r in refs])
            m = kind_mins(E, M, kk, refs, Lv)
            r = {name: float(ok[ok_].mean()) for name, ok in schemes_x(m).items()}
            r |= {"n_splices": len(refs), "n_pairs": int(ok_.sum())}
            out["b"][f"{K}|{o}"] = r
            lines.append(f"K={K} o={o} splices={len(refs)} pairs={int(ok_.sum())}: "
                         + "  ".join(f"{k}={v:.2%}" for k, v in r.items() if "@" in k)
                         + "  median s4=" + f"{np.median(m['s4'][ok_]):.3f} s4x={np.median(m['s4x'][ok_]):.3f}")
    json.dump(out, open(WORK / "report_overlap.json", "w"), indent=1, default=float)
    print("\n".join(lines))


def stage_timing_overlap():
    """1 thread: reference 60 s -> trim + 4h2+8 windows (5 x 4 s every 2 s + 8 s after 12 s) vs 4+8, the min."""
    se = make_se(1)
    utts = sorted(glob.glob(str(LIBRI / "dev-clean/84/121123/*.flac")))
    x60, used = concat_from(utts, 0, 60)
    lv = embed(se, speech_edges(live_chunks(utts, used, LIVE3_S)[0]))

    def wins(t, x):
        h = head_windows(t)
        return h["s4"] + (head_x4(t) if x else []) + h["r8"]

    for name, x in (("4+8", False), ("4h2+8", True), ("4+8", False), ("4h2+8", True)):  # twice: load drifts
        fn = lambda x=x: min(embed(se, s) @ lv for s in wins(speech_only(x60), x))  # noqa: E731
        fn()
        ts = []
        for _ in range(10):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        print(f"ref 60 s {name} ({len(wins(speech_only(x60), x))} windows) trim+embed+min: "
              f"median={np.median(ts) * 1000:.0f}ms p90={np.percentile(ts, 90) * 1000:.0f}ms", flush=True)


# ---------- stage 4: the reference's raw start (voice._start) with an absolute floor, the hidden-sound refusal ----------
# start4 / report_start4 / timing_start4 are for the scheme of 1d79286 (_start -> (start, hidden, rest), LEVEL_DB):
# run them with VOICE_PY = that commit's voice.py. Stage 6 below is for the scheme after it (save() writes the floor).
VOICE_PY = Path(os.environ.get("VOICE_PY", Path(__file__).resolve().parents[1] / "peresmeshnik/voice.py"))
VOICE_FNS = ("_frames", "_speech", "_start", "_windows")
FILE4 = ("clean", "opus24k", "white30", "white20")  # white<snr>: white noise at SNR to the speech, over the whole file
VARIANTS = tuple((fl, hd) for fl in (-55.0, -60.0) for hd in (2.0, 3.0, 4.0))  # (floor, margin dB): hidden s
BYPASS_K, BYPASS_DB, BYPASS_SNR = (5, 10), (3, 5, 8), (30, 20)


def voice_ns(fns=VOICE_FNS) -> dict:
    """voice.py's functions `fns` (+ its Report) and all its UPPER_CASE constants, exec'd from the source itself (no
    copy to drift): the bench calls the app's own code; FLOOR_DB can be changed in the namespace for the variants."""
    import ast
    from dataclasses import dataclass, field
    tree = ast.parse(VOICE_PY.read_text(encoding="utf-8"))
    ns = {"np": np, "dataclass": dataclass, "field": field}
    for b in tree.body:
        names = {t.id for t in ast.walk(b) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
        if isinstance(b, ast.FunctionDef) and b.name in fns or isinstance(b, ast.ClassDef) and b.name == "Report" \
                or isinstance(b, ast.Assign) and names and all(n.isupper() for n in names):
            try:
                exec(compile(ast.Module([b], []), str(VOICE_PY), "exec"), ns)
            except NameError:  # a constant built from something not loaded here
                pass
    assert all(k in ns for k in fns if k != "_onset"), sorted(set(fns) - set(ns))  # _onset: since FIXES-s3-40 9
    return ns


def hidden_s(ns, x, floor: float, margin: float) -> float:
    """The refusal measured here and dropped (BENCH; it refused the owner's own file 27-97 %): seconds of the raw start
    above `floor` at LEVEL_DB, louder than the noise of the pauses (p10) by more than `margin` dB and no louder than
    _speech's threshold. Refused at more than 1 s."""
    f = ns["_frames"](x - x.mean(), SR)
    db = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)
    live = db[db > -100]
    if not len(live):
        return 0.0
    gain, noise = ns["LEVEL_DB"] - float(np.percentile(live, 95)), float(np.percentile(live, 10))
    d = db[:int(ns["START_S"] * 1000 / 30)]
    return float(np.sum((d > -100) & (d + gain > floor) & (d > noise + margin) & (d <= max(noise + 6, -60.0)))) * 0.03


def hidden_variants(ns, x) -> dict[str, float]:
    return {f"{-fl:.0f}_{hd:.0f}": hidden_s(ns, x, fl, hd) for fl, hd in VARIANTS}


def with_noise(ns, x, snr, rng):
    """x + white noise whose RMS is `snr` dB below the RMS of x's speech (_speech), and that noise RMS."""
    nr = rms(ns["_speech"](x, SR)) / 10 ** (snr / 20)
    return np.clip(x + rng.standard_normal(len(x)) * nr, -1, 1).astype(np.float32), nr


def gated(ns, x, K):
    """The first K s of speech of x (its own frames above the _speech threshold) with everything else -> digital
    silence, the timeline kept; scaled to speech RMS 1."""
    f = ns["_frames"](x - x.mean(), SR)
    db = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)
    on = db > max(float(np.percentile(db[db > -100], 10)) + 6, -60.0)
    c = int(np.searchsorted(np.cumsum(on), round(K / 0.03)) + 1)
    g = f[:c] * on[:c, None]
    return (g / rms(g[on[:c]])).reshape(-1).astype(np.float32)


def stage_start4():
    """argv[2] = shard "i/n". ref 40 / 60 s, FILE4: new = _windows(_start + rest) (+ hidden for VARIANTS), old =
    4h2+8 of _speech (s4 s4x r8; clean / opus24k: emb_head + emb_overlap). splice (splice_pairs): the first K s of
    another person's speech, pauses digital silence, at d dB above the owner's noise, then the owner's 40 s + white
    noise at SNR (= ref white<SNR> 40 s): new and old. argv[3] = another FLOOR_DB ("-60"): new only, ref 40 s clean /
    white30 and the splices at SNR 30 (where the floor cuts sound: elsewhere all of it is above both floors)."""
    i, n = map(int, sys.argv[2].split("/"))
    floor = sys.argv[3] if len(sys.argv) > 3 else None
    ns, se, gender, rows, info, t0 = voice_ns(), make_se(1), libri_speakers(), [], [], time.time()
    if floor:
        ns["FLOOR_DB"] = float(floor)
    imps = {}
    for spk, ch, imp, ich in splice_pairs():
        imps.setdefault((spk, ch), []).append((imp, ich))

    def add(meta, x, old):
        st, rest = ns["_start"](x, SR), ns["_speech"](x[int(ns["START_S"] * SR):], SR)
        wins = ns["_windows"](st, SR)
        rows.extend((meta | {"scheme": "new", "i": q, "thr": thr}, embed(se, w)) for q, (w, thr) in enumerate(wins))
        info.append(meta | {"hidden": hidden_variants(ns, x), "start_s": (len(st) - len(rest)) / SR, "rest_s": len(rest) / SR,
                            "n_new": len(wins), "raw_s": len(x) / SR})
        if old:
            t = ns["_speech"](x, SR)
            h = head_windows(t)
            ws = [(w, 0.40) for w in h["s4"] + head_x4(t)] + [(w, 0.45) for w in h["r8"]]
            info[-1]["speech_s"] = len(t) / SR
            rows.extend((meta | {"scheme": "old", "i": q, "thr": thr}, embed(se, w)) for q, (w, thr) in enumerate(ws))

    for j, (spk, ch, utts) in enumerate(chapters()):
        if j % n != i:
            continue
        base = {"spk": spk, "ch": ch, "g": gender[spk]}
        for L in REF_SECS if floor is None else (FILE_SEC,):
            fx, _ = concat_from(utts, 0, L)
            for fc in (FILE4 if floor is None else ("clean", "white30")) if fx is not None else ():
                w = fc.startswith("white")
                x = with_noise(ns, fx, int(fc[5:]), np.random.default_rng([int(spk), int(ch), int(fc[5:]), L]))[0] \
                    if w else file_cond(fx, fc)
                add(base | {"set": "ref", "cond": fc, "L": L}, x, w and floor is None)
        fx, _ = concat_from(utts, 0, FILE_SEC)
        for imp, ich in imps.get((spk, ch), []):
            oth, _ = concat_from(sorted(glob.glob(str(LIBRI / "dev-clean" / imp / ich / "*.flac"))), 0, 30)
            for snr in BYPASS_SNR if floor is None else (30,):
                own, nr = with_noise(ns, fx, snr, np.random.default_rng([int(spk), int(ch), snr, FILE_SEC]))
                for K in BYPASS_K:
                    g = gated(ns, oth, K)
                    for d in BYPASS_DB:
                        add(base | {"set": "splice", "imp": imp, "imp_ch": ich, "K": K, "d": d, "snr": snr},
                            np.concatenate([g * nr * 10 ** (d / 20), own]), floor is None)
        print(f"{j} {spk}/{ch} rows={len(rows)} t={time.time() - t0:.0f}s", flush=True)
    np.savez(WORK / f"emb_start4{'f' + floor.lstrip('-') if floor else ''}_{i}.npz", e=np.stack([e for _, e in rows]),
             m=json.dumps([m for m, _ in rows]), info=json.dumps(info))


def min_margin(E, M, keyf, refs, Lv):
    """refs x lives: min over the reference's windows of (cosine - the window's threshold); >= 0 passes."""
    pos = {}
    for r, m in enumerate(M):
        pos.setdefault(keyf(m), []).append(r)
    thr = np.array([m["thr"] for m in M])
    return np.stack([np.min(E[pos[r]] @ Lv.T - thr[pos[r], None], axis=0) for r in refs])


def stage_report_start4():
    """argv[2] = another FLOOR_DB ("-60"): its new windows (emb_start4f60_*) against the old scheme of emb_start4_*."""
    floor = sys.argv[2].lstrip("-") if len(sys.argv) > 2 else ""
    H = [load2(p) for p in sorted(WORK.glob("emb_head_*.npz"))]
    Ov = [load2(p) for p in sorted(WORK.glob("emb_overlap_*.npz"))]
    S4 = [np.load(p) for p in sorted(WORK.glob("emb_start4_*.npz"))]
    HE, HM = np.concatenate([e for e, _ in H]), sum((m for _, m in H), [])
    OE, OM = np.concatenate([e for e, _ in Ov]), sum((m for _, m in Ov), [])
    NE = np.concatenate([z["e"] for z in S4])
    NM = sum((json.loads(str(z["m"])) for z in S4), [])
    info = sum((json.loads(str(z["info"])) for z in S4), [])
    if floor:
        F = [np.load(p) for p in sorted(WORK.glob(f"emb_start4f{floor}_*.npz"))]
        keep = [j for j, m in enumerate(NM) if m["scheme"] == "old"]
        NE = np.concatenate([NE[keep]] + [z["e"] for z in F])
        NM = [NM[j] for j in keep] + sum((json.loads(str(z["m"])) for z in F), [])
        ik = lambda r: tuple(r.get(k) for k in ("set", "spk", "ch", "cond", "L", "imp", "K", "d", "snr"))  # noqa: E731
        start = {ik(r): r["start_s"] for z in F for r in json.loads(str(z["info"]))}
        for r in info:
            r["start_s"] = start.get(ik(r), r["start_s"])
    li = [j for j, m in enumerate(HM) if m["set"] == "live"]
    le, lm = HE[li], [HM[j] for j in li]
    g = {m["spk"]: m["g"] for m in lm}
    thr_of = {"s4": 0.40, "s4x": 0.40, "r8": 0.45}
    old_cached = [(HE[j], HM[j]) for j in range(len(HM)) if HM[j]["set"] == "ref" and HM[j]["kind"] in ("s4", "r8")] \
        + [(OE[j], OM[j]) for j in range(len(OM)) if OM[j]["set"] == "ref"]
    old_cached = [(e, m | {"scheme": "old", "thr": thr_of[m["kind"]]}) for e, m in old_cached]
    rk = ("spk", "ch", "cond", "L")
    sk = ("spk", "ch", "imp", "K", "d", "snr")
    hid = {tuple(r[k] for k in (rk if r["set"] == "ref" else sk)): r for r in info}
    out, lines = {"a": {}, "hidden": {}, "c": {}}, []
    f_ = floor or "55"  # the refusal options: (floor_margin, HIDDEN_S)
    ins_opts = ((f"{f_}_3", 1.0), (f"{f_}_4", 1.0), (f"{f_}_3", 2.0), (f"{f_}_4", 2.0), (f"{f_}_3", 3.0))

    def scheme(name, keyset):
        E = [NE[j] for j, m in enumerate(NM) if m["scheme"] == name and m["set"] == keyset]
        M = [m for m in NM if m["scheme"] == name and m["set"] == keyset]
        if name == "old" and keyset == "ref":
            E, M = E + [e for e, _ in old_cached], M + [m for _, m in old_cached]
        return np.stack(E), M

    def q(a):
        a = np.asarray(a)
        return {"p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}

    lines.append("## hidden s of the start (refs): share > HIDDEN_S 1 s, p50/p95/max; variants floor_margin")
    for L in REF_SECS:
        for fc in FILE4:
            rs = [r for r in info if r["set"] == "ref" and r["cond"] == fc and r["L"] == L]
            d = {v: {"share": float(np.mean([r["hidden"][v] > 1.0 for r in rs]))} | q([r["hidden"][v] for r in rs])
                 | {f"share>{hs}": float(np.mean([r["hidden"][v] > hs for r in rs])) for hs in (1.5, 2.0, 3.0)}
                 for v in rs[0]["hidden"]}
            d["start_s"] = q([r["start_s"] for r in rs])
            out["hidden"][f"{L}|{fc}"] = d
            lines.append(f"ref={L}s {fc} n={len(rs)} start_s p50={d['start_s']['p50']:.1f}: " + "  ".join(
                f"{v} {x['share']:.1%} ({x['p50']:.2f}/{x['p95']:.2f}/{x['max']:.2f})"
                for v, x in d.items() if v != "start_s"))
    (OEn, OMn), (NEn, NMn) = scheme("old", "ref"), scheme("new", "ref")
    lines.append("## (a) FRR (own, another chapter) / FAR (other same sex): old 4h2+8 of _speech; new windows; "
                 "new + refusal at hidden (floor_margin) > s")
    for L in REF_SECS:
        for fc in FILE4:
            for lc in LIVE2_CONDS:
                refs = sorted({tuple(m[k] for k in rk) for m in NMn if m["cond"] == fc and m["L"] == L})
                if not refs:
                    continue
                fmeta = [{"spk": r[0], "ch": r[1], "g": g[r[0]]} for r in refs]
                lj = [j for j, m in enumerate(lm) if m["cond"] == lc]
                gen, imp = masks(fmeta, [lm[j] for j in lj])
                kf = lambda m: tuple(m[k] for k in rk)  # noqa: E731
                old = min_margin(OEn, OMn, kf, refs, le[lj]) >= 0
                new = min_margin(NEn, NMn, kf, refs, le[lj]) >= 0
                r = {"n_gen": int(gen.sum()), "n_imp": int(imp.sum())}
                for name, ok in (("old", old), ("new", new)) + tuple(
                        (f"new+{v}>{hs:g}", new & np.array([hid[ref]["hidden"][v] <= hs for ref in refs])[:, None])
                        for v, hs in ins_opts):
                    r[name] = {"frr": float(1 - ok[gen].mean()), "far": float(ok[imp].mean())}
                out["a"][f"{L}|{fc}|{lc}"] = r
                lines.append(f"ref={L}s {fc}/{lc} gen={r['n_gen']} imp={r['n_imp']}: " + "  ".join(
                    f"{k} {v['frr']:.2%}/{v['far']:.3%}" for k, v in r.items() if isinstance(v, dict)))
    (OEs, OMs), (NEs, NMs) = scheme("old", "splice"), scheme("new", "splice")
    lj = [j for j, m in enumerate(lm) if m["cond"] == "clean"]
    Lv, lmeta = le[lj], [lm[j] for j in lj]
    lines.append("## (c) bypass: K s of another person (pauses digital, d dB above the owner's pause noise) + the owner "
                 "40 s + white noise at SNR; live = the owner, another chapter, clean; pass rate")
    for snr in BYPASS_SNR:
        for K in (0,) + BYPASS_K:
            for d in BYPASS_DB if K else (0,):
                kf = (lambda m: tuple(m[k] for k in sk)) if K else (lambda m: tuple(m[k] for k in rk))  # noqa: E731
                if K:
                    refs = sorted({kf(m) for m in NMs if m["K"] == K and m["d"] == d and m["snr"] == snr})
                    oe, om, ne, nm = OEs, OMs, NEs, NMs
                else:  # the owner's own file with that noise (= the splice without the piece)
                    own = sorted({(m["spk"], m["ch"]) for m in NMs})
                    have = {tuple(m[k] for k in rk) for m in NMn}
                    refs = [r for r in ((s, c, f"white{snr}", FILE_SEC) for s, c in own) if r in have]
                    oe, om, ne, nm = OEn, OMn, NEn, NMn
                if not refs:
                    continue
                ok_ = np.array([[lv["spk"] == r[0] and lv["ch"] != r[1] for lv in lmeta] for r in refs])
                old = min_margin(oe, om, kf, refs, Lv) >= 0
                new = min_margin(ne, nm, kf, refs, Lv) >= 0
                r = {"n_splices": len(refs), "n_pairs": int(ok_.sum()), "old": float(old[ok_].mean()),
                     "new_windows": float(new[ok_].mean())}
                for v, hs in ins_opts:
                    ins = np.array([hid[ref]["hidden"][v] <= hs for ref in refs])
                    r[f"ins_{v}>{hs:g}"] = float(np.broadcast_to(ins[:, None], ok_.shape)[ok_].mean())
                    r[f"new+{v}>{hs:g}"] = float((new & ins[:, None])[ok_].mean())
                r["hidden_55_3"] = q([hid[ref]["hidden"]["55_3"] for ref in refs])
                r["start_s"] = q([hid[ref]["start_s"] for ref in refs])
                r["speech_s"] = q([hid[ref].get("speech_s", 0) for ref in refs])
                out["c"][f"{snr}|{K}|{d}"] = r
                lines.append(f"snr={snr} K={K} d={d} splices={r['n_splices']} pairs={r['n_pairs']}: " + "  ".join(
                    f"{k}={v:.2%}" for k, v in r.items() if isinstance(v, float))
                    + f"  hidden p50/p95/max={r['hidden_55_3']['p50']:.2f}/{r['hidden_55_3']['p95']:.2f}/"
                      f"{r['hidden_55_3']['max']:.2f}  start_s p50={r['start_s']['p50']:.1f}"
                      f"  old speech_s p50={r['speech_s']['p50']:.1f}")
    out["info"] = info  # per reference: hidden s of every variant, start / rest / speech s
    json.dump(out, open(WORK / f"report_start4{'f' + floor if floor else ''}.json", "w"), indent=1, default=float)
    print("\n".join(lines))


def stage_timing_start4():
    """1 thread, reference 60 s: new (_start + _windows, embed, min) vs old (4h2+8 of _speech), and _start alone."""
    ns, se = voice_ns(), make_se(1)
    utts = sorted(glob.glob(str(LIBRI / "dev-clean/84/121123/*.flac")))
    x60, used = concat_from(utts, 0, 60)
    lv = embed(se, speech_edges(live_chunks(utts, used, LIVE3_S)[0]))

    def new():
        return min(embed(se, w) @ lv - t for w, t in ns["_windows"](ns["_start"](x60, SR), SR))

    def old():
        t = ns["_speech"](x60, SR)
        h = head_windows(t)
        return min(embed(se, w) @ lv for w in h["s4"] + head_x4(t) + h["r8"])

    for name, fn in (("_start alone", lambda: ns["_start"](x60, SR)), ("old 4h2+8", old), ("new", new),
                     ("old 4h2+8", old), ("new", new)):  # twice: load drifts
        fn()
        ts = []
        for _ in range(10):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        print(f"ref 60 s {name}: median={np.median(ts) * 1000:.0f}ms p90={np.percentile(ts, 90) * 1000:.0f}ms",
              flush=True)
    st, _, rest = ns["_start"](x60, SR)
    print(f"new: start {len(st) / SR:.1f} s + rest {len(rest) / SR:.1f} s, "
          f"{len(ns['_windows'](np.concatenate([st, rest]), SR))} windows; old: speech "
          f"{len(ns['_speech'](x60, SR)) / SR:.1f} s")


# ---------- stage 6: what save() writes (frames FLOOR_DB under the median speech frame -> digital silence) is checked ----
VOICE_FNS6 = ("_frames", "_speech", "_start", "_windows", "_under_floor", "check", "_band_db")
FLOORS6 = (25.0, 30.0, 35.0)
RAISE_S = (0, 3, 6)  # (c): the first s of my speech +20 dB, then 5 s of another person at -10 dB to my speech, then me 40 s


def saved(ns, x):
    """voice.save()'s numbers: (reference.wav as read back, frames of the start written as silence, the level-set
    recording before that). Its file and json writing left out; PCM_16 through soundfile, as save() writes it."""
    import io
    x = np.asarray(x, np.float32)[:int(ns["MAX_S"] * SR)]
    report = ns["check"](x, SR)
    peak = float(np.max(np.abs(x))) or 1.0
    gain = min(10 ** ((ns["TARGET_DB"] - report.level_db) / 20), 10 ** (-1 / 20) / peak)
    y = np.clip(x * gain, -1, 1)
    saved.onset = ns["_onset"](y, SR) if "_onset" in ns else 0  # FIXES-s3-40 9: the room before the first word cut
    y = y[saved.onset:]
    pre = y.copy()
    under = ns["_under_floor"](y, SR)
    fl = SR * 30 // 1000
    y[:len(under) * fl].reshape(-1, fl)[under] = 0
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return sf.read(buf, dtype="float32")[0], under, pre


def speech_frames(ns, x):
    """check()'s speech frames of x: louder than the pauses' p10 + 6 dB, not under -60."""
    f = ns["_frames"](x - x.mean(), SR)
    db = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)
    live = db[db > -100]
    return db > max(float(np.percentile(live, 10)) + 6, -60.0) if len(live) else db > 0


def stage_save6():
    """argv[2] = shard "i/n", argv[3] = part: "1" = ref 40 s clean / opus24k + the bypass of stage 4 (K s of another
    person, pauses digital, d dB above the owner's noise, then the owner 40 s + white noise at SNR); "2" = ref 40 s
    white30 / white20, ref 60 s clean / white20 + (c) the floor raised (RAISE_S s of mine at +20 dB, 5 s of another
    person at -10 dB, me 40 s; 0 = no loud start). Each at FLOORS6: the file as save() writes it, checked as
    confirm_live does (_windows(_start)). Refs: every window's vector (FRR / FAR in the report); splices: only the pass
    against the owner's clean lives of other chapters (windows in _windows' order, stopped once every live fails)."""
    import hashlib
    i, n = map(int, sys.argv[2].split("/"))
    part = sys.argv[3]
    ns, se, gender, rows, info, t0 = voice_ns(VOICE_FNS6), make_se(1), libri_speakers(), [], [], time.time()
    H = [load2(p) for p in sorted(WORK.glob("emb_head_*.npz"))]
    HE, HM = np.concatenate([e for e, _ in H]), sum((m for _, m in H), [])
    lives = {}
    for e, m in zip(HE, HM):
        if m["set"] == "live" and m["cond"] == "clean":
            lives.setdefault(m["spk"], []).append((m["ch"], e))
    imps, cache = {}, {}
    for spk, ch, imp, ich in splice_pairs():
        imps.setdefault((spk, ch), []).append((imp, ich))

    def emb(w):
        k = hashlib.blake2b(w.tobytes(), digest_size=16).digest()
        if k not in cache:
            cache[k] = embed(se, w)
        return cache[k]

    def ref(meta, x):
        sp = speech_frames(ns, saved(ns, x)[2])
        for fl in FLOORS6:
            ns["FLOOR_DB"] = fl
            y, under, _ = saved(ns, x)
            st = ns["_start"](y, SR)
            wins = ns["_windows"](st, SR)
            rows.extend((meta | {"floor": fl, "i": q, "thr": thr}, emb(w)) for q, (w, thr) in enumerate(wins))
            info.append(meta | {"floor": fl, "zeroed_s": float(under.sum()) * 0.03,
                                "speech_lost_s": float((under & sp[:len(under)]).sum()) * 0.03,
                                "speech30_s": float(sp[:len(under)].sum()) * 0.03, "n_win": len(wins)})

    def splice(meta, x, a, b, Lv):
        """x with another person at samples a:b (frame-aligned)."""
        fl_ = SR * 30 // 1000
        on = np.mean(ns["_frames"](x[a:b], SR) ** 2, axis=1) > 1e-10
        for fl in FLOORS6:
            ns["FLOOR_DB"] = fl
            y, under, _ = saved(ns, x)
            kept = np.mean(ns["_frames"](y[a:b], SR) ** 2, axis=1) > 1e-10
            ok, k = np.ones(len(Lv), bool), 0
            for w, thr in ns["_windows"](ns["_start"](y, SR), SR):
                ok &= emb(w) @ Lv.T >= thr
                k += 1
                if not ok.any():
                    break
            info.append(meta | {"floor": fl, "pass": ok.tolist(), "n_emb": k, "fore_on_s": float(on.sum()) * 0.03,
                                "fore_kept_s": float((kept & on).sum()) * 0.03, "zeroed_s": float(under.sum()) * 0.03})
        assert a % fl_ == 0

    for j, (spk, ch, utts) in enumerate(chapters()):
        if j % n != i:
            continue
        cache.clear()
        base = {"spk": spk, "ch": ch, "g": gender[spk]}
        for L, fc in ((40, "clean"), (40, "opus24k")) if part == "1" else \
                ((40, "white30"), (40, "white20"), (60, "clean"), (60, "white20")):
            fx, _ = concat_from(utts, 0, L)
            if fx is None:
                continue
            x = with_noise(ns, fx, int(fc[5:]), np.random.default_rng([int(spk), int(ch), int(fc[5:]), L]))[0] \
                if fc.startswith("white") else file_cond(fx, fc)
            ref(base | {"set": "ref", "cond": fc, "L": L}, x)
        Lv = np.stack([e for c, e in lives.get(spk, []) if c != ch] or [np.zeros(HE.shape[1], np.float32)])
        for imp, ich in imps.get((spk, ch), []):
            oth, _ = concat_from(sorted(glob.glob(str(LIBRI / "dev-clean" / imp / ich / "*.flac"))), 0, 30)
            sm = base | {"set": "splice", "imp": imp, "imp_ch": ich}
            if part == "1":
                fx, _ = concat_from(utts, 0, FILE_SEC)
                for snr in BYPASS_SNR:
                    own, nr = with_noise(ns, fx, snr, np.random.default_rng([int(spk), int(ch), snr, FILE_SEC]))
                    for K in BYPASS_K:
                        g = gated(ns, oth, K)
                        for d in BYPASS_DB:
                            splice(sm | {"K": K, "d": d, "snr": snr}, np.concatenate([g * nr * 10 ** (d / 20), own]),
                                   0, len(g), Lv)
            elif (fx := concat_from(utts, 0, FILE_SEC + max(RAISE_S))[0]) is not None:  # 3 chapters are shorter
                g = gated(ns, oth, 5) * rms(ns["_speech"](fx, SR)) * 10 ** (-10 / 20)
                for R in RAISE_S:
                    x = np.concatenate([fx[:R * SR] * 10, g, fx[R * SR:(R + FILE_SEC) * SR]])
                    x *= min(1.0, 0.99 / float(np.max(np.abs(x))))  # no clipping: the whole file turned down
                    splice(sm | {"R": R}, x, R * SR, R * SR + len(g), Lv)
        print(f"{j} {spk}/{ch} rows={len(rows)} info={len(info)} t={time.time() - t0:.0f}s", flush=True)
    np.savez(WORK / f"emb_save6_{part}_{i}.npz", e=np.stack([e for _, e in rows]) if rows else np.zeros((0, HE.shape[1])),
             m=json.dumps([m for m, _ in rows]), info=json.dumps(info))


def stage_report_save6():
    H = [load2(p) for p in sorted(WORK.glob("emb_head_*.npz"))]
    HE, HM = np.concatenate([e for e, _ in H]), sum((m for _, m in H), [])
    Z = [np.load(p) for p in sorted(WORK.glob("emb_save6_*.npz"))]
    E = np.concatenate([z["e"] for z in Z if len(z["e"])])
    M = sum((json.loads(str(z["m"])) for z in Z), [])
    info = sum((json.loads(str(z["info"])) for z in Z), [])
    li = [j for j, m in enumerate(HM) if m["set"] == "live"]
    le, lm = HE[li], [HM[j] for j in li]
    g = {m["spk"]: m["g"] for m in lm}
    rk = ("spk", "ch", "cond", "L", "floor")
    out, lines = {"a": {}, "b": {}, "c": {}, "d": {}}, []

    def q(a):
        a = np.asarray(a, float)
        return {"p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())} if len(a) else {}

    lines.append("## (a) FRR (own, another chapter) / FAR (other same sex) of the file as save() writes it, per FLOOR_DB")
    for L, fc in sorted({(m["L"], m["cond"]) for m in M}):
        for lc in LIVE2_CONDS:
            lj = [j for j, m in enumerate(lm) if m["cond"] == lc]
            parts = []
            for fl in FLOORS6:
                refs = sorted({tuple(m[k] for k in rk) for m in M if m["cond"] == fc and m["L"] == L and m["floor"] == fl})
                if not refs:
                    continue
                fmeta = [{"spk": r[0], "ch": r[1], "g": g[r[0]]} for r in refs]
                gen, imp = masks(fmeta, [lm[j] for j in lj])
                ok = min_margin(E, M, lambda m: tuple(m[k] for k in rk), refs, le[lj]) >= 0
                r = {"frr": float(1 - ok[gen].mean()), "far": float(ok[imp].mean()), "n_gen": int(gen.sum()),
                     "n_imp": int(imp.sum())}
                out["a"][f"{L}|{fc}|{lc}|{fl:g}"] = r
                parts.append(f"{fl:g}: {r['frr']:.2%}/{r['far']:.3%}")
            lines.append(f"ref={L}s {fc}/{lc} gen={r['n_gen']} imp={r['n_imp']}: " + "  ".join(parts))
    lines.append("## (d) the start (30 s) written as silence, s: all / my speech (check()'s frames) p50/p95/max")
    for L, fc in sorted({(r["L"], r["cond"]) for r in info if r["set"] == "ref"}):
        for fl in FLOORS6:
            rs = [r for r in info if r["set"] == "ref" and r["cond"] == fc and r["L"] == L and r["floor"] == fl]
            d = {"zeroed": q([r["zeroed_s"] for r in rs]), "speech_lost": q([r["speech_lost_s"] for r in rs]),
                 "speech30": q([r["speech30_s"] for r in rs]), "n": len(rs)}
            out["d"][f"{L}|{fc}|{fl:g}"] = d
            lines.append(f"ref={L}s {fc} floor={fl:g} n={len(rs)}: zeroed {d['zeroed']['p50']:.2f}/{d['zeroed']['p95']:.2f}/"
                         f"{d['zeroed']['max']:.2f}  speech lost {d['speech_lost']['p50']:.2f}/{d['speech_lost']['p95']:.2f}/"
                         f"{d['speech_lost']['max']:.2f} (of speech {d['speech30']['p50']:.1f})")

    def splices(key, sel, title):
        lines.append(title)
        for k in sorted({tuple(r[x] for x in key) for r in info if sel(r)}):
            for fl in FLOORS6:
                rs = [r for r in info if sel(r) and tuple(r[x] for x in key) == k and r["floor"] == fl]
                if not rs:
                    continue
                P = [(p, r["fore_kept_s"]) for r in rs for p in r["pass"]]
                d = {"n_splices": len(rs), "n_pairs": len(P), "pass": float(np.mean([p for p, _ in P])),
                     "silenced": float(np.mean([r["fore_kept_s"] == 0 for r in rs])),
                     "danger": float(np.mean([p and kept > 0 for p, kept in P])),
                     "danger_1s": float(np.mean([p and kept >= 1 for p, kept in P])),
                     "fore_kept_s": q([r["fore_kept_s"] for r in rs]), "fore_on_s": q([r["fore_on_s"] for r in rs]),
                     "n_emb": float(np.mean([r["n_emb"] for r in rs]))}
                out["b" if "K" in key else "c"]["|".join(map(str, k)) + f"|{fl:g}"] = d
                lines.append(" ".join(f"{x}={v}" for x, v in zip(key, k)) + f" floor={fl:g} splices={d['n_splices']} "
                             f"pairs={d['n_pairs']}: pass={d['pass']:.2%} silenced={d['silenced']:.1%} "
                             f"danger={d['danger']:.2%} danger(kept>=1s)={d['danger_1s']:.2%} kept s p50/p95="
                             f"{d['fore_kept_s']['p50']:.2f}/{d['fore_kept_s']['p95']:.2f} of {d['fore_on_s']['p50']:.2f}")

    splices(("snr", "K", "d"), lambda r: r["set"] == "splice" and "K" in r,
            "## (b) bypass of stage 4; pass = the check passes (live: the owner, clean); silenced = the other person all "
            "written as silence; danger = in the file and passed")
    splices(("R",), lambda r: r["set"] == "splice" and "R" in r,
            "## (c) the floor raised: R s of mine at +20 dB, 5 s of another at -10 dB to my speech, me 40 s")
    json.dump(out, open(WORK / "report_save6.json", "w"), indent=1, default=float)
    print("\n".join(lines))


# ---------- stage 8 (before 0.3.0): the head of the recording against its own rest (same session, no synthesis) ----------
HR_T = (0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70)


def head_rest(E, head, rest):
    """min over the head windows of the cosine with the rest's mean vector (unit)."""
    r = E[rest].mean(axis=0)
    return float(np.min(E[head] @ (r / (np.linalg.norm(r) + 1e-12))))


def stage_report_headrest():
    """(a) the owner's file as save() writes it (emb_save6, FLOOR_DB 35): refused before (the windows against the live
    reading) and with "the head (its 4 s windows) like the rest of the recording" at HR_T; another person of the same sex
    passing. (b) splices at the start (emb_head + emb_overlap: K s of another person after o s of the owner): passing
    before (4h2@0.40+8) and with the rule."""
    H = [load2(p) for p in sorted(WORK.glob("emb_head_*.npz"))]
    HE, HM = np.concatenate([e for e, _ in H]), sum((m for _, m in H), [])
    li = [j for j, m in enumerate(HM) if m["set"] == "live"]
    le, lm = HE[li], [HM[j] for j in li]
    g = {m["spk"]: m["g"] for m in lm}
    out, lines = {"a": {}, "b": {}}, []
    Z = [np.load(p) for p in sorted(WORK.glob("emb_save6_*.npz"))]
    E = np.concatenate([z["e"] for z in Z if len(z["e"])])
    M = sum((json.loads(str(z["m"])) for z in Z), [])
    rk = ("spk", "ch", "cond", "L", "floor")
    lines.append("## (a) own refused / another passing (same sex), before -> with head-rest >= T; and head-rest of the "
                 "owner's files p1 / p5 / min")
    for L, fc in sorted({(m["L"], m["cond"]) for m in M}):
        sel = [j for j, m in enumerate(M) if m["L"] == L and m["cond"] == fc and m["floor"] == 35.0]
        Es, Ms = E[sel], [M[j] for j in sel]
        refs = sorted({tuple(m[k] for k in rk) for m in Ms})
        pos = {}
        for r, m in enumerate(Ms):
            pos.setdefault(tuple(m[k] for k in rk), []).append(r)
        hr = np.array([head_rest(Es, [q for q in pos[r] if Ms[q]["thr"] < 0.45], [q for q in pos[r] if Ms[q]["thr"] >= 0.45]
                                 or pos[r]) for r in refs])
        fmeta = [{"spk": r[0], "ch": r[1], "g": g[r[0]]} for r in refs]
        line = f"ref={L}s {fc} refs={len(refs)} head-rest p1/p5/min {np.percentile(hr, 1):.3f}/{np.percentile(hr, 5):.3f}/{hr.min():.3f}:"
        for lc in LIVE2_CONDS:
            lj = [j for j, m in enumerate(lm) if m["cond"] == lc]
            gen, imp = masks(fmeta, [lm[j] for j in lj])
            ok = min_margin(Es, Ms, lambda m: tuple(m[k] for k in rk), refs, le[lj]) >= 0
            r = {"frr": float(1 - ok[gen].mean()), "far": float(ok[imp].mean()), "n_gen": int(gen.sum()),
                 "n_imp": int(imp.sum())} | {f"frr@{t}": float(1 - (ok & (hr >= t)[:, None])[gen].mean()) for t in HR_T} \
                | {f"far@{t}": float((ok & (hr >= t)[:, None])[imp].mean()) for t in HR_T}
            out["a"][f"{L}|{fc}|{lc}"] = r | {"hr_p1": float(np.percentile(hr, 1)), "hr_min": float(hr.min())}
            line += f"\n  live {lc} gen={r['n_gen']}: FRR {r['frr']:.2%} -> " + " ".join(
                f"{t}: {r[f'frr@{t}']:.2%}" for t in HR_T) + f"; FAR {r['far']:.3%} -> {r['far@0.6']:.3%} at 0.6"
        lines.append(line)
    Ov = [load2(p) for p in sorted(WORK.glob("emb_overlap_*.npz"))]
    OE, OM = np.concatenate([e for e, _ in Ov]), sum((m for _, m in Ov), [])
    lines.append("## (b) splices: o s of the owner, K s of another person, the owner on; live = the owner (clean, other "
                 "chapter); pass before -> with head-rest >= T")
    lj = [j for j, m in enumerate(lm) if m["cond"] == "clean"]
    Lv, lmeta = le[lj], [lm[j] for j in lj]
    kk = ("spk", "ch", "imp", "K")
    for K in HEAD_K:
        h0 = [j for j, m in enumerate(HM) if m["set"] == "splice" and m["K"] == K]
        for o in (0,) + OFFS:
            oj = [j for j, m in enumerate(OM) if m["set"] == "splice" and m["K"] == K and m["o"] == o]
            keep = ("w8", "s4", "r8") if o == 0 else ("r8",)
            hj = [j for j in h0 if HM[j]["kind"] in keep or (o and HM[j]["kind"] == "w8" and HM[j]["i"] > 0)]
            Ek, Mk = np.concatenate([HE[hj], OE[oj]]), [HM[j] for j in hj] + [OM[j] for j in oj]
            refs = sorted({tuple(m[k] for k in kk) for m in Mk})
            ok_ = np.array([[lv["spk"] == r[0] and lv["ch"] != r[1] for lv in lmeta] for r in refs])
            before = schemes_x(kind_mins(Ek, Mk, kk, refs, Lv))["4h2@0.40+8"]
            pos = {}
            for q, m in enumerate(Mk):
                pos.setdefault(tuple(m[k] for k in kk), []).append(q)
            hr = np.array([head_rest(Ek, [q for q in pos[r] if Mk[q]["kind"] in ("s4", "s4x")],
                                     [q for q in pos[r] if Mk[q]["kind"] == "r8"]) for r in refs])
            r = {"before": float(before[ok_].mean()), "n_splices": len(refs), "n_pairs": int(ok_.sum()),
                 "hr_p50": float(np.median(hr)), "hr_max": float(hr.max())} | {
                f"after@{t}": float((before & (hr >= t)[:, None])[ok_].mean()) for t in HR_T}
            out["b"][f"{K}|{o}"] = r
            lines.append(f"K={K} o={o} splices={len(refs)} pairs={r['n_pairs']}: pass {r['before']:.2%} -> "
                         + " ".join(f"{t}: {r[f'after@{t}']:.2%}" for t in HR_T)
                         + f"; head-rest p50 {r['hr_p50']:.3f} max {r['hr_max']:.3f}")
    json.dump(out, open(WORK / "report_headrest.json", "w"), indent=1, default=float)
    print("\n".join(lines))


# ---------- stage 8b: the whole rule of the app now: the windows (+ the engines' prompt) and the head like the rest ----------
VOICE_FNS8 = VOICE_FNS6 + ("_prompt", "head_rest", "_onset")


def check8(ns, se, y):
    """(window vectors with thresholds, the old rule's count of them, head_rest) of save()'s file y, as confirm_live: the
    head windows and the engines' prompt against the rest's (none after the start: refused, -1)."""
    head, rest = ns["_windows"](ns["_start"](y, SR), SR, split=True)
    head, rest = [embed(se, w) for w in head], [embed(se, w) for w in rest]
    wins = [(e, ns["HEAD_MATCH"]) for e in head] + [(e, ns["MATCH"]) for e in rest]
    old = len(wins)
    if len(p := ns["_prompt"](y, SR)) >= SR // 2:
        head.append(embed(se, p))
        wins.append((head[-1], ns["HEAD_MATCH"]))
    return wins, old, ns["head_rest"](head, rest) if rest else -1.0
QUIET8 = (-52.0, -57.0, -62.0)  # (q) another person's 10 s, pauses digital, at this p95 dBFS, then the owner


def stage_prompt8():
    """argv[2] = shard "i/n". The file as save() writes it, checked as confirm_live does now (VOICE_PY = voice.py now):
    every window of _windows(_start), the engines' prompt (_prompt) as one more head window, head_rest, the seconds of
    sound in that prompt (refused under PROMPT_MIN_S). refs: as save6's
    (40 s clean / opus24k / white30 / white20, 60 s clean / white20): every vector kept (FRR / FAR in the report);
    splices, only the pass against the owner's clean lives, now and before (the windows alone, review 3): (h) K s of
    another person after o s of the owner, (b) stage 4's way around (K s at d dB over the owner's noise, SNR), (q) their
    10 s at QUIET8 dBFS."""
    i, n = map(int, sys.argv[2].split("/"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from clone_prompt_splice import quiet
    ns, se, gender, rows, info, t0 = voice_ns(VOICE_FNS8), make_se(1), libri_speakers(), [], [], time.time()
    H = [load2(p) for p in sorted(WORK.glob("emb_head_*.npz"))]
    HE, HM = np.concatenate([e for e, _ in H]), sum((m for _, m in H), [])
    lives = {}
    for e, m in zip(HE, HM):
        if m["set"] == "live" and m["cond"] == "clean":
            lives.setdefault(m["spk"], []).append((m["ch"], e))
    imps = {}
    for spk, ch, imp, ich in splice_pairs():
        imps.setdefault((spk, ch), []).append((imp, ich))

    def check(y):
        return check8(ns, se, y)

    def ref(meta, x):
        y = saved(ns, x)[0]
        wins, old, hr = check(y)
        ps = len(ns["_prompt"](y, SR)) / SR
        rows.extend((meta | {"i": q, "thr": float(thr), "prompt": q >= old, "hr": hr, "prompt_s": ps}, e)
                    for q, (e, thr) in enumerate(wins))

    def splice(meta, x, a, b, Lv):
        y, under, _ = saved(ns, x)
        a, b = max(a - saved.onset, 0), max(b - saved.onset, 0)  # the file starts at its first word
        kept = float(np.sum(np.mean(ns["_frames"](y[a:b], SR) ** 2, axis=1) > 1e-10)) * 0.03
        wins, old, hr = check(y)
        E, T = np.stack([e for e, _ in wins]), np.array([t for _, t in wins])
        m = E @ Lv.T - T[:, None]
        info.append(meta | {"before": (m[:old].min(axis=0) >= 0).tolist(), "windows": (m.min(axis=0) >= 0).tolist(),
                            "hr": hr, "kept_s": kept, "prompt_s": len(ns["_prompt"](y, SR)) / SR})

    for j, (spk, ch, utts) in enumerate(chapters()):
        if j % n != i:
            continue
        base = {"spk": spk, "ch": ch, "g": gender[spk]}
        for L, fc in ((40, "clean"), (40, "opus24k"), (40, "white30"), (40, "white20"), (60, "clean"), (60, "white20")):
            fx, _ = concat_from(utts, 0, L)
            if fx is None:
                continue
            x = with_noise(ns, fx, int(fc[5:]), np.random.default_rng([int(spk), int(ch), int(fc[5:]), L]))[0] \
                if fc.startswith("white") else file_cond(fx, fc)
            ref(base | {"set": "ref", "cond": fc, "L": L}, x)
        Lv = np.stack([e for c, e in lives.get(spk, []) if c != ch] or [np.zeros(HE.shape[1], np.float32)])
        for imp, ich in imps.get((spk, ch), []):
            oth, _ = concat_from(sorted(glob.glob(str(LIBRI / "dev-clean" / imp / ich / "*.flac"))), 0, 30)
            sm = base | {"set": "splice", "imp": imp, "imp_ch": ich}
            fx, _ = concat_from(utts, 0, FILE_SEC)
            os_ = speech_only(oth)
            for K in HEAD_K:
                own = concat_from(utts, 0, FILE_SEC - K)[0]
                for o in (0,) + OFFS:
                    c = speech_cut(own, o) if o else 0
                    piece = os_[:K * SR] * (rms(ns["_speech"](own, SR)) / rms(os_[:K * SR]))
                    splice(sm | {"fam": "h", "K": K, "o": o}, np.concatenate([own[:c], piece, own[c:]]), c, c + len(piece), Lv)
            for snr in BYPASS_SNR:
                own, nr = with_noise(ns, fx, snr, np.random.default_rng([int(spk), int(ch), snr, FILE_SEC]))
                for K in BYPASS_K:
                    g = gated(ns, oth, K)
                    for d in BYPASS_DB:
                        splice(sm | {"fam": "b", "K": K, "d": d, "snr": snr},
                               np.concatenate([g * nr * 10 ** (d / 20), own]), 0, len(g), Lv)
            for snr in (30, 20):  # FIXES-s3-40 9: their 5 s in the owner's steady noise, d dB over it, then the owner
                sp = rms(ns["_speech"](fx, SR))
                for d in (3, 5, 8):
                    piece = os_[:5 * SR] * (sp / 10 ** (snr / 20) * 10 ** (d / 20) / rms(os_[:5 * SR]))
                    x = with_noise(ns, np.concatenate([piece, fx]), snr, np.random.default_rng([int(spk), int(ch), snr, d]))[0]
                    splice(sm | {"fam": "n", "K": 5, "d": d, "snr": snr}, x, 0, len(piece), Lv)
            for q in QUIET8:
                x = quiet(oth, fx, q)[:int(10 * SR) // 480 * 480]
                splice(sm | {"fam": "q", "db": q}, np.concatenate([x, fx * (10 ** (-20 / 20) / rms(ns["_speech"](fx, SR)))]),
                       0, len(x), Lv)
        print(f"{j} {spk}/{ch} rows={len(rows)} info={len(info)} t={time.time() - t0:.0f}s", flush=True)
    np.savez(WORK / f"emb_prompt8_{i}.npz", e=np.stack([e for _, e in rows]), m=json.dumps([m for m, _ in rows]),
             info=json.dumps(info))


def stage_report_prompt8():
    H = [load2(p) for p in sorted(WORK.glob("emb_head_*.npz"))]
    HE, HM = np.concatenate([e for e, _ in H]), sum((m for _, m in H), [])
    li = [j for j, m in enumerate(HM) if m["set"] == "live"]
    le, lm = HE[li], [HM[j] for j in li]
    g = {m["spk"]: m["g"] for m in lm}
    Z = [np.load(p) for p in sorted(WORK.glob("emb_prompt8_*.npz"))]
    E = np.concatenate([z["e"] for z in Z])
    M = sum((json.loads(str(z["m"])) for z in Z), [])
    info = sum((json.loads(str(z["info"])) for z in Z), [])
    rk, out, lines, ns = ("spk", "ch", "cond", "L"), {"a": {}, "b": {}}, [], voice_ns(VOICE_FNS8)
    lines.append("## (a) own refused / another passing (same sex): review 3 (the windows) -> the windows + the engines' "
                 "prompt -> + head-rest >= HEAD_REST (each T); head-rest of the owner's files p1 / min")
    for L, fc in sorted({(m["L"], m["cond"]) for m in M}):
        sel = [j for j, m in enumerate(M) if m["L"] == L and m["cond"] == fc]
        Es, Ms = E[sel], [M[j] for j in sel]
        refs = sorted({tuple(m[k] for k in rk) for m in Ms})
        hr = np.array([next(m["hr"] if m["prompt_s"] >= ns["PROMPT_MIN_S"] else -1.0 for m in Ms
                            if tuple(m[k] for k in rk) == r) for r in refs])  # a silent prompt: refused as head-rest
        fmeta = [{"spk": r[0], "ch": r[1], "g": g[r[0]]} for r in refs]
        line = f"ref={L}s {fc} refs={len(refs)} head-rest p1 {np.percentile(hr, 1):.3f} min {hr.min():.3f}:"
        for lc in LIVE2_CONDS:
            lj = [j for j, m in enumerate(lm) if m["cond"] == lc]
            gen, imp = masks(fmeta, [lm[j] for j in lj])
            old = [j for j, m in enumerate(Ms) if not m["prompt"]]
            ok0 = min_margin(Es[old], [Ms[j] for j in old], lambda m: tuple(m[k] for k in rk), refs, le[lj]) >= 0
            ok1 = min_margin(Es, Ms, lambda m: tuple(m[k] for k in rk), refs, le[lj]) >= 0
            r = {"frr0": float(1 - ok0[gen].mean()), "frr1": float(1 - ok1[gen].mean()), "far0": float(ok0[imp].mean()),
                 "n_gen": int(gen.sum())} | {f"frr@{t}": float(1 - (ok1 & (hr >= t)[:, None])[gen].mean()) for t in HR_T} \
                | {f"far@{t}": float((ok1 & (hr >= t)[:, None])[imp].mean()) for t in HR_T}
            out["a"][f"{L}|{fc}|{lc}"] = r
            line += (f"\n  live {lc} gen={r['n_gen']}: FRR {r['frr0']:.2%} -> {r['frr1']:.2%} -> "
                     + " ".join(f"{t}: {r[f'frr@{t}']:.2%}" for t in HR_T)
                     + f"; FAR {r['far0']:.3%} -> {r['far@0.6']:.3%} at 0.6")
        lines.append(line)
    lines.append("## (b) splices, against the owner's clean lives of other chapters: pass (review 3) -> windows + prompt -> "
                 "+ head-rest >= T; danger = some of the other person kept in the file and passed")
    for fam, keys in (("h", ("K", "o")), ("b", ("snr", "K", "d")), ("n", ("snr", "d")), ("q", ("db",))):
        for k in sorted({tuple(r[x] for x in keys) for r in info if r["fam"] == fam}):
            rs = [r for r in info if r["fam"] == fam and tuple(r[x] for x in keys) == k]
            P0 = [p for r in rs for p in r["before"]]
            P1 = [p for r in rs for p in r["windows"]]
            d = {"n": len(rs), "pairs": len(P0), "before": float(np.mean(P0)), "windows": float(np.mean(P1)),
                 "danger_before": float(np.mean([p and r["kept_s"] > 0 for r in rs for p in r["before"]]))} | {
                f"after@{t}": float(np.mean([p and r["hr"] >= t and r["prompt_s"] >= ns["PROMPT_MIN_S"]
                                             for r in rs for p in r["windows"]])) for t in HR_T} | {
                f"danger@{t}": float(np.mean([p and r["hr"] >= t and r["prompt_s"] >= ns["PROMPT_MIN_S"] and r["kept_s"] > 0
                                              for r in rs for p in r["windows"]])) for t in HR_T}
            out["b"][fam + "|" + "|".join(map(str, k))] = d
            lines.append(f"{fam} " + " ".join(f"{x}={v}" for x, v in zip(keys, k)) + f" splices={d['n']} pairs={d['pairs']}: "
                         f"{d['before']:.2%} (danger {d['danger_before']:.2%}) -> {d['windows']:.2%} -> "
                         + " ".join(f"{t}: {d[f'after@{t}']:.2%}/{d[f'danger@{t}']:.2%}" for t in HR_T))
    json.dump(out, open(WORK / "report_prompt8.json", "w"), indent=1, default=float)
    print("\n".join(lines))


ESC50 = ROOT / "data/esc50/ESC-50-master"  # ESC-50 (CC BY-NC 3.0): 5 s clips, 40 a class, 44.1 kHz
LEAD9_CONDS = ("clean", "white30", "white20")  # the owner's file as it is / with a laptop's noise at SNR 30 / 20 dB
LEAD9_EVENTS = ("coughing", "laughing", "sneezing", "um")  # at 2-6 s of the owner's speech, at its level


def esc50(category: str) -> list[np.ndarray]:
    """ESC-50's clips of `category` at SR, only their sound (frames within 30 dB of the clip's loudest), RMS 1."""
    out = []
    for r in csv.DictReader(open(ESC50 / "meta/esc50.csv")):
        if r["category"] != category:
            continue
        x, sr = sf.read(ESC50 / "audio" / r["filename"], dtype="float64", always_2d=True)
        x = x.mean(axis=1)
        y = np.fft.irfft(np.fft.rfft(x)[:int(len(x) * SR / sr) // 2 + 1], int(len(x) * SR / sr)) * SR / sr  # band-limited
        f = y[:len(y) // 480 * 480].reshape(-1, 480)
        db = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)
        on = db > db.max() - 30
        i, j = np.flatnonzero(on)[[0, -1]]
        a = f[i:j + 1].reshape(-1)
        out.append((a / rms(f[on].reshape(-1))).astype(np.float32))
    return out


def um(own: np.ndarray, rng, sec: float = 0.6) -> np.ndarray:
    """An "um" in the owner's voice, roughly: 120 ms of their loudest voiced speech held for `sec` (overlapped Hann
    pieces), faded in and out; RMS 1."""
    f = own[:len(own) // 480 * 480].reshape(-1, 480)
    k = int(np.argsort(np.mean(f ** 2, axis=1))[-1 - int(rng.integers(0, 20))])
    piece = own[max(k * 480 - 960, 0):][:1920] * np.hanning(1920)
    out = np.zeros(int(sec * SR) + 1920)
    for at in range(0, int(sec * SR), 960):
        out[at:at + 1920] += piece
    out = out[:int(sec * SR)] * np.minimum(1, np.minimum(np.arange(int(sec * SR)), np.arange(int(sec * SR))[::-1]) / 1600)
    return (out / rms(out)).astype(np.float32)


def room(ns, fx, sec: float, rng) -> np.ndarray:
    """`sec` s of the owner's own room: steady noise at the level of their file's pauses (its frames under _speech's
    threshold). Those frames shuffled instead held bits of words above it, where save()'s cut stops: not a room."""
    f = ns["_frames"](fx - fx.mean(), SR)
    on = speech_frames(ns, fx)
    q = f[~on] if (~on).sum() >= 10 else f[np.argsort(np.mean(f ** 2, axis=1))[:max(10, len(f) // 10)]]
    return (rms(q.reshape(-1)) * rng.standard_normal(int(sec * SR))).astype(np.float32)


def stage_lead9():
    """argv[2] = shard "i/n". FIXES-s3-40 9: the owner's own file (40 s; clean, or a laptop's white noise in all of it at
    SNR 30 / 20) with a live start, checked as confirm_live does now (check8): the base; (r) 1 / 2 / 3 s of their own
    room before the first word; (br) a breath there (ESC-50 'breathing', 1 s at most, at -15 dB to the speech, after
    0.5 s of room); (ev) a cough / laugh / sneeze (ESC-50, 1.5 s at most) or an "um" (the owner's own voice held 0.6 s)
    at 2-6 s of the speech, at the speech's level. Every window kept (the owner's refusals against their lives, as
    report_prompt8), with head_rest and the seconds of the prompt."""
    i, n = map(int, sys.argv[2].split("/"))
    ns, se, gender, rows, t0 = voice_ns(VOICE_FNS8), make_se(1), libri_speakers(), [], time.time()
    clips = {c: esc50(c) for c in ("breathing", *LEAD9_EVENTS[:3])}
    for j, (spk, ch, utts) in enumerate(chapters()):
        if j % n != i:
            continue
        fx, _ = concat_from(utts, 0, FILE_SEC)
        if fx is None:
            continue
        rng = np.random.default_rng([int(spk), int(ch), 9])
        sp = rms(ns["_speech"](fx, SR))
        var = {"base": fx}
        for L in (1, 2, 3):
            var[f"room{L}"] = np.concatenate([room(ns, fx, L, rng), fx])
        b = clips["breathing"][rng.integers(len(clips["breathing"]))][:SR] * sp * 10 ** (-15 / 20)
        var["breath"] = np.concatenate([room(ns, fx, 0.5, rng), b, room(ns, fx, 0.3, rng), fx])
        for ev in LEAD9_EVENTS:
            e = (um(fx, rng) if ev == "um" else clips[ev][rng.integers(len(clips[ev]))][:int(1.5 * SR)]) * sp
            c = speech_cut(fx, float(rng.uniform(2, 6)))
            var[ev] = np.concatenate([fx[:c], e, fx[c:]])
        for fc in LEAD9_CONDS:
            for v, x in var.items():
                if fc != "clean":
                    x = with_noise(ns, x, int(fc[5:]), np.random.default_rng([int(spk), int(ch), int(fc[5:]), len(x)]))[0]
                y = saved(ns, x)[0]
                wins, old, hr = check8(ns, se, y)
                ps = len(ns["_prompt"](y, SR)) / SR
                meta = {"spk": spk, "ch": ch, "g": gender[spk], "cond": fc, "var": v, "hr": hr, "prompt_s": ps}
                rows.extend((meta | {"i": q, "thr": float(thr)}, e) for q, (e, thr) in enumerate(wins))
        print(f"{j} {spk}/{ch} rows={len(rows)} t={time.time() - t0:.0f}s", flush=True)
    save2(WORK / f"emb_lead9_{i}.npz", rows)


def stage_report_lead9():
    """The owner's refusals (every file x their lives of other chapters, LIVE2_CONDS) with a live start against the same
    files without one: refused by the windows, by head_rest (< HEAD_REST) or a silent prompt (< PROMPT_MIN_S), in all."""
    H = [load2(p) for p in sorted(WORK.glob("emb_head_*.npz"))]
    HE, HM = np.concatenate([e for e, _ in H]), sum((m for _, m in H), [])
    li = [j for j, m in enumerate(HM) if m["set"] == "live"]
    le, lm = HE[li], [HM[j] for j in li]
    Z = [load2(p) for p in sorted(WORK.glob("emb_lead9_*.npz"))]
    E, M = np.concatenate([e for e, _ in Z]), sum((m for _, m in Z), [])
    ns, rk, out, lines = voice_ns(VOICE_FNS8), ("spk", "ch"), {}, []
    for fc in LEAD9_CONDS:
        for v in dict.fromkeys(m["var"] for m in M):
            sel = [j for j, m in enumerate(M) if m["cond"] == fc and m["var"] == v]
            Es, Ms = E[sel], [M[j] for j in sel]
            refs = sorted({tuple(m[k] for k in rk) for m in Ms})
            first = {tuple(m[k] for k in rk): m for m in reversed(Ms)}
            hr = np.array([first[r]["hr"] for r in refs])
            ps = np.array([first[r]["prompt_s"] for r in refs])
            fmeta = [{"spk": r[0], "ch": r[1], "g": first[r]["g"]} for r in refs]
            for lc in LIVE2_CONDS:
                lj = [j for j, m in enumerate(lm) if m["cond"] == lc]
                gen, _ = masks(fmeta, [lm[j] for j in lj])
                ok_w = min_margin(Es, Ms, lambda m: tuple(m[k] for k in rk), refs, le[lj]) >= 0
                ok_h = (hr >= ns["HEAD_REST"])[:, None] & np.ones_like(ok_w)
                ok_p = (ps >= ns["PROMPT_MIN_S"])[:, None] & np.ones_like(ok_w)
                r = {"n_files": len(refs), "n_gen": int(gen.sum()), "frr": float(1 - (ok_w & ok_h & ok_p)[gen].mean()),
                     "frr_windows": float(1 - ok_w[gen].mean()), "frr_headrest": float(1 - ok_h[gen].mean()),
                     "frr_prompt": float(1 - ok_p[gen].mean()), "hr_p1": float(np.percentile(hr, 1)),
                     "hr_min": float(hr.min()), "prompt_min": float(ps.min())}
                out[f"{fc}|{v}|{lc}"] = r
                lines.append(f"{fc:8s} {v:9s} live {lc:15s} files={r['n_files']} pairs={r['n_gen']}: refused "
                             f"{r['frr']:.2%} (windows {r['frr_windows']:.2%}, head-rest {r['frr_headrest']:.2%}, prompt "
                             f"{r['frr_prompt']:.2%}); head-rest p1 {r['hr_p1']:.3f} min {r['hr_min']:.3f}; prompt min "
                             f"{r['prompt_min']:.1f} s")
    json.dump(out, open(WORK / "report_lead9.json", "w"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    {"embed": stage_embed, "report": stage_report, "timing": stage_timing, "padded": stage_padded,
     "report_padded": stage_report_padded, "wlive": stage_wlive, "wfile": stage_wfile, "splice": stage_splice,
     "report2": stage_report2, "timing2": stage_timing2, "trimcheck": stage_trimcheck,
     "trim3": stage_trim3, "report_trim3": stage_report_trim3, "head": stage_head, "report_head": stage_report_head,
     "timing_head": stage_timing_head, "overlap": stage_overlap, "report_overlap": stage_report_overlap,
     "timing_overlap": stage_timing_overlap, "start4": stage_start4, "report_start4": stage_report_start4,
     "timing_start4": stage_timing_start4, "save6": stage_save6,
     "report_save6": stage_report_save6, "report_headrest": stage_report_headrest,
     "prompt8": stage_prompt8, "report_prompt8": stage_report_prompt8, "lead9": stage_lead9,
     "report_lead9": stage_report_lead9}[sys.argv[1]]()
