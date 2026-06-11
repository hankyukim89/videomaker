"""Audio/video pipeline: speaker splitting, Edge TTS with word timestamps,
audio concat, scene-to-audio alignment, SRT, ffmpeg render."""
import asyncio
import json
import math
import os
import random
import re
import shutil
import subprocess
import wave

FFMPEG = shutil.which("ffmpeg")
if not FFMPEG:
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG = "ffmpeg"

GAP_SEC = 0.30  # silence between speaker segments


def run_ff(args, cwd=None):
    p = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error"] + args,
                       capture_output=True, text=True, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-1200:]}")


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ---------- script splitting ----------
SPK_RE = re.compile(r"\[\s*(?:speaker|host|voice)?\s*(\d+)\s*\]", re.I)


def split_script(script):
    """Split on [Speaker N] markers -> [{'speaker': '1', 'text': ...}]."""
    parts = []
    matches = list(SPK_RE.finditer(script))
    if not matches:
        t = script.strip()
        return [{"speaker": "1", "text": t}] if t else []
    if matches[0].start() > 0:
        head = script[:matches[0].start()].strip()
        if head:
            parts.append({"speaker": "1", "text": head})
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(script)
        t = script[m.end():end].strip()
        if t:
            parts.append({"speaker": m.group(1), "text": t})
    return parts


def strip_markers(script):
    return SPK_RE.sub(" ", script)


TAG_ANY_RE = re.compile(r"\[[^\]]*\]")


def strip_tags(script):
    """Remove ALL bracketed tags: [IMG ...], [SFX ...], [Speaker N], emotion tags."""
    return TAG_ANY_RE.sub(" ", script)


# ---------- Edge TTS ----------
async def _edge_one(text, voice, rate, pitch, volume, out_path):
    import edge_tts
    kw = {}
    rp = round((float(rate) - 1) * 100)
    kw["rate"] = f"{'+' if rp >= 0 else ''}{rp}%"
    pp = int(pitch)
    kw["pitch"] = f"{'+' if pp >= 0 else ''}{pp}Hz"
    vp = int(volume) - 100
    kw["volume"] = f"{'+' if vp >= 0 else ''}{vp}%"
    com = edge_tts.Communicate(text, voice, **kw)
    words = []
    with open(out_path, "wb") as f:
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append({"word": chunk["text"],
                              "start": chunk["offset"] / 1e7,
                              "end": (chunk["offset"] + chunk["duration"]) / 1e7})
    return words


def edge_tts_segment(text, voice, controls, out_path):
    return asyncio.run(_edge_one(text, voice,
                                 controls.get("rate", 1.0), controls.get("pitch", 0),
                                 controls.get("volume", 100), out_path))


async def edge_list_voices():
    import edge_tts
    vs = await edge_tts.list_voices()
    return [{"id": v["ShortName"], "label": f"{v['ShortName']} ({v['Gender']})", "locale": v["Locale"]}
            for v in vs]


# ---------- normalization + concat ----------

def to_wav24(src, dst, tempo=1.0):
    af = ["aresample=24000"]
    t = float(tempo)
    if abs(t - 1.0) > 1e-3:
        while t > 2.0:
            af.append("atempo=2.0"); t /= 2.0
        while t < 0.5:
            af.append("atempo=0.5"); t /= 0.5
        af.append(f"atempo={t:.4f}")
    run_ff(["-i", src, "-ac", "1", "-ar", "24000", "-af", ",".join(af), dst])


def make_silence(path, sec=GAP_SEC):
    run_ff(["-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono", "-t", str(sec), path])


def concat_wavs(paths, dst, workdir):
    lst = os.path.join(workdir, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"file '{os.path.abspath(p)}'\n".replace("\\", "/"))
    run_ff(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", dst])


def wav_to_mp3(src, dst):
    run_ff(["-i", src, "-codec:a", "libmp3lame", "-q:a", "3", dst])


# ---------- word timestamp estimation (no-timestamp engines) ----------

def estimate_words(text, start, duration):
    toks = text.split()
    if not toks:
        return []
    weights = [max(len(t), 2) + 1.5 for t in toks]
    total = sum(weights)
    out, t = [], start
    for tok, w in zip(toks, weights):
        d = duration * w / total
        out.append({"word": tok, "start": round(t, 3), "end": round(t + d, 3)})
        t += d
    return out


# ---------- scene <-> word alignment ----------

def _norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def map_scenes_to_words(scene_texts, words, total_duration):
    """Sequential fuzzy match of scene texts onto the global word-timestamp list.
    Returns [(start, end)] per scene. Falls back to proportional allocation."""
    wtoks = [_norm(w["word"]) for w in words]
    n = len(words)
    spans, ptr, ok = [], 0, True
    for st in scene_texts:
        stoks = [t for t in (_norm(x) for x in strip_tags(st).split()) if t]
        if not stoks:
            # empty narration (e.g. two visual tags back to back): zero-width marker here
            spans.append((min(ptr, n - 1), min(ptr, n - 1) - 1))
            continue
        if ptr >= n:
            ok = False
            break
        begin = ptr
        matched = 0
        for tok in stoks:
            j = ptr
            hit = -1
            while j < min(ptr + 6, n):
                if wtoks[j] == tok:
                    hit = j
                    break
                j += 1
            if hit >= 0:
                ptr = hit + 1
                matched += 1
            else:
                ptr = min(ptr + 1, n)
        if matched < max(1, len(stoks) * 0.4):
            ok = False
            break
        spans.append((begin, min(ptr, n) - 1))
    if ok and len(spans) == len(scene_texts):
        out = []
        for i, (a, b) in enumerate(spans):
            start = 0.0 if i == 0 else words[a]["start"]
            if b < a:  # zero-width marker: brief hold at this position
                end = start
            else:
                end = total_duration if i == len(spans) - 1 else words[min(b + 1, n - 1)]["start"]
            out.append((round(start, 3), round(max(end, start + 0.5), 3)))
        for i in range(1, len(out)):
            out[i] = (out[i - 1][1], out[i][1])
        return out
    # fallback: proportional by character count
    lens = [max(len(strip_markers(t)), 1) for t in scene_texts]
    total = sum(lens)
    out, t = [], 0.0
    for L in lens:
        d = total_duration * L / total
        out.append((round(t, 3), round(t + d, 3)))
        t += d
    if out:
        out[-1] = (out[-1][0], round(total_duration, 3))
    return out


# ---------- subtitles ----------

def _ts(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(words, max_chars=42, max_dur=4.5):
    cues, cur, start = [], [], None
    for w in words:
        if start is None:
            start = w["start"]
        cur.append(w)
        text = " ".join(x["word"] for x in cur)
        endpunct = re.search(r"[.!?,;:]$", w["word"].strip())
        if len(text) >= max_chars or (w["end"] - start) >= max_dur or (endpunct and len(text) > 18):
            cues.append((start, w["end"], text))
            cur, start = [], None
    if cur:
        cues.append((start, cur[-1]["end"], " ".join(x["word"] for x in cur)))
    out = []
    for i, (a, b, t) in enumerate(cues, 1):
        out.append(f"{i}\n{_ts(a)} --> {_ts(b)}\n{t}\n")
    return "\n".join(out)


_SENT_RE = re.compile(r"[^.!?…]+[.!?…]*[\"')\]]*\s*")


def split_sentences(text):
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def build_srt_sentences(script, words, total_duration, max_chars=90):
    """Subtitle cues = full sentences from the script (split at commas only when
    a sentence is very long), timed by aligning to the word timestamps."""
    sents = split_sentences(strip_tags(script))
    chunks = []
    for s in sents:
        if len(s) <= max_chars:
            chunks.append(s)
            continue
        parts = re.split(r"(?<=[,;:])\s+", s)
        cur = ""
        for p in parts:
            if cur and len(cur) + len(p) + 1 > max_chars:
                chunks.append(cur)
                cur = p
            else:
                cur = (cur + " " + p).strip()
        if cur:
            chunks.append(cur)
    if not chunks or not words:
        return ""
    spans = map_scenes_to_words(chunks, words, total_duration)
    out = []
    for i, ((a, b), t) in enumerate(zip(spans, chunks), 1):
        out.append(f"{i}\n{_ts(a)} --> {_ts(b)}\n{t}\n")
    return "\n".join(out)


# ---------- render ----------
KB_MOVES = ["in_center", "out_center", "in_tl", "in_br", "pan_lr", "pan_rl"]


def make_placeholder(path, W, H):
    run_ff(["-f", "lavfi", "-i", f"color=c=0x14181d:s={W}x{H}", "-frames:v", "1", path])


def scale_filter(scale_mode, W, H, fps):
    """Plain -vf chain for non-Ken-Burns clips. Returns (vf, is_complex)."""
    tail = f"fps={fps},format=yuv420p"
    if scale_mode == "fit_black":
        return (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,{tail}", False)
    if scale_mode == "fit_blur":
        return (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},boxblur=24:2[bg];"
                f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=({W}-w)/2:({H}-h)/2,{tail}", True)
    if scale_mode == "stretch":
        return (f"scale={W}:{H},{tail}", False)
    if scale_mode == "center":
        return (f"scale=w='trunc(iw*min(1,min({W}/iw,{H}/ih))/2)*2':"
                f"h='trunc(ih*min(1,min({W}/iw,{H}/ih))/2)*2',"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,{tail}", False)
    # fill (default): cover + crop
    return (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},{tail}", False)


def _kb_filter(dur, fps, W, H, move, zmax=1.12):
    frames = max(int(round(dur * fps)), 1)
    zspd = (zmax - 1.0) / frames
    big_w, big_h = W * 2, H * 2
    base = f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,crop={big_w}:{big_h}"
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    if move == "in_center":
        z, x, y = f"min(zoom+{zspd:.6f},{zmax})", cx, cy
    elif move == "out_center":
        z, x, y = f"max({zmax}-{zspd:.6f}*on,1.0)", cx, cy
    elif move == "in_tl":
        z, x, y = f"min(zoom+{zspd:.6f},{zmax})", "0", "0"
    elif move == "in_br":
        z, x, y = f"min(zoom+{zspd:.6f},{zmax})", "iw-(iw/zoom)", "ih-(ih/zoom)"
    elif move == "pan_lr":
        z, x, y = f"{zmax}", f"(iw-(iw/zoom))*on/{frames}", cy
    else:  # pan_rl
        z, x, y = f"{zmax}", f"(iw-(iw/zoom))*(1-on/{frames})", cy
    return f"{base},zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={fps},format=yuv420p"


def _static_filter(W, H, fps):
    return f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={fps},format=yuv420p"


def render_video(workdir, scenes, audio_wav, resolution, motion, subtitles_mode,
                 srt_text, out_path, status, fps=30, fade=0.5, sub_size="medium",
                 scale_mode="fill", music=None, sfx=None):
    """scenes: [{image, duration, motion?}]. music: {path, volume, duck} or None.
    sfx: [{path, at, volume}] one-shot sound effects mixed at exact times.
    status: dict updated in place."""
    W, H = (int(x) for x in resolution.split("x"))
    clipdir = os.path.join(workdir, "clips")
    os.makedirs(clipdir, exist_ok=True)
    use_fade = motion in ("fade", "kenburns_fade")
    pad = fade if use_fade else 0.0
    rng = random.Random(42)
    clips = []
    n = len(scenes)
    for i, sc in enumerate(scenes):
        status.update(stage=f"Rendering scene {i + 1}/{n}", progress=int(65 * i / max(n, 1)))
        dur = max(float(sc["duration"]), 0.6) + (pad if i < n - 1 else pad / 2 + 0.2)
        sc_motion = sc.get("motion") or motion
        clip = os.path.join(clipdir, f"clip{i:03d}.mp4")
        enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-an", clip]
        if sc_motion in ("kenburns", "kenburns_fade") and scale_mode == "fill":
            vf = _kb_filter(dur, fps, W, H, rng.choice(KB_MOVES))
            run_ff(["-loop", "1", "-i", sc["image"], "-t", f"{dur:.3f}", "-vf", vf] + enc)
        else:
            vf, is_complex = scale_filter(scale_mode, W, H, fps)
            flag = "-filter_complex" if is_complex else "-vf"
            run_ff(["-loop", "1", "-i", sc["image"], "-t", f"{dur:.3f}", flag, vf] + enc)
        clips.append(clip)

    status.update(stage="Joining scenes", progress=72)
    silent = os.path.join(workdir, "video_silent.mp4")
    if use_fade and n > 1:
        inputs = []
        for c in clips:
            inputs += ["-i", c]
        fc, prev = [], "[0:v]"
        t = 0.0
        for i in range(1, n):
            t += float(scenes[i - 1]["duration"])
            off = max(t - fade / 2, 0.01)
            outl = f"[vx{i}]"
            fc.append(f"{prev}[{i}:v]xfade=transition=fade:duration={fade}:offset={off:.3f}{outl}")
            prev = outl
        total = sum(float(s["duration"]) for s in scenes)
        run_ff(inputs + ["-filter_complex", ";".join(fc), "-map", prev,
                         "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "veryfast",
                         "-crf", "19", "-pix_fmt", "yuv420p", silent])
    else:
        lst = os.path.join(workdir, "clips.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for c in clips:
                f.write(f"file '{os.path.abspath(c)}'\n".replace("\\", "/"))
        run_ff(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent])

    status.update(stage="Adding audio" + (" + subtitles" if subtitles_mode == "embedded" else ""), progress=85)
    args = ["-i", silent, "-i", audio_wav]
    vfilters = []
    if subtitles_mode == "embedded" and srt_text:
        srt_path = os.path.join(workdir, "subs.srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
        fs = {"small": 14, "medium": 18, "large": 24}.get(sub_size, 18)
        if H > W:
            fs = int(fs * 1.4)
        style = f"FontName=Arial,FontSize={fs},Outline=2,Shadow=0,MarginV=40,Bold=1"
        vfilters.append(f"subtitles=subs.srt:force_style='{style}'")
    if vfilters:
        args += ["-vf", ",".join(vfilters), "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "19", "-pix_fmt", "yuv420p"]
    else:
        args += ["-c:v", "copy"]
    sfx = [s for s in (sfx or []) if s.get("path") and os.path.exists(s["path"])]
    more_steps = bool(sfx) or bool(music and music.get("path"))
    voiced = "voiced.mp4" if more_steps else os.path.basename(out_path)
    args += ["-c:a", "aac", "-b:a", "192k", "-shortest", voiced]
    run_ff(args, cwd=workdir)

    if sfx:
        status.update(stage=f"Mixing {len(sfx)} sound effects", progress=90)
        cur = voiced
        # mix in batches to keep command lines sane
        for bi in range(0, len(sfx), 24):
            batch = sfx[bi:bi + 24]
            nxt = f"sfxmix{bi}.mp4" if (bi + 24 < len(sfx) or (music and music.get("path"))) else os.path.basename(out_path)
            if nxt == cur:
                nxt = "sfxmix_final.mp4"
            ins = ["-i", cur]
            for s in batch:
                ins += ["-i", s["path"]]
            fc, labels = [], []
            for k, s in enumerate(batch, start=1):
                ms = max(int(float(s["at"]) * 1000), 0)
                fc.append(f"[{k}:a]volume={float(s.get('volume', 0.5)):.3f},"
                          f"adelay={ms}|{ms}[sf{k}]")
                labels.append(f"[sf{k}]")
            fc.append(f"[0:a]{''.join(labels)}amix=inputs={len(batch) + 1}:duration=first:"
                      f"dropout_transition=0:normalize=0[a]")
            run_ff(ins + ["-filter_complex", ";".join(fc), "-map", "0:v", "-map", "[a]",
                          "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", nxt], cwd=workdir)
            cur = nxt
        voiced = cur

    if music and music.get("path"):
        status.update(stage="Mixing background music", progress=93)
        total = sum(float(s["duration"]) for s in scenes)
        vol = float(music.get("volume", 0.25))
        if music.get("duck", True):
            fc = (f"[1:a]volume={vol}[bg];"
                  f"[bg][0:a]sidechaincompress=threshold=0.03:ratio=10:attack=25:release=500[duckbg];"
                  f"[0:a][duckbg]amix=inputs=2:duration=first:dropout_transition=0.5,"
                  f"volume=2,afade=t=out:st={max(total - 2, 0):.2f}:d=2[a]")
        else:
            fc = (f"[1:a]volume={vol}[bg];[0:a][bg]amix=inputs=2:duration=first:"
                  f"dropout_transition=0.5,volume=2,afade=t=out:st={max(total - 2, 0):.2f}:d=2[a]")
        run_ff(["-i", voiced, "-stream_loop", "-1", "-i", music["path"],
                "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                os.path.basename(out_path)], cwd=workdir)

    final = os.path.join(workdir, os.path.basename(out_path))
    if os.path.abspath(final) != os.path.abspath(out_path):
        shutil.move(final, out_path)
    status.update(stage="Done", progress=100, file=os.path.basename(out_path))
