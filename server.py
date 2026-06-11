"""AI Video Maker - local server. Run: python server.py  ->  http://127.0.0.1:8765"""
import io
import json
import os
import re
import shutil
import threading
import time
import zipfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import media
import providers
import registry

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECTS = os.path.join(ROOT, "projects")
PROFILES = os.path.join(ROOT, "profiles")
SFX_DIR = os.path.join(ROOT, "sfx")
VOICES_DIR = os.path.join(ROOT, "voices")  # reference clips for Chatterbox voice cloning
SETTINGS_FILE = os.path.join(ROOT, "settings.json")
os.makedirs(PROJECTS, exist_ok=True)
os.makedirs(PROFILES, exist_ok=True)
os.makedirs(SFX_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)

app = FastAPI(title="AI Video Maker")
RENDER_STATUS = {}


@app.exception_handler(providers.ProviderError)
def provider_error_handler(request, exc):
    """Surface provider errors as readable messages instead of a generic 500."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------- settings ----------------
def load_settings():
    s = json.loads(json.dumps(registry.DEFAULT_SETTINGS))
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                loaded = json.load(f)
            for k, v in loaded.items():
                if k == "keys":
                    s["keys"].update(v)
                else:
                    s[k] = v
        except Exception:
            pass
    return s


def save_settings(s):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)


def cad(usd):
    return usd * float(load_settings().get("usd_cad", 1.40))


def model_price(mid, field, default):
    ov = load_settings().get("price_overrides", {}).get(mid, {})
    return float(ov.get(field, default))


def get_key(provider):
    return load_settings().get("keys", {}).get(provider, "")


def safe_name(name):
    n = re.sub(r"[^\w\- ]", "", name).strip()
    if not n:
        raise HTTPException(400, "No project selected — create or open a project first (top-left).")
    return n


def pdir(project):
    d = os.path.join(PROJECTS, safe_name(project))
    if not os.path.isdir(d):
        raise HTTPException(404, f"Project '{project}' not found")
    return d


def enriched_models():
    s = load_settings()
    fx = float(s.get("usd_cad", 1.40))
    ov = s.get("price_overrides", {})

    def enrich(m, kind):
        m = dict(m)
        o = ov.get(m["id"], {})
        for f in ("price_in", "price_out", "price_chars", "price_image"):
            if f in m:
                m[f] = float(o.get(f, m[f]))
                m[f + "_cad"] = round(m[f] * fx, 4)
        m["kind"] = kind
        return m

    return {"text": [enrich(m, "text") for m in registry.TEXT_MODELS],
            "tts": [enrich(m, "tts") for m in registry.TTS_MODELS],
            "image": [enrich(m, "image") for m in registry.IMAGE_MODELS],
            "usd_cad": fx}


def text_cost(model_id, meta, tin, tout):
    usd = (tin * model_price(model_id, "price_in", meta.get("price_in", 0)) +
           tout * model_price(model_id, "price_out", meta.get("price_out", 0))) / 1e6
    return round(cad(usd), 5)


# ---------------- bootstrap / settings ----------------
@app.get("/")
def index():
    with open(os.path.join(ROOT, "static", "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/bootstrap")
def bootstrap():
    profs = json.loads(json.dumps(registry.DEFAULT_PROFILES))
    for fn in sorted(os.listdir(PROFILES)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(PROFILES, fn), encoding="utf-8") as f:
                    profs[fn[:-5]] = json.load(f)
            except Exception:
                pass
    return {"settings": load_settings(), "models": enriched_models(),
            "script_format_default": SCRIPT_FORMAT,
            "profiles": profs, "default_profiles": list(registry.DEFAULT_PROFILES.keys()),
            "projects": sorted(d for d in os.listdir(PROJECTS)
                               if os.path.isdir(os.path.join(PROJECTS, d)))}


class SettingsIn(BaseModel):
    settings: dict


@app.put("/api/settings")
def put_settings(body: SettingsIn):
    save_settings(body.settings)
    return {"ok": True, "models": enriched_models()}


class ProviderIn(BaseModel):
    provider: str


@app.post("/api/models/live")
def models_live(body: ProviderIn):
    return {"models": providers.list_models_live(body.provider, get_key(body.provider))}


# ---------------- profiles ----------------
class ProfileIn(BaseModel):
    name: str
    data: dict


@app.post("/api/profiles")
def save_profile(body: ProfileIn):
    with open(os.path.join(PROFILES, safe_name(body.name) + ".json"), "w", encoding="utf-8") as f:
        json.dump(body.data, f, indent=2)
    return {"ok": True}


@app.delete("/api/profiles/{name}")
def del_profile(name: str):
    p = os.path.join(PROFILES, safe_name(name) + ".json")
    if os.path.exists(p):
        os.remove(p)
    return {"ok": True}


# ---------------- projects ----------------
class NameIn(BaseModel):
    name: str


@app.post("/api/project/new")
def project_new(body: NameIn):
    d = os.path.join(PROJECTS, safe_name(body.name))
    if os.path.exists(d):
        raise HTTPException(400, "A project with that name already exists")
    for sub in ("", "audio", "images", "exports", "work"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    state = {"name": safe_name(body.name), "created": time.strftime("%Y-%m-%d %H:%M"), "costs": []}
    with open(os.path.join(d, "project.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return {"ok": True, "state": state}


@app.get("/api/project/{name}")
def project_get(name: str):
    with open(os.path.join(pdir(name), "project.json"), encoding="utf-8") as f:
        return {"state": json.load(f)}


class StateIn(BaseModel):
    state: dict


@app.post("/api/project/{name}/save")
def project_save(name: str, body: StateIn):
    d = pdir(name)
    body.state["name"] = safe_name(name)
    with open(os.path.join(d, "project.json"), "w", encoding="utf-8") as f:
        json.dump(body.state, f, indent=2)
    if body.state.get("script", {}).get("text"):
        with open(os.path.join(d, "script.txt"), "w", encoding="utf-8") as f:
            f.write(body.state["script"]["text"])
    return {"ok": True}


class SaveAsIn(BaseModel):
    new_name: str
    state: dict


@app.post("/api/project/{name}/saveas")
def project_saveas(name: str, body: SaveAsIn):
    src = pdir(name)
    dst = os.path.join(PROJECTS, safe_name(body.new_name))
    if os.path.exists(dst):
        raise HTTPException(400, "A project with that name already exists")
    shutil.copytree(src, dst)
    body.state["name"] = safe_name(body.new_name)

    def fix(o):
        if isinstance(o, str):
            return o.replace(f"/files/{safe_name(name)}/", f"/files/{safe_name(body.new_name)}/")
        if isinstance(o, list):
            return [fix(x) for x in o]
        if isinstance(o, dict):
            return {k: fix(v) for k, v in o.items()}
        return o

    state = fix(body.state)
    with open(os.path.join(dst, "project.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return {"ok": True, "state": state}


@app.delete("/api/project/{name}")
def project_delete(name: str):
    shutil.rmtree(pdir(name))
    return {"ok": True}


# ---------------- script ----------------
SCRIPT_FORMAT = """OUTPUT FORMAT — MANDATORY, this overrides any conflicting instruction above.
Write the script as spoken narration with inline visual tags. Example of the style — note how tag placement varies: sometimes at the start of a sentence, sometimes mid-sentence right before the word it illustrates (great for lists and reveals):

[IMG 1: Stick man looking at a bear in a zoo.]
Right now, somewhere on Earth, a grown adult is looking at
[IMG 1 EDIT: Zoom in toward the bear, which stares back blankly.]
a bear and
[IMG 1 EDIT 2: Close-up of the man, finger on his chin, pondering.] [SFX: pop]
thinking,
[IMG 1 EDIT 3: Same close-up, now with a thought bubble showing the bear being petted.]
"I wonder if I could pet that."
[IMG 2: Plain beige background.]
People try to pet
[IMG 2 EDIT: A deer pops into the left side.] [SFX: whoosh_fast]
deer,
[IMG 2 EDIT 2: A fox appears beside the deer, which remains.]
foxes,
[IMG 2 EDIT 3: A raccoon joins; the deer and fox remain.]
raccoons,
[IMG 3: A porcupine and a blowfish; a stick man reaches toward them.]
and occasionally animals that look specifically designed by evolution
[IMG 3 EDIT: A big warning symbol pops up between the man and the animals.] [SFX: error_beep]
to discourage touching.

TAG RULES:
- [IMG n: ...] = a brand-new scene number n. Describe the full frame: subjects, action, composition, camera angle. NO art-style words (no "photo", "illustration", "cinematic") — style is applied separately.
- [IMG n EDIT: ...] = an edit applied to the ORIGINAL image n. Describe ONLY the change.
- [IMG n EDIT 2: ...] = edits the RESULT of the previous edit — a chain: EDIT 3 edits the result of EDIT 2, and so on. Use numbered chains when changes accumulate (objects piling up, a story progressing); use a plain unnumbered EDIT for a fresh variation of the original.
- STRUCTURE — CRITICAL: change the visual roughly every 5-12 spoken words. Place each tag exactly where the visual should change — sentence start or mid-sentence, whatever fits the moment. After 3-5 edits of one scene, cut to a NEW [IMG] scene. A typical minute of video has 3-6 distinct scenes. NEVER tell the whole story inside one scene's edit chain.
- [SFX: name] = a sound effect at that exact moment — reveals, punchlines, things appearing (about every 3rd-5th visual). These are the user's actual sound files; use ONLY these names, NEVER invent your own: {sfx}
- Everything outside [brackets] is read aloud word for word: no headings, no markdown, no notes, no stage directions.
- Multiple speakers ONLY if the instructions above ask for them: mark every line with [Speaker 1], [Speaker 2], and put ONE [CAST: 1 = Name (gender), 2 = Name (gender)] tag on the very first line so the app knows who is who (e.g. [CAST: 1 = Maya (female), 2 = Jin (male)]). Never repeat the CAST tag; it is never read aloud.
- Follow the target length given above; if none is given, write roughly 700-900 spoken words (about 5 minutes)."""


class ScriptIn(BaseModel):
    provider: str
    model: str
    topic: str
    preprompt: str = ""
    format_prompt: str = ""   # editable tag-format prompt; blank = built-in default
    controls: dict = {}
    speakers: int = 1
    length_hint: str = ""
    personas: dict = {}


@app.post("/api/script")
def gen_script(body: ScriptIn):
    meta = next((m for m in registry.TEXT_MODELS if m["id"] == body.model), {})
    sys_parts = [body.preprompt.strip()] if body.preprompt.strip() else []
    if body.speakers > 1:
        sys_parts.append(
            f"The script has {body.speakers} speakers. Mark EVERY line with [Speaker 1] .. [Speaker {body.speakers}]. "
            "Output nothing except the marked dialogue.")
        gen = (body.personas or {}).get("general", "").strip()
        if gen:
            sys_parts.append("Show concept: " + gen)
        for sid, p in ((body.personas or {}).get("speakers", {}) or {}).items():
            bits = []
            if p.get("name"):
                bits.append(f"name: {p['name']}")
            if p.get("gender"):
                bits.append(f"gender: {p['gender']}")
            if p.get("style"):
                bits.append(f"personality & speech style: {p['style']}")
            if p.get("example"):
                bits.append(f'example of how they talk: "{p["example"]}"')
            if bits:
                sys_parts.append(f"[Speaker {sid}] — " + "; ".join(bits))
    else:
        sys_parts.append("Output ONLY the narration text to be read aloud. No headings, no markdown, no notes.")
    if body.length_hint:
        sys_parts.append(f"Target length: {body.length_hint}.")
    fmt = body.format_prompt.strip() or SCRIPT_FORMAT
    avail = available_sfx()
    sfx_text = ", ".join(avail) if avail else \
        "NONE — the sound folder is empty, do NOT write any [SFX] tags"
    sys_parts.append(fmt.replace("{sfx}", sfx_text))
    system = "\n\n".join(sys_parts)
    text, tin, tout = providers.generate_text(
        body.provider, get_key(body.provider), body.model, system,
        f"Topic: {body.topic}", body.controls, meta.get("reasoning", False))
    return {"text": text.strip(), "tokens_in": tin, "tokens_out": tout,
            "cost_cad": text_cost(body.model, meta, tin, tout)}


# ---------------- tagged-script parser (free, no AI call) ----------------
ANY_TAG_RE = re.compile(
    r"\[\s*(?:(IMG|IMAGE|PIC|PICTURE)\s*(\d+)?(\s*EDIT\s*\d*)?|(SFX))\s*:\s*([^\]]+)\]", re.I)
CAST_RE = re.compile(r"\[\s*CAST\s*:[^\]]*\]", re.I)  # [CAST: 1 = Maya (female), ...] — metadata, never read aloud


def parse_tagged_script(script):
    """Parse a script with inline [IMG n: ...] / [IMG n EDIT: ...] / [SFX: name] tags.
    Returns beats in the same shape the AI director produces, or None if no tags."""
    if not ANY_TAG_RE.search(script):
        return None
    beats, chains, lead = [], {}, ""
    last = 0
    events = []
    for m in ANY_TAG_RE.finditer(script):
        events.append(("text", script[last:m.start()]))
        if m.group(4):  # SFX
            events.append(("sfx", m.group(5).strip()))
        else:
            edit_part = m.group(3)
            if edit_part:
                knum = re.search(r"\d+", edit_part)
                edit_k = int(knum.group()) if knum else 0   # 0 = unnumbered EDIT
            else:
                edit_k = None                                # not an edit: new scene
            events.append(("img", (m.group(2) or "1").strip(), edit_k, m.group(5).strip()))
        last = m.end()
    events.append(("text", script[last:]))
    for ev in events:
        if ev[0] == "text":
            if beats:
                beats[-1]["text"] += " " + ev[1]
            else:
                lead += ev[1]
        elif ev[0] == "sfx":
            name = re.sub(r"\.(mp3|wav|ogg|flac|m4a)$", "", ev[1].strip(), flags=re.I)
            name = re.sub(r"[^\w\-]", "_", name)
            if beats and not beats[-1]["sfx"]:
                beats[-1]["sfx"] = name
        else:
            _, n, edit_k, prompt = ev
            chain = chains.get(n)  # [original, edit1 result, edit2 result, ...]
            if edit_k is None or (not chain and not beats):
                shot, eo = "new", -1
            elif not chain:
                shot, eo = "edit", len(beats) - 1          # EDIT of an unknown number: edit previous beat
            elif edit_k <= 1:
                shot, eo = "edit", chain[0]                # plain EDIT / EDIT 1: edit the ORIGINAL
            else:
                shot, eo = "edit", chain[min(edit_k - 1, len(chain) - 1)]  # EDIT k: edits result of step k-1
            beats.append({"text": "", "image_prompt": prompt, "shot": shot,
                          "edit_of": eo, "sfx": "", "energy": "calm"})
            if shot == "new":
                chains[n] = [len(beats) - 1]
            else:
                chains.setdefault(n, []).append(len(beats) - 1)
    if not beats:
        return None
    if lead.strip():
        beats[0]["text"] = (lead + " " + beats[0]["text"])
    for b in beats:
        b["text"] = re.sub(r"\s+", " ", media.strip_tags(b["text"])).strip()
    return beats


def auto_segments(narration, max_chars=300):
    """Sentence-grouped voice segments from plain narration (speaker markers respected)."""
    segs = []
    for part in media.split_script(narration):
        cur = ""
        for s in media.split_sentences(part["text"]):
            if cur and len(cur) + len(s) + 1 > max_chars:
                segs.append({"speaker": part["speaker"], "text": cur, "direction": ""})
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            segs.append({"speaker": part["speaker"], "text": cur, "direction": ""})
    return segs


# ---------------- director (two-layer segmentation) ----------------
DIRECTOR_SYS = """You are a film director preparing a narration script for an AI-made video. Produce a JSON object with these keys:

"segments": the VOICE layer. Split the script into voice segments at natural pause points (speaker turns, paragraph breaks). Each: {{"speaker":"1","text":"...","direction":"..."}}. "text" = EXACT contiguous chunk of the script, in order, covering the whole script, no overlaps (normalize whitespace only; do NOT include [Speaker N] markers in text). Segments are typically 2-6 sentences. "direction" = short acting note ("hushed, tense").

"beats": the IMAGE layer, independent of segments. Split the SAME script into visual beats: {{"text":"...","shot":"new|edit","edit_of":<beat index>,"image_prompt":"...","sfx":"","energy":"calm|build|peak"}}. Same exact-coverage rule. {pacing_note} You MAY split mid-sentence at a dramatic word for emphasis. Avoid equal-length beats.

"shot" — THE KEY CREATIVE TOOL. Two kinds of beat:
- "new": a brand-new scene. "image_prompt" describes the full frame.
- "edit": MODIFY the image of an earlier beat instead of drawing a new scene. "edit_of" = index (0-based) of the beat whose image gets edited; "image_prompt" = ONLY the change ("change his expression to crying", "add two loudspeakers beside the bed", "zoom in close on her hands", "add a speech bubble with a repeat symbol above the phone"). Edits are PERFECT when narration adds a detail, emotion or twist to the SAME scene — they make the video feel alive and continuous. Use them often: a typical video should be roughly 40-60% edit beats, in runs (new scene → 2-4 successive edits building on each other → new scene). An edit may chain off another edit, but never more than 4 edits deep from the original "new" beat; then start a fresh "new" scene. "edit_of" must always point to an EARLIER beat.

"sfx": optional sound effect played exactly when this beat's image appears — use for punchlines, reveals, twists and scene changes (roughly every 3rd-5th beat, not all of them). Pick ONLY from this list (else ""): {sfx_list}

"image_prompt" rules for "new" shots — CRITICAL: describe only WHAT is in the frame — the exact people, creatures, objects, places, actions and numbers that chunk of narration mentions, plus composition/camera angle, so a viewer hearing those words sees exactly that. Do NOT mention any art style, medium, photography, film, realism, lighting aesthetic, or color grading — the user applies the style separately (style pre-prompt or reference images). Re-use the bible's recurring subjects by repeating their physical descriptions (image generators have no memory between images). NEVER generic stock scenes ("a person thinking") — always concrete and specific to the sentence. Never text, captions, or logos in the image (speech bubbles and symbols added via edits are fine).

"bible": a continuity sheet — 2-4 sentences precisely describing the recurring subjects only (each person/creature/place: build, colors, clothing, distinguishing features). NO art-style description — content only.

"title": a strong video title.
"thumbnail_prompt": one image prompt for a clickable thumbnail (no text in image).
"chapters": [{{"title":"...","beat":<index of first beat of chapter>}}] — 3-8 YouTube chapters.

Return ONLY the JSON object."""

PACING_NOTES = {
    "frequent": ("CUT FAST: the picture should change roughly every {min_sec}-{mid_sec} seconds "
                 "(~{min_w}-{mid_w} words per beat) almost everywhere, like a fast-paced video essay. "
                 "Only rare calm holds up to {max_sec}s (~{max_w} words)."),
    "normal": ("VARY pacing deliberately: long holds ({max_sec}s, ~{max_w} words) during calm explanation, "
               "quick cuts ({min_sec}s, ~{min_w} words) at dramatic peaks."),
}


class DirectorIn(BaseModel):
    project: str
    provider: str
    model: str
    script: str
    topic: str = ""
    min_sec: float = 3
    max_sec: float = 12
    pacing: str = "normal"  # normal | frequent


def _as_text(v):
    """LLMs sometimes return objects where we asked for strings — coerce safely."""
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def parse_json_block(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise HTTPException(500, "Director did not return JSON — try again or another model.")
    raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
        except json.JSONDecodeError as e:
            raise HTTPException(500, f"Bad JSON from director: {e}")


def validate_beat_plan(beats):
    """Enforce edit rules: edit_of points to an earlier beat, chains max 4 deep,
    sfx names must be in the library. Invalid edits become 'new' shots."""
    depth = {}
    for i, b in enumerate(beats):
        shot = (b.get("shot") or "new").lower()
        try:
            eo = int(b.get("edit_of", -1))
        except (TypeError, ValueError):
            eo = -1
        if shot == "edit" and 0 <= eo < i and depth.get(eo, 0) < 4:
            b["shot"] = "edit"
            b["edit_of"] = eo
            depth[i] = depth.get(eo, 0) + 1
        else:
            b["shot"] = "new"
            b["edit_of"] = -1
            depth[i] = 0
        sfx = _as_text(b.get("sfx", "")).strip().removesuffix(".mp3").removesuffix(".wav")
        b["sfx"] = sfx if sfx in registry.SFX_NAMES else ""
    return beats


# camera-move edit prompts used when force-splitting long beats in "frequent" pacing
CAMERA_EDITS = [
    "Zoom in slightly closer on the main subject; keep everything else exactly identical.",
    "Zoom in tighter, close framing on the focal point; same scene, same style.",
    "Shift the camera angle slightly to one side; same scene, subjects and style.",
]


def _chunk_text(text, chunk_w):
    """Split text into ~chunk_w-word chunks, preferring sentence boundaries."""
    sents = [s.strip() for s in re.findall(r"[^.!?…]+[.!?…]*", text) if s.strip()] or [text]
    chunks, cur, curw = [], "", 0
    for s in sents:
        w = len(s.split())
        if w > chunk_w * 1.7:  # giant sentence: hard-split by words
            if cur:
                chunks.append(cur)
                cur, curw = "", 0
            toks = s.split()
            for j in range(0, len(toks), chunk_w):
                chunks.append(" ".join(toks[j:j + chunk_w]))
            continue
        if cur and curw + w > chunk_w:
            chunks.append(cur)
            cur, curw = s, w
        else:
            cur = (cur + " " + s).strip()
            curw += w
    if cur:
        chunks.append(cur)
    return chunks or [text]


def split_long_beats(beats, limit_w, chunk_w):
    """Frequent pacing enforcement: any beat longer than limit_w words is split into
    ~chunk_w-word chunks. Continuation chunks become EDIT shots of the first chunk
    (subtle camera moves), so the picture still changes every few seconds.
    Returns (new_beats, remap old_index -> new_index)."""
    out, remap = [], {}
    for i, b in enumerate(beats):
        remap[i] = len(out)
        words = b["text"].split()
        chunks = _chunk_text(b["text"], chunk_w) if len(words) > limit_w else [b["text"]]
        fb = dict(b)
        fb["text"] = chunks[0]
        out.append(fb)
        first_new = len(out) - 1
        for j, ch in enumerate(chunks[1:]):
            out.append({"text": ch, "shot": "edit", "edit_of": first_new, "_cont": True,
                        "image_prompt": CAMERA_EDITS[j % len(CAMERA_EDITS)],
                        "sfx": "", "energy": b.get("energy", "calm")})
    for nb in out:  # remap original edit references to the new index space
        if nb.pop("_cont", False):
            continue
        if nb.get("shot") == "edit":
            nb["edit_of"] = remap.get(nb.get("edit_of"), nb.get("edit_of"))
    return out, remap


@app.post("/api/director")
def director(body: DirectorIn):
    script = CAST_RE.sub(" ", body.script)  # cast tag is metadata — never narrated
    # Tagged script? Parse it directly — instant, free, and exactly what the user wrote.
    parsed = parse_tagged_script(script)
    if parsed is not None:
        segs = auto_segments(ANY_TAG_RE.sub(" ", script))
        if not segs:
            raise HTTPException(400, "No narration text found outside the [tags].")
        return {"segments": segs, "beats": parsed, "bible": "", "title": "",
                "thumbnail_prompt": "", "chapters": [], "cost_cad": 0, "parsed": True}
    meta = next((m for m in registry.TEXT_MODELS if m["id"] == body.model), {})
    mn, mx = int(body.min_sec), int(body.max_sec)
    mid = max(mn + 2, (mn + mx) // 2)
    pacing_note = PACING_NOTES.get(body.pacing, PACING_NOTES["normal"]).format(
        min_sec=mn, mid_sec=mid, max_sec=mx,
        min_w=int(mn * 2.6), mid_w=int(mid * 2.6), max_w=int(mx * 2.6))
    system = DIRECTOR_SYS.format(pacing_note=pacing_note,
                                 sfx_list=", ".join(registry.SFX_NAMES))
    user = (f"Topic of the video: {body.topic}\n\n" if body.topic else "") + f"Script:\n\n{script}"
    text, tin, tout = providers.generate_text(
        body.provider, get_key(body.provider), body.model, system,
        user, {"temperature": 0.5, "max_tokens": 32768},
        meta.get("reasoning", False))
    d = parse_json_block(text)
    segs = [s for s in d.get("segments", []) if s.get("text", "").strip()]
    beats = [b for b in d.get("beats", []) if b.get("text", "").strip()]
    if not segs or not beats:
        raise HTTPException(500, "Director returned empty segments or beats — try again.")
    for s in segs:
        s["speaker"] = _as_text(s.get("speaker", "1")) or "1"
        s["direction"] = _as_text(s.get("direction", ""))
        s["text"] = media.strip_markers(_as_text(s["text"])).strip()
    for b in beats:
        b["image_prompt"] = _as_text(b.get("image_prompt", ""))
        b["energy"] = _as_text(b.get("energy", "calm")) or "calm"
        b["text"] = media.strip_markers(_as_text(b["text"])).strip()
    beats = validate_beat_plan(beats)
    remap = None
    if body.pacing == "frequent":
        chunk_w = max(int(mn * 2.6) + 2, 8)       # ~3s of speech per chunk
        limit_w = int(chunk_w * 1.5)              # split anything noticeably longer
        beats, remap = split_long_beats(beats, limit_w, chunk_w)
    chapters = []
    for c in (d.get("chapters") or []):
        if isinstance(c, dict):
            try:
                bi = int(c.get("beat", 0))
                if remap:
                    bi = remap.get(bi, bi)
                chapters.append({"title": _as_text(c.get("title", "")), "beat": bi})
            except (ValueError, TypeError):
                pass
    return {"segments": segs, "beats": beats,
            "bible": _as_text(d.get("bible", "")), "title": _as_text(d.get("title", "")),
            "thumbnail_prompt": _as_text(d.get("thumbnail_prompt", "")),
            "chapters": chapters,
            "cost_cad": text_cost(body.model, meta, tin, tout)}


# ---------------- voices / tts ----------------
def chatterbox_voice_list():
    out = [{"id": "default", "label": "Default voice (built-in)"}]
    for fn in sorted(os.listdir(VOICES_DIR)):
        if os.path.splitext(fn)[1].lower() in (".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a"):
            out.append({"id": fn, "label": f"Clone of: {os.path.splitext(fn)[0]}",
                        "url": f"/voices/{fn}?v={int(time.time())}"})
    return out


@app.get("/api/voices")
def voices(engine: str):
    try:
        if engine == "edge":
            import asyncio
            return {"voices": asyncio.run(media.edge_list_voices())}
        if engine == "azure":
            s = load_settings()
            return {"voices": providers.voices_azure(get_key("azure"), s.get("azure_region", "eastus"))}
        if engine in ("chatterbox", "replicate_tts"):
            return {"voices": chatterbox_voice_list()}
        if engine == "elevenlabs":
            key = get_key("elevenlabs")
            if not key:
                raise HTTPException(400, "Add your ElevenLabs API key in Settings first.")
            return {"voices": providers.voices_elevenlabs(key)}
    except providers.ProviderError as e:
        raise HTTPException(400, str(e))
    return {"voices": []}


@app.post("/api/voices/clone")
async def voice_clone_upload(file: UploadFile = File(...)):
    """Save a reference clip for Chatterbox voice cloning (5-20s of clean speech is ideal)."""
    name = re.sub(r"[^\w.\-]", "_", file.filename or "voice.wav")
    with open(os.path.join(VOICES_DIR, name), "wb") as f:
        f.write(await file.read())
    return {"voices": chatterbox_voice_list(), "added": name}


def apply_direction(engine, controls, text, direction):
    """Translate a voice direction into engine-specific form. Returns (text, controls)."""
    controls = dict(controls)
    direction = (direction or "").strip()
    if not direction:
        return text, controls
    if engine in ("eleven_v3", "chatterbox", "chatterbox-replicate"):
        # both support inline expression tags like [laugh], [sigh], [whispering]
        tags = "".join(re.findall(r"\[[^\]]+\]", direction))
        plain = re.sub(r"\[[^\]]+\]", "", direction).strip(" ,")
        prefix = tags if tags else (f"[{plain}] " if plain else "")
        return (prefix + " " + text).strip(), controls
    if engine == "azure":
        # use the direction as the speaking style if none set (works on style-capable voices)
        if not (controls.get("style") or "").strip():
            controls["style"] = direction
        return text, controls
    return text, controls  # edge, multilingual_v2 etc: direction ignored


def _tts_one(engine_model, provider, voice, text, controls, out_base,
             prev_text=None, next_text=None):
    """Generate one segment -> (wav24_path, words_or_None). out_base has no extension."""
    if provider == "edge":
        mp3 = out_base + ".mp3"
        words = media.edge_tts_segment(text, voice, controls, mp3)
        wav = out_base + ".wav"
        media.to_wav24(mp3, wav)
        return wav, words
    if provider == "azure":
        s = load_settings()
        raw, words = providers.tts_azure(get_key("azure"), s.get("azure_region", "eastus"),
                                         voice, text, controls)
        src = out_base + "_raw.wav"
        with open(src, "wb") as f:
            f.write(raw)
        wav = out_base + ".wav"
        media.to_wav24(src, wav)
        return wav, words
    if provider == "chatterbox":
        clone = voice and voice != "default"
        vpath = os.path.join(VOICES_DIR, os.path.basename(voice)) if clone else ""
        if clone and not os.path.exists(vpath):
            raise HTTPException(400, f"Voice clone file '{voice}' not found in the voices folder.")
        variant = ("original" if engine_model == "chatterbox-v1"
                   else "mtl" if engine_model == "chatterbox-mtl" else "turbo")
        raw = providers.tts_chatterbox(vpath, text, controls, variant)
        src = out_base + "_raw.wav"
        with open(src, "wb") as f:
            f.write(raw)
        wav = out_base + ".wav"
        media.to_wav24(src, wav)
        return wav, None
    if provider == "replicate_tts":
        clone = voice and voice != "default"
        vpath = os.path.join(VOICES_DIR, os.path.basename(voice)) if clone else ""
        if clone and not os.path.exists(vpath):
            raise HTTPException(400, f"Voice clone file '{voice}' not found in the voices folder.")
        path = ("resemble-ai/chatterbox" if engine_model == "chatterbox-v1-replicate"
                else "resemble-ai/chatterbox-turbo")
        raw = providers.tts_chatterbox_replicate(get_key("replicate"), vpath, text, controls, path)
        src = out_base + "_raw"
        with open(src, "wb") as f:
            f.write(raw)
        wav = out_base + ".wav"
        media.to_wav24(src, wav)
        return wav, None
    if provider == "elevenlabs":
        raw, words = providers.tts_elevenlabs(get_key("elevenlabs"), engine_model, voice, text,
                                              controls, prev_text, next_text)
        mp3 = out_base + ".mp3"
        with open(mp3, "wb") as f:
            f.write(raw)
        wav = out_base + ".wav"
        media.to_wav24(mp3, wav)
        return wav, words
    raise HTTPException(400, f"Unknown TTS provider {provider}")


class SegTTSIn(BaseModel):
    project: str
    seg_index: int
    engine: str
    voice: str
    text: str
    direction: str = ""
    controls: dict = {}
    prev_text: str = ""
    next_text: str = ""


@app.post("/api/tts/segment")
def tts_segment(body: SegTTSIn):
    d = pdir(body.project)
    adir = os.path.join(d, "audio")
    os.makedirs(adir, exist_ok=True)
    meta = next((m for m in registry.TTS_MODELS if m["id"] == body.engine), {})
    provider = meta.get("provider", "edge")
    text, controls = apply_direction(body.engine, body.controls, body.text, body.direction)
    stem = os.path.join(adir, f"seg{body.seg_index:03d}")
    k = 1
    while os.path.exists(f"{stem}_v{k}.wav"):
        k += 1
    base = f"{stem}_v{k}"
    wav, words = _tts_one(body.engine, provider, body.voice, text, controls, base,
                          body.prev_text or None, body.next_text or None)
    dur = media.wav_duration(wav)
    if not words:
        words = media.estimate_words(body.text, 0.0, dur)
    chars = len(body.text)
    usd = chars * model_price(body.engine, "price_chars", meta.get("price_chars", 0)) / 1e6
    return {"file": os.path.basename(wav), "duration": round(dur, 3), "words": words,
            "url": f"/files/{body.project}/audio/{os.path.basename(wav)}?v={int(time.time())}",
            "version": k, "chars": chars, "cost_cad": round(cad(usd), 5)}


class StitchIn(BaseModel):
    project: str
    files: list


@app.post("/api/tts/stitch")
def tts_stitch(body: StitchIn):
    d = pdir(body.project)
    adir = os.path.join(d, "audio")
    paths = []
    for fn in body.files:
        p = os.path.join(adir, os.path.basename(fn))
        if not os.path.exists(p):
            raise HTTPException(400, f"Missing segment audio {fn} — regenerate it.")
        paths.append(p)
    silence = os.path.join(adir, "_gap.wav")
    media.make_silence(silence)
    seq = []
    for i, p in enumerate(paths):
        seq.append(p)
        if i < len(paths) - 1:
            seq.append(silence)
    mix = os.path.join(adir, "mix.wav")
    media.concat_wavs(seq, mix, adir)
    mp3 = os.path.join(adir, "mix.mp3")
    media.wav_to_mp3(mix, mp3)
    total = media.wav_duration(mix)
    words = None
    s = load_settings()
    if s.get("align_with_whisper") and get_key("openai"):
        try:
            words = providers.whisper_words(get_key("openai"), mp3)
        except Exception:
            words = None
    return {"audio_url": f"/files/{body.project}/audio/mix.mp3?v={int(time.time())}",
            "duration": round(total, 3), "gap": media.GAP_SEC, "whisper_words": words}


@app.post("/api/tts/upload")
async def tts_upload(project: str = Form(...), script: str = Form(""),
                     file: UploadFile = File(...)):
    d = pdir(project)
    adir = os.path.join(d, "audio")
    os.makedirs(adir, exist_ok=True)
    raw = os.path.join(adir, "upload_" + re.sub(r"[^\w.\-]", "", file.filename or "audio.webm"))
    with open(raw, "wb") as f:
        f.write(await file.read())
    mix = os.path.join(adir, "mix.wav")
    media.to_wav24(raw, mix)
    mp3 = os.path.join(adir, "mix.mp3")
    media.wav_to_mp3(mix, mp3)
    total = media.wav_duration(mix)
    words = []
    s = load_settings()
    if s.get("align_with_whisper") and get_key("openai"):
        try:
            words = providers.whisper_words(get_key("openai"), mp3)
        except Exception:
            words = []
    if not words and script.strip():
        words = media.estimate_words(media.strip_tags(script), 0.0, total)
    return {"audio_url": f"/files/{project}/audio/mix.mp3?v={int(time.time())}",
            "duration": round(total, 3), "words": words}


class PreviewIn(BaseModel):
    engine: str
    voice: str
    controls: dict = {}


PREVIEW_TEXTS = {
    "en": "Hey there! This is a quick preview of how this voice sounds.",
    "ko": "안녕하세요! 이 목소리가 어떻게 들리는지 간단히 들려드릴게요.",
    "ja": "こんにちは！この声がどんな風に聞こえるか、少しだけお聞かせします。",
    "zh": "你好！这是这个声音的快速试听。",
    "es": "¡Hola! Esta es una vista previa rápida de cómo suena esta voz.",
    "fr": "Bonjour ! Voici un aperçu rapide de cette voix.",
    "de": "Hallo! Das ist eine kurze Vorschau, wie diese Stimme klingt.",
    "it": "Ciao! Questa è una breve anteprima di come suona questa voce.",
    "pt": "Olá! Esta é uma prévia rápida de como esta voz soa.",
    "ru": "Привет! Это короткий пример того, как звучит этот голос.",
    "hi": "नमस्ते! यह इस आवाज़ की एक छोटी झलक है।",
    "ar": "مرحباً! هذه معاينة سريعة لصوت هذا المتحدث.",
    "tr": "Merhaba! Bu sesin nasıl duyulduğuna dair hızlı bir önizleme.",
    "nl": "Hallo! Dit is een korte preview van hoe deze stem klinkt.",
    "pl": "Cześć! To krótka próbka brzmienia tego głosu.",
    "el": "Γεια σου! Αυτή είναι μια γρήγορη προεπισκόπηση αυτής της φωνής.",
    "he": "שלום! זוהי תצוגה מקדימה קצרה של הקול הזה.",
}


@app.post("/api/tts/preview")
def tts_preview(body: PreviewIn):
    meta = next((m for m in registry.TTS_MODELS if m["id"] == body.engine), {})
    tmp = os.path.join(ROOT, "static", "_preview")
    os.makedirs(tmp, exist_ok=True)
    base = os.path.join(tmp, "preview")
    lang = str((body.controls or {}).get("language_id", "")).strip().lower()
    text = PREVIEW_TEXTS.get(lang, PREVIEW_TEXTS["en"])
    wav, _ = _tts_one(body.engine, meta.get("provider", "edge"), body.voice,
                      text, body.controls, base)
    return {"url": f"/static/_preview/{os.path.basename(wav)}?v={int(time.time())}"}


# ---------------- beats retime ----------------
class RetimeIn(BaseModel):
    texts: list
    words: list
    duration: float


@app.post("/api/beats/retime")
def retime(body: RetimeIn):
    spans = media.map_scenes_to_words(body.texts, body.words, body.duration)
    return {"spans": [{"start": a, "end": b, "duration": round(b - a, 3)} for a, b in spans]}


# ---------------- images ----------------
MIME_BY_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def load_refs(project_dir, limit):
    """STYLE reference images from project refs/ dir -> [(mime, bytes)], capped at limit."""
    rdir = os.path.join(project_dir, "refs")
    refs = []
    if limit > 0 and os.path.isdir(rdir):
        for fn in sorted(os.listdir(rdir))[:limit]:
            mime = MIME_BY_EXT.get(os.path.splitext(fn)[1].lower())
            if mime:
                with open(os.path.join(rdir, fn), "rb") as f:
                    refs.append((mime, f.read()))
    return refs


def load_chars(project_dir, limit):
    """CHARACTER reference images from chars/ dir -> [(name, mime, bytes)].
    The filename (without extension) is the character's name — prompts that
    mention that name pull this exact character into the scene."""
    cdir = os.path.join(project_dir, "chars")
    chars = []
    if limit > 0 and os.path.isdir(cdir):
        for fn in sorted(os.listdir(cdir))[:limit]:
            mime = MIME_BY_EXT.get(os.path.splitext(fn)[1].lower())
            if mime:
                with open(os.path.join(cdir, fn), "rb") as f:
                    chars.append((os.path.splitext(fn)[0].replace("_", " "), mime, f.read()))
    return chars


class ImageIn(BaseModel):
    project: str
    beat_index: int  # -1 = thumbnail
    model: str
    preprompt: str = ""
    bible: str = ""
    prompt: str = ""
    ratio: str = "16:9"
    controls: dict = {}
    use_refs: bool = True
    grounding: bool = False     # Grounding with Google Search (news / real-time facts)
    edit_from: str = ""         # filename in images/ — EDIT this image instead of a new scene


EDIT_FALLBACK_MODEL = "gemini-2.5-flash-image"  # models that can't edit hand EDIT beats to Nano Banana


@app.post("/api/image/warmup")
def image_warmup():
    """Load the local FLUX pipeline in the background so the first image is instant."""
    threading.Thread(target=providers.warmup_flux_safe, daemon=True).start()
    return {"ok": True}


@app.post("/api/image/cancel")
def image_cancel():
    """Abort the local image generation currently in flight (second press of Stop)."""
    providers.cancel_flux()
    return {"ok": True}


@app.post("/api/image")
def gen_image(body: ImageIn):
    if not body.prompt.strip():
        raise HTTPException(400, "This beat has no image prompt — write one first.")
    d = pdir(body.project)
    meta = next((m for m in registry.IMAGE_MODELS if m["id"] == body.model), {})
    edit_base = None
    if body.edit_from:
        p = os.path.join(d, "images", os.path.basename(body.edit_from.split("?")[0]))
        if os.path.exists(p):
            with open(p, "rb") as f:
                edit_base = ("image/png", f.read())
    model_id = body.model
    if edit_base and not meta.get("can_edit"):
        # e.g. FLUX makes the new scenes, Nano Banana applies the edits
        model_id = EDIT_FALLBACK_MODEL
        meta = next((m for m in registry.IMAGE_MODELS if m["id"] == model_id), meta)
    if edit_base:
        # edit mode: the base image already carries style + continuity + characters
        full = body.prompt.strip()
        refs, chars = [], []
    else:
        total = int(meta.get("max_refs", 0))
        # characters get dedicated slots when the model documents them, else share the pool
        char_cap = int(meta.get("ref_characters", 0) or total)
        chars = load_chars(d, min(char_cap, total)) if (body.use_refs and total) else []
        refs = load_refs(d, max(0, total - len(chars))) if body.use_refs else []
        parts = []
        if body.preprompt.strip():
            parts.append(body.preprompt.strip())
        if body.bible.strip():
            parts.append("Visual continuity (keep recurring subjects consistent): " + body.bible.strip())
        parts.append(body.prompt)
        full = ". ".join(p.rstrip(".") for p in parts if p)
    key_name = "replicate" if meta.get("provider") == "replicate" else meta.get("provider", "")
    data = providers.generate_image(meta.get("provider"), get_key(key_name),
                                    model_id, full, body.ratio, body.controls, refs,
                                    edit_base=edit_base,
                                    grounding=body.grounding and bool(meta.get("grounding")),
                                    image_search=body.grounding and bool(meta.get("image_search")),
                                    model_path=meta.get("model_path"), chars=chars)
    idir = os.path.join(d, "images")
    os.makedirs(idir, exist_ok=True)
    stem = "thumbnail" if body.beat_index < 0 else f"beat{body.beat_index:03d}"
    k = 1
    while os.path.exists(os.path.join(idir, f"{stem}_v{k}.png")):
        k += 1
    fn = f"{stem}_v{k}.png"
    with open(os.path.join(idir, fn), "wb") as f:
        f.write(data)
    usd = model_price(model_id, "price_image", meta.get("price_image", 0))
    return {"url": f"/files/{body.project}/images/{fn}", "version": k,
            "prompt_used": full, "cost_cad": round(cad(usd), 5)}


# ---------------- sfx ----------------
SFX_EXTS = (".mp3", ".wav", ".ogg", ".flac", ".m4a")


def sfx_path(name):
    for ext in SFX_EXTS:
        p = os.path.join(SFX_DIR, name + ext)
        if os.path.exists(p):
            return p
    return None


def available_sfx():
    """Names of the sound files actually present in the sfx/ folder."""
    return sorted({os.path.splitext(f)[0] for f in os.listdir(SFX_DIR)
                   if os.path.splitext(f)[1].lower() in SFX_EXTS})


@app.get("/api/sfx")
def sfx_list():
    """Suggested 50-name library + whatever extra files are in the folder."""
    have = set(available_sfx())
    out = [{"name": n, "available": n in have} for n in registry.SFX_NAMES]
    out += [{"name": n, "available": True} for n in sorted(have - set(registry.SFX_NAMES))]
    return {"sfx": out}


# ---------------- music ----------------
@app.get("/api/refs")
def refs_list(project: str):
    d = pdir(project)
    rdir = os.path.join(d, "refs")
    out = []
    if os.path.isdir(rdir):
        for fn in sorted(os.listdir(rdir)):
            if os.path.splitext(fn)[1].lower() in MIME_BY_EXT:
                out.append({"file": fn, "url": f"/files/{project}/refs/{fn}"})
    return {"refs": out}


@app.post("/api/refs/upload")
async def refs_upload(project: str = Form(...), files: list[UploadFile] = File(...)):
    d = pdir(project)
    rdir = os.path.join(d, "refs")
    os.makedirs(rdir, exist_ok=True)
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in MIME_BY_EXT:
            continue
        k = 1
        while os.path.exists(os.path.join(rdir, f"ref{k:02d}{ext}")):
            k += 1
        with open(os.path.join(rdir, f"ref{k:02d}{ext}"), "wb") as out:
            out.write(await f.read())
    return refs_list(project)


@app.delete("/api/refs/{project}/{fn}")
def refs_delete(project: str, fn: str):
    p = os.path.join(pdir(project), "refs", os.path.basename(fn))
    if os.path.exists(p):
        os.remove(p)
    return refs_list(project)


# ---------------- characters (named reference images) ----------------
@app.get("/api/chars")
def chars_list(project: str):
    d = pdir(project)
    cdir = os.path.join(d, "chars")
    out = []
    if os.path.isdir(cdir):
        for fn in sorted(os.listdir(cdir)):
            if os.path.splitext(fn)[1].lower() in MIME_BY_EXT:
                out.append({"file": fn, "name": os.path.splitext(fn)[0].replace("_", " "),
                            "url": f"/files/{project}/chars/{fn}"})
    return {"chars": out}


@app.post("/api/chars/upload")
async def chars_upload(project: str = Form(...), name: str = Form(""),
                       file: UploadFile = File(...)):
    d = pdir(project)
    cdir = os.path.join(d, "chars")
    os.makedirs(cdir, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1].lower() or ".png"
    if ext not in MIME_BY_EXT:
        raise HTTPException(400, "Use a png / jpg / webp image.")
    base = re.sub(r"[^\w\- ]", "", name.strip()) or os.path.splitext(file.filename or "character")[0]
    base = re.sub(r"\s+", "_", base.strip()) or "character"
    fn, k = base + ext, 1
    while os.path.exists(os.path.join(cdir, fn)):
        k += 1
        fn = f"{base}{k}{ext}"
    with open(os.path.join(cdir, fn), "wb") as f:
        f.write(await file.read())
    return chars_list(project)


class CharRenameIn(BaseModel):
    project: str
    file: str
    name: str


@app.post("/api/chars/rename")
def chars_rename(body: CharRenameIn):
    d = pdir(body.project)
    cdir = os.path.join(d, "chars")
    src = os.path.join(cdir, os.path.basename(body.file))
    if not os.path.exists(src):
        raise HTTPException(404, "Character image not found.")
    base = re.sub(r"\s+", "_", re.sub(r"[^\w\- ]", "", body.name.strip()))
    if not base:
        raise HTTPException(400, "Give the character a name.")
    dst = os.path.join(cdir, base + os.path.splitext(src)[1])
    if os.path.abspath(dst) != os.path.abspath(src):
        if os.path.exists(dst):
            raise HTTPException(400, "A character with that name already exists.")
        os.rename(src, dst)
    return chars_list(body.project)


@app.delete("/api/chars/{project}/{fn}")
def chars_delete(project: str, fn: str):
    p = os.path.join(pdir(project), "chars", os.path.basename(fn))
    if os.path.exists(p):
        os.remove(p)
    return chars_list(project)


@app.post("/api/music/upload")
async def music_upload(project: str = Form(...), file: UploadFile = File(...)):
    d = pdir(project)
    adir = os.path.join(d, "audio")
    os.makedirs(adir, exist_ok=True)
    ext = os.path.splitext(file.filename or "music.mp3")[1][:8] or ".mp3"
    path = os.path.join(adir, "music" + ext)
    for old in os.listdir(adir):
        if old.startswith("music."):
            os.remove(os.path.join(adir, old))
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"file": os.path.basename(path),
            "url": f"/files/{project}/audio/{os.path.basename(path)}?v={int(time.time())}"}


# ---------------- render ----------------
class RenderIn(BaseModel):
    project: str
    beats: list
    words: list
    resolution: str = "1920x1080"
    motion: str = "kenburns_fade"
    scale_mode: str = "fill"
    subtitles: str = "none"
    sub_size: str = "medium"
    fps: int = 30
    missing: str = "neighbor"  # neighbor | placeholder
    music: dict = {}
    sfx: dict = {}             # {enabled: bool, volume: float}
    script: str = ""
    duration: float = 0


@app.post("/api/render")
def render(body: RenderIn):
    d = pdir(body.project)
    audio = os.path.join(d, "audio", "mix.wav")
    if not os.path.exists(audio):
        raise HTTPException(400, "Generate the voice audio first (Voice tab).")
    work = os.path.join(d, "work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    W, H = (int(x) for x in body.resolution.split("x"))
    placeholder = os.path.join(work, "placeholder.png")

    def beat_image(sc):
        sel = sc.get("selected", -1)
        vs = sc.get("versions", [])
        if 0 <= sel < len(vs):
            return os.path.join(d, "images", os.path.basename(vs[sel].split("?")[0]))
        return None

    scenes = []
    last_img = None
    pending = []
    any_img = False
    for sc in body.beats:
        img = beat_image(sc)
        if img:
            any_img = True
    if not any_img:
        raise HTTPException(400, "No beat has an image yet — generate at least one (Images tab).")
    for sc in body.beats:
        img = beat_image(sc)
        if img is None:
            if body.missing == "placeholder":
                if not os.path.exists(placeholder):
                    media.make_placeholder(placeholder, W, H)
                img = placeholder
            else:  # neighbor: reuse last image, or backfill from the next one
                img = last_img
        entry = {"image": img, "duration": sc["duration"], "motion": sc.get("motion") or None}
        scenes.append(entry)
        if img is None:
            pending.append(entry)
        else:
            last_img = img
            for p in pending:
                p["image"] = img
            pending = []
    if pending:
        if not os.path.exists(placeholder):
            media.make_placeholder(placeholder, W, H)
        for p in pending:
            p["image"] = placeholder

    total = body.duration or (body.words[-1]["end"] if body.words else 0)
    if body.script.strip() and body.words:
        srt = media.build_srt_sentences(body.script, body.words, total)
    else:
        srt = media.build_srt(body.words) if body.words else ""
    if body.subtitles in ("srt", "both", "embedded") and srt:
        with open(os.path.join(d, "exports", "subtitles.srt"), "w", encoding="utf-8") as f:
            f.write(srt)
    music = None
    mf = (body.music or {}).get("file", "")
    if mf:
        mp = os.path.join(d, "audio", os.path.basename(mf))
        if os.path.exists(mp):
            music = {"path": mp, "volume": float(body.music.get("volume", 0.25)),
                     "duck": bool(body.music.get("duck", True))}
    sfx_events = []
    if (body.sfx or {}).get("enabled", True):
        vol = float((body.sfx or {}).get("volume", 0.5))
        for sc in body.beats:
            name = (sc.get("sfx") or "").strip()
            if name:
                p = sfx_path(name)
                if p:
                    sfx_events.append({"path": p, "at": float(sc.get("start", 0)), "volume": vol})
    out = os.path.join(d, "exports", "video.mp4")
    status = {"stage": "Starting", "progress": 0, "error": "", "file": ""}
    RENDER_STATUS[body.project] = status

    def job():
        try:
            media.render_video(work, scenes, audio, body.resolution, body.motion,
                               "embedded" if body.subtitles in ("embedded", "both") else "none",
                               srt, out, status, fps=body.fps, sub_size=body.sub_size,
                               scale_mode=body.scale_mode, music=music, sfx=sfx_events)
        except Exception as e:
            status.update(stage="Error", error=str(e)[:1000])

    threading.Thread(target=job, daemon=True).start()
    return {"ok": True}


@app.get("/api/render/status")
def render_status(project: str):
    s = RENDER_STATUS.get(project, {"stage": "Idle", "progress": 0, "error": "", "file": ""})
    out = dict(s)
    if s.get("file"):
        out["url"] = f"/files/{project}/exports/{s['file']}?v={int(time.time())}"
    return out


# ---------------- export ----------------
def chapters_text(state):
    ch = (state.get("director") or {}).get("chapters") or []
    beats = state.get("beats") or []
    if not ch or not beats:
        return ""
    lines = []
    for c in ch:
        try:
            t = float(beats[int(c.get("beat", 0))].get("start", 0))
        except (IndexError, ValueError, TypeError):
            continue
        m, sec = divmod(int(t), 60)
        h, m = divmod(m, 60)
        stamp = f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
        lines.append(f"{stamp} {c.get('title', '')}")
    if lines and not lines[0].startswith(("0:00", "0:00:00")):
        lines.insert(0, "0:00 Intro")
    return "\n".join(lines)


@app.get("/api/export/{name}")
def export(name: str, what: str = "all"):
    d = pdir(name)
    state = {}
    try:
        with open(os.path.join(d, "project.json"), encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        pass
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def add_dir(sub, pattern=None):
            p = os.path.join(d, sub)
            if os.path.isdir(p):
                for fn in sorted(os.listdir(p)):
                    if pattern is None or re.match(pattern, fn):
                        z.write(os.path.join(p, fn), f"{sub}/{fn}")

        if what in ("all", "script"):
            sp = os.path.join(d, "script.txt")
            if os.path.exists(sp):
                z.write(sp, "script.txt")
            z.write(os.path.join(d, "project.json"), "project.json")
            ct = chapters_text(state)
            if ct:
                z.writestr("chapters.txt", ct)
            title = (state.get("director") or {}).get("title", "")
            if title:
                z.writestr("title.txt", title)
        if what in ("all", "audio"):
            add_dir("audio", r"(mix\.(mp3|wav)|seg\d+(_v\d+)?\.wav|music\..*)$")
        if what in ("all", "images"):
            add_dir("images")
        if what in ("all", "video"):
            add_dir("exports")
    buf.seek(0)
    return Response(buf.read(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}_{what}.zip"'})


app.mount("/files", StaticFiles(directory=PROJECTS), name="files")
app.mount("/voices", StaticFiles(directory=VOICES_DIR), name="voices")
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8765")).start()
    uvicorn.run(app, host="127.0.0.1", port=8765)
