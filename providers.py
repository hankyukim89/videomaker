"""Raw HTTP calls to OpenAI / Anthropic / Gemini / Azure / ElevenLabs + local Chatterbox."""
import base64
import json
import struct
import threading
import requests

TIMEOUT = 300

# One local-GPU job at a time: an 8GB card can't run Chatterbox TTS and FLUX
# image generation simultaneously — concurrent requests queue here instead of crashing.
_GPU_LOCK = threading.Lock()


class ProviderError(Exception):
    pass


def _check(r):
    if r.status_code >= 400:
        try:
            msg = json.dumps(r.json())[:500]
        except Exception:
            msg = r.text[:500]
        raise ProviderError(f"HTTP {r.status_code}: {msg}")
    return r


# ====================== TEXT ======================

def text_openai(key, model, system, user, controls, reasoning=False):
    body = {"model": model, "messages": []}
    if system:
        body["messages"].append({"role": "system", "content": system})
    body["messages"].append({"role": "user", "content": user})
    mt = int(controls.get("max_tokens", 8192))
    if reasoning:
        body["max_completion_tokens"] = mt
        if controls.get("reasoning_effort"):
            body["reasoning_effort"] = controls["reasoning_effort"]
    else:
        body["max_tokens"] = mt
        for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            if k in controls and controls[k] is not None:
                body[k] = float(controls[k])
    r = _check(requests.post("https://api.openai.com/v1/chat/completions",
                             headers={"Authorization": f"Bearer {key}"}, json=body, timeout=TIMEOUT))
    d = r.json()
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def text_claude(key, model, system, user, controls):
    body = {"model": model, "max_tokens": int(controls.get("max_tokens", 8192)),
            "messages": [{"role": "user", "content": user}]}
    if system:
        body["system"] = system
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if src in controls and controls[src] is not None:
            v = controls[src]
            body[dst] = int(v) if dst == "top_k" else float(v)
    r = _check(requests.post("https://api.anthropic.com/v1/messages",
                             headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                             json=body, timeout=TIMEOUT))
    d = r.json()
    text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    u = d.get("usage", {})
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


def text_gemini(key, model, system, user, controls):
    gen = {"maxOutputTokens": int(controls.get("max_tokens", 8192))}
    if controls.get("temperature") is not None:
        gen["temperature"] = float(controls["temperature"])
    if controls.get("top_p") is not None:
        gen["topP"] = float(controls["top_p"])
    if controls.get("top_k") is not None:
        gen["topK"] = int(controls["top_k"])
    body = {"contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": gen}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = _check(requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key}, json=body, timeout=TIMEOUT))
    d = r.json()
    try:
        parts = d["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        raise ProviderError(f"Gemini returned no candidates: {json.dumps(d)[:400]}")
    text = "".join(p.get("text", "") for p in parts)
    u = d.get("usageMetadata", {})
    return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


def generate_text(provider, key, model, system, user, controls, reasoning=False):
    if not key and provider != "edge":
        raise ProviderError(f"No API key configured for {provider}")
    if provider == "openai":
        return text_openai(key, model, system, user, controls, reasoning)
    if provider == "claude":
        return text_claude(key, model, system, user, controls)
    if provider == "gemini":
        return text_gemini(key, model, system, user, controls)
    raise ProviderError(f"Unknown text provider {provider}")


# ====================== TTS ======================

def _pcm_to_wav(pcm, rate=24000, channels=1, width=2):
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + \
        struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * channels * width, channels * width, width * 8) + \
        b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def tts_azure(key, region, voice, text, controls):
    """Azure Speech via SDK (gives word-level timestamps).
    Returns (wav_bytes, words[{word,start,end}])."""
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        raise ProviderError("Azure Speech SDK not installed. Run: "
                            ".venv\\Scripts\\pip install azure-cognitiveservices-speech")
    if not key:
        raise ProviderError("Add your Azure Speech key + region in Settings first.")
    cfg = speechsdk.SpeechConfig(subscription=key, region=region or "eastus")
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
    cfg.request_word_level_timestamps()
    synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
    words = []

    def on_word(evt):
        # audio_offset is in ticks (100ns)
        start = evt.audio_offset / 1e7
        dur = evt.duration.total_seconds() if evt.duration else 0.3
        words.append({"word": evt.text, "start": round(start, 3), "end": round(start + dur, 3)})

    synth.synthesis_word_boundary.connect(on_word)
    rate = float(controls.get("rate", 1.0))
    pitch = int(controls.get("pitch", 0))
    style = (controls.get("style") or "").strip()
    import html
    safe = html.escape(text)
    inner = f'<prosody rate="{round((rate - 1) * 100)}%" pitch="{pitch:+d}%">{safe}</prosody>'
    if style:
        inner = f'<mstts:express-as style="{html.escape(style)}">{inner}</mstts:express-as>'
    ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">'
            f'<voice name="{voice}">{inner}</voice></speak>')
    res = synth.speak_ssml_async(ssml).get()
    if res.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        detail = ""
        if res.reason == speechsdk.ResultReason.Canceled:
            detail = res.cancellation_details.error_details or str(res.cancellation_details.reason)
        raise ProviderError(f"Azure TTS failed: {detail or res.reason}")
    return res.audio_data, [w for w in words if w["word"].strip()]


def voices_azure(key, region):
    if not key:
        raise ProviderError("Add your Azure Speech key + region in Settings first.")
    r = _check(requests.get(
        f"https://{region or 'eastus'}.tts.speech.microsoft.com/cognitiveservices/voices/list",
        headers={"Ocp-Apim-Subscription-Key": key}, timeout=30))
    out = []
    for v in r.json():
        styles = v.get("StyleList") or []
        lab = f"{v['ShortName']} ({v.get('Gender', '?')})" + (f" · styles: {', '.join(styles[:6])}" if styles else "")
        out.append({"id": v["ShortName"], "label": lab, "locale": v.get("Locale", "")})
    return out


def tts_elevenlabs(key, model, voice_id, text, controls, prev_text=None, next_text=None):
    """Returns (mp3_bytes, words_or_None). Uses /with-timestamps for word-level timing."""
    vs = {}
    for k in ("stability", "similarity_boost", "style", "speed"):
        if controls.get(k) is not None:
            vs[k] = float(controls[k])
    body = {"text": text, "model_id": model}
    if vs:
        body["voice_settings"] = vs
    if prev_text:
        body["previous_text"] = prev_text[-500:]
    if next_text:
        body["next_text"] = next_text[:500]
    r = _check(requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        json=body, timeout=TIMEOUT))
    d = r.json()
    audio = base64.b64decode(d["audio_base64"])
    words = None
    al = d.get("alignment") or {}
    chars = al.get("characters") or []
    starts = al.get("character_start_times_seconds") or []
    ends = al.get("character_end_times_seconds") or []
    if chars and len(chars) == len(starts) == len(ends):
        words, cur, w_start = [], "", None
        prev_e = 0.0
        for c, s, e in zip(chars, starts, ends):
            if c.isspace():
                if cur:
                    words.append({"word": cur, "start": round(w_start, 3), "end": round(prev_e, 3)})
                    cur, w_start = "", None
            else:
                if w_start is None:
                    w_start = s
                cur += c
            prev_e = e
        if cur:
            words.append({"word": cur, "start": round(w_start, 3), "end": round(prev_e, 3)})
        # ElevenLabs v3 may include audio tags like [laugh] in alignment - drop bracketed tokens
        words = [w for w in words if not (w["word"].startswith("[") and w["word"].endswith("]"))]
    return audio, words


# ---------------------- Chatterbox (local, free) — Turbo + original ----------------------
_CHATTERBOX = {}        # variant ("turbo" | "original") -> loaded model
_CB_DEFAULT_CONDS = {}  # variant -> built-in default-voice conditionals (captured at load)
_CB_CONDS = {}          # (variant, path, mtime, exaggeration) -> prepared clone conditionals


def _chatterbox_class(variant):
    """Return the model class for the requested local Chatterbox variant."""
    if variant == "original":
        from chatterbox.tts import ChatterboxTTS as CB
        return CB
    if variant == "mtl":  # Multilingual — 23 languages incl. Korean, same cloning
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as CB
        return CB
    # default: Turbo (fall back to the original build if Turbo isn't in this install)
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS as CB
        return CB
    except ImportError:
        from chatterbox.tts import ChatterboxTTS as CB
        return CB


def _patch_tokenizer_dtype(model):
    """Reference clips loudness-normalised by Chatterbox come back as float64, which
    blows up the s3 tokenizer's mel matmul ('expected scalar type Double but found Float').
    Force the audio to float32 right before the spectrogram so cloning never crashes."""
    import torch
    tok = getattr(getattr(model, "s3gen", None), "tokenizer", None)
    if tok is None or getattr(tok, "_vm_dtype_patched", False):
        return
    orig = tok.log_mel_spectrogram

    def patched(audio, padding=0):
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        return orig(audio.float(), padding)

    tok.log_mel_spectrogram = patched
    tok._vm_dtype_patched = True


def _chatterbox_model(variant="turbo"):
    if variant in _CHATTERBOX:
        return _CHATTERBOX[variant]
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        CB = _chatterbox_class(variant)
        model = CB.from_pretrained(device=device)
        _patch_tokenizer_dtype(model)
        _CHATTERBOX[variant] = model
        _CB_DEFAULT_CONDS[variant] = getattr(model, "conds", None)
    except ImportError:
        raise ProviderError(
            "Chatterbox is not installed. Run: .venv\\Scripts\\pip install chatterbox-tts\n"
            "(First run also downloads the model, ~2 GB. NVIDIA GPU strongly recommended; "
            "CPU works but is slow.)")
    except Exception as e:
        raise ProviderError(f"Chatterbox failed to load: {e}")
    return _CHATTERBOX[variant]


def _detect_lang(text):
    """Best-effort script detection so Multilingual 'just knows' the language."""
    counts = {}
    for ch in text:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            k = "ko"      # Hangul
        elif 0x3040 <= o <= 0x30FF:
            k = "ja"      # Hiragana / Katakana
        elif 0x4E00 <= o <= 0x9FFF:
            k = "zh"      # CJK ideographs (kanji counted as zh, fixed below)
        elif 0x0400 <= o <= 0x04FF:
            k = "ru"      # Cyrillic
        elif 0x0590 <= o <= 0x05FF:
            k = "he"      # Hebrew
        elif 0x0600 <= o <= 0x06FF:
            k = "ar"      # Arabic
        elif 0x0900 <= o <= 0x097F:
            k = "hi"      # Devanagari
        elif 0x0370 <= o <= 0x03FF:
            k = "el"      # Greek
        else:
            continue
        counts[k] = counts.get(k, 0) + 1
    if not counts:
        return "en"
    if counts.get("ja") and counts.get("zh"):
        counts["ja"] += counts.pop("zh")  # kana present -> kanji belong to Japanese
    return max(counts, key=counts.get)


def _cb_set_voice(model, variant, voice_path, exaggeration):
    """Analyse the reference clip ONCE per (voice, exaggeration) and cache the
    resulting conditionals — in memory for this session AND on disk next to the
    clip, so a voice you've used before is never re-analysed, even after restart."""
    import os
    import torch
    exag = 0.5 if exaggeration is None else float(exaggeration)
    try:
        mtime = os.path.getmtime(voice_path)  # replacing the clip invalidates both caches
    except OSError:
        mtime = 0
    key = (variant, voice_path, mtime, round(exag, 3))
    conds = _CB_CONDS.get(key)
    if conds is not None:
        model.conds = conds
        return
    cpath = f"{voice_path}.{variant}.{round(exag, 3)}.conds.pt"
    if os.path.exists(cpath) and os.path.getmtime(cpath) >= mtime:
        try:  # disk cache from a previous session
            dev = getattr(model, "device", "cpu")
            try:
                conds = torch.load(cpath, map_location=dev, weights_only=False)
            except TypeError:  # older torch: no weights_only kwarg
                conds = torch.load(cpath, map_location=dev)
            if hasattr(conds, "to"):
                conds = conds.to(dev)
        except Exception:
            conds = None  # corrupt/incompatible cache — just re-analyse
    if conds is None:
        try:
            model.prepare_conditionals(voice_path, exaggeration=exag)
        except TypeError:  # Turbo build takes no exaggeration arg
            model.prepare_conditionals(voice_path)
        conds = model.conds
        try:
            torch.save(conds, cpath)
        except Exception:
            pass  # disk cache is best-effort only
    model.conds = conds
    _CB_CONDS[key] = conds


def tts_chatterbox(voice_path, text, controls, variant="turbo"):
    """Local Chatterbox. variant='turbo' (fast) or 'original' (emotion slider).
    voice_path = reference audio for cloning ('' = default voice).
    Supports inline expression tags like [laugh], [sigh], [cough]. Returns wav bytes."""
    with _GPU_LOCK:  # never run TTS and local image generation at the same time
        model = _chatterbox_model(variant)
        kw = {}
        for k in ("exaggeration", "cfg_weight", "temperature"):
            if controls.get(k) is not None:
                try:
                    kw[k] = float(controls[k])
                except (TypeError, ValueError):
                    pass
        if "Turbo" in type(model).__name__:
            # Turbo ignores these (and warns about them) — don't send
            kw.pop("exaggeration", None)
            kw.pop("cfg_weight", None)
        if variant == "mtl":
            lang = (controls.get("language_id") or "auto").strip().lower()
            kw["language_id"] = _detect_lang(text) if lang in ("", "auto") else lang
        if voice_path:
            _cb_set_voice(model, variant, voice_path, kw.get("exaggeration"))
        elif _CB_DEFAULT_CONDS.get(variant) is not None:
            model.conds = _CB_DEFAULT_CONDS[variant]  # restore built-in voice after cloning
        try:
            wav = model.generate(text, **kw)
        except TypeError:  # model may not accept all kwargs
            wav = model.generate(text, **{k: v for k, v in kw.items() if k == "language_id"})
    # write the WAV ourselves — torchaudio.save now requires the extra torchcodec package
    import numpy as np
    data = wav.squeeze().detach().cpu().numpy()
    if data.ndim > 1:
        data = data[0]
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2").tobytes()
    return _pcm_to_wav(pcm, rate=int(model.sr))


# ---------------------- Replicate (rented GPU per request) ----------------------
import time as _time

_REPLICATE_FILE_CACHE = {}  # local path -> hosted url


def _replicate_poll(key, d):
    while d.get("status") in ("starting", "processing"):
        _time.sleep(2)
        d = _check(requests.get(d["urls"]["get"],
                                headers={"Authorization": f"Bearer {key}"}, timeout=60)).json()
    if d.get("status") != "succeeded":
        raise ProviderError(f"Replicate run failed: {str(d.get('error'))[:300]}")
    return d["output"]


def replicate_run(key, model_path, input_dict):
    """Run an official model on Replicate, waiting for the result. Returns output."""
    if not key:
        raise ProviderError("Add your Replicate API key in Settings first (replicate.com → API tokens).")
    r = requests.post(f"https://api.replicate.com/v1/models/{model_path}/predictions",
                      headers={"Authorization": f"Bearer {key}", "Prefer": "wait=60",
                               "Content-Type": "application/json"},
                      json={"input": input_dict}, timeout=TIMEOUT)
    return _replicate_poll(key, _check(r).json())


def replicate_upload(key, path):
    """Upload a local file to Replicate's file storage; returns a URL usable as model input."""
    cached = _REPLICATE_FILE_CACHE.get(path)
    if cached:
        return cached
    with open(path, "rb") as f:
        r = _check(requests.post("https://api.replicate.com/v1/files",
                                 headers={"Authorization": f"Bearer {key}"},
                                 files={"content": (path.replace("\\", "/").split("/")[-1], f)},
                                 timeout=TIMEOUT))
    url = (r.json().get("urls") or {}).get("get") or r.json().get("url")
    if not url:
        raise ProviderError("Replicate file upload returned no URL")
    _REPLICATE_FILE_CACHE[path] = url
    return url


def _first_url(output):
    """Replicate outputs vary: a url string, list of urls, or dict — find the first url."""
    if isinstance(output, str) and output.startswith("http"):
        return output
    if isinstance(output, list):
        for o in output:
            u = _first_url(o)
            if u:
                return u
    if isinstance(output, dict):
        for o in output.values():
            u = _first_url(o)
            if u:
                return u
    return None


def tts_chatterbox_replicate(key, voice_path, text, controls,
                             model_path="resemble-ai/chatterbox-turbo"):
    """Chatterbox (Turbo or original) on Replicate's GPUs. Returns audio bytes (wav/mp3)."""
    inp = {"text": text}
    for k in ("temperature", "exaggeration", "cfg_weight"):
        if controls.get(k) is not None:
            inp[k] = float(controls[k])
    if voice_path:
        inp["audio_prompt"] = replicate_upload(key, voice_path)
    try:
        out = replicate_run(key, model_path, inp)
    except ProviderError as e:
        msg = str(e)
        if "422" in msg:  # field naming differs across versions — retry with 'prompt'
            inp.pop("exaggeration", None)
            inp.pop("cfg_weight", None)
            inp["prompt"] = inp.pop("text", text)
            out = replicate_run(key, model_path, inp)
        else:
            raise
    url = _first_url(out)
    if not url:
        raise ProviderError(f"Chatterbox (Replicate) returned no audio: {str(out)[:200]}")
    return _check(requests.get(url, timeout=TIMEOUT)).content


def replicate_upload_bytes(key, data, name="ref.png"):
    """Upload raw image bytes to Replicate's file storage (cached by content hash)."""
    import hashlib
    import io as _io
    h = "sha1:" + hashlib.sha1(data).hexdigest()
    cached = _REPLICATE_FILE_CACHE.get(h)
    if cached:
        return cached
    r = _check(requests.post("https://api.replicate.com/v1/files",
                             headers={"Authorization": f"Bearer {key}"},
                             files={"content": (name, _io.BytesIO(data))},
                             timeout=TIMEOUT))
    url = (r.json().get("urls") or {}).get("get") or r.json().get("url")
    if not url:
        raise ProviderError("Replicate file upload returned no URL")
    _REPLICATE_FILE_CACHE[h] = url
    return url


def image_flux(key, model_path, prompt, ratio, refs=None, edit_base=None, chars=None):
    """FLUX on Replicate. schnell/dev = text only; FLUX.2 Pro also takes an edit
    base + named character refs + style refs via image_input. Returns image bytes."""
    imgs = []
    if edit_base is not None:
        imgs.append(replicate_upload_bytes(key, edit_base[1], "edit_base.png"))
        prompt = ("Edit the attached image. Apply ONLY this change, keeping everything else - "
                  "art style, characters, background, colors, framing - exactly identical: "
                  + prompt)
    else:
        pre = []
        for name, _mime, data in (chars or []):
            imgs.append(replicate_upload_bytes(key, data, "char.png"))
        if chars:
            names = ", ".join(n for n, _, _ in chars)
            pre.append(f"The first {len(chars)} attached image(s) are character references, in order: {names}. "
                       "Whenever the scene mentions one of them, draw EXACTLY that character - same face, "
                       "hair, outfit and colors, only the pose/expression changes.")
        for _mime, data in (refs or []):
            imgs.append(replicate_upload_bytes(key, data, "style.png"))
        if refs:
            pre.append("The remaining attached image(s) define the art style - copy it exactly; "
                       "it matches the style described in the text.")
        if pre:
            prompt = " ".join(pre) + " Scene to draw: " + prompt
    inp = {"prompt": prompt, "aspect_ratio": ratio, "num_outputs": 1,
           "output_format": "png", "disable_safety_checker": True}
    if imgs:
        inp["image_input"] = imgs
    try:
        out = replicate_run(key, model_path, inp)
    except ProviderError as e:
        if "422" not in str(e):
            raise
        # model rejects some field names (schemas differ per FLUX version) — retry minimal
        inp = {"prompt": prompt, "aspect_ratio": ratio, "output_format": "png"}
        if imgs:
            inp["image_input"] = imgs
        out = replicate_run(key, model_path, inp)
    url = _first_url(out)
    if not url:
        raise ProviderError(f"FLUX returned no image: {str(out)[:200]}")
    return _check(requests.get(url, timeout=TIMEOUT)).content


def voices_elevenlabs(key):
    r = _check(requests.get("https://api.elevenlabs.io/v1/voices",
                            headers={"xi-api-key": key}, timeout=30))
    return [{"id": v["voice_id"], "label": f"{v['name']} ({v.get('labels', {}).get('gender', '?')})"}
            for v in r.json().get("voices", [])]


def whisper_words(key, audio_path):
    """OpenAI whisper transcription with word timestamps. Returns [{word,start,end}]."""
    with open(audio_path, "rb") as f:
        r = _check(requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (audio_path.split("/")[-1].split("\\")[-1], f)},
            data={"model": "whisper-1", "response_format": "verbose_json",
                  "timestamp_granularities[]": "word"},
            timeout=TIMEOUT))
    d = r.json()
    return [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
            for w in d.get("words", [])]


# ====================== IMAGES (Gemini / Nano Banana only) ======================

def image_gemini(key, model, prompt, ratio, controls, refs=None, edit_base=None,
                 grounding=False, image_search=False, chars=None):
    """Nano Banana models via generateContent. Returns image bytes.

    refs:       [(mime, bytes)] STYLE reference images (new scenes).
    chars:      [(name, mime, bytes)] named CHARACTER reference images.
    edit_base:  (mime, bytes) - an existing frame to EDIT; prompt = the edit instruction.
    grounding:  use Google Search as a tool (real-time facts, news, weather, charts).
    image_search: additionally enable Google Image Search grounding (3.1 Flash only).
    """
    gen = {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": ratio}}
    if controls.get("imageSize"):
        gen["imageConfig"]["imageSize"] = controls["imageSize"]
    parts = []
    if edit_base is not None:
        mime, data = edit_base
        parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}})
        prompt = ("Edit the attached image. Apply ONLY this change, keeping everything else - "
                  "art style, characters, background, colors, framing - exactly identical: "
                  + prompt)
    else:
        pre = []
        for name, mime, data in (chars or []):  # label each character image with its name
            parts.append({"text": f"Character reference - this is {name}:"})
            parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}})
        if chars:
            names = ", ".join(n for n, _, _ in chars)
            pre.append(f"CHARACTER LOCK: the labelled reference images define these characters: {names}. "
                       "Whenever the scene mentions one of them by name or role, draw EXACTLY that character - "
                       "same face, hair, body, outfit and colors, only the pose/expression changes.")
        for mime, data in (refs or []):
            parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}})
        if refs:
            pre.append("STYLE LOCK: copy the EXACT art style of the unlabelled reference images - same medium, "
                       "same line work, same color palette, same level of simplicity/detail. "
                       "The style description in the text below and the reference images describe the SAME look - "
                       "follow both; if they ever conflict, the reference images win.")
        if pre:
            prompt = " ".join(pre) + " Scene to draw: " + prompt
    parts.append({"text": prompt})
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}
    if grounding:
        if image_search:
            body["tools"] = [{"googleSearch": {"searchTypes": {"webSearch": {}, "imageSearch": {}}}}]
        else:
            body["tools"] = [{"googleSearch": {}}]
    r = _check(requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key}, json=body, timeout=TIMEOUT))
    d = r.json()
    try:
        for p in d["candidates"][0]["content"]["parts"]:
            if "inlineData" in p:
                return base64.b64decode(p["inlineData"]["data"])
    except (KeyError, IndexError):
        pass
    raise ProviderError(f"Gemini image returned no image: {json.dumps(d)[:400]}")


# ---------------------- FLUX.2 Klein 4B (local GPU, free) ----------------------
_FLUX_PIPE = None
_FLUX_CANCEL = threading.Event()  # set via /api/image/cancel to abort mid-generation


def warmup_flux_safe():
    """Load the FLUX pipeline in the background (errors are silent — the real
    generation call will surface them properly)."""
    try:
        with _GPU_LOCK:
            _flux_klein_pipe()
    except Exception:
        pass


def cancel_flux():
    _FLUX_CANCEL.set()

_FLUX_DIMS = {  # ratio -> (width, height): ~1MP, multiples of 16
    "1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344),
    "4:3": (1152, 864), "3:4": (864, 1152), "2:3": (832, 1248), "3:2": (1248, 832),
    "4:5": (896, 1120), "5:4": (1120, 896), "21:9": (1536, 656)}


def _flux_klein_pipe():
    """Load FLUX.2 Klein 4B once. On <15GB cards (e.g. RTX 3070 8GB) the
    transformer + text encoder are quantised to 4-bit and offloaded to system RAM."""
    global _FLUX_PIPE
    if _FLUX_PIPE is not None:
        return _FLUX_PIPE
    try:
        import torch
        from diffusers import Flux2KleinPipeline
    except ImportError:
        raise ProviderError(
            "Local FLUX is not installed yet. Run (one time):\n"
            ".venv\\Scripts\\pip install -U diffusers transformers accelerate bitsandbytes\n"
            "The first image also downloads the FLUX.2 Klein 4B weights (~9 GB).")
    if not torch.cuda.is_available():
        raise ProviderError("Local FLUX needs an NVIDIA GPU (CUDA). Pick a cloud image model instead.")
    repo = "black-forest-labs/FLUX.2-klein-4B"
    try:
        vram = torch.cuda.get_device_properties(0).total_memory
        if vram < 15e9:
            try:  # 8-12 GB cards: 4-bit weights + module-level CPU offload (fast path)
                try:
                    from diffusers import PipelineQuantizationConfig
                except ImportError:
                    from diffusers.quantizers import PipelineQuantizationConfig
                quant = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                  "bnb_4bit_compute_dtype": torch.bfloat16},
                    components_to_quantize=["transformer", "text_encoder"])
                pipe = Flux2KleinPipeline.from_pretrained(repo, quantization_config=quant,
                                                          torch_dtype=torch.bfloat16)
                pipe.enable_model_cpu_offload()
            except Exception:  # bitsandbytes missing/failed: slower but safe layer offload
                pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
                pipe.enable_sequential_cpu_offload()
        else:
            pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload()
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError(f"FLUX.2 Klein failed to load: {str(e)[:400]}")
    # decode/attend in chunks — keeps the VRAM spike inside 8GB so the driver
    # never falls back to (50x slower) system-RAM thrashing
    for obj, names in ((pipe, ("enable_attention_slicing",)),
                       (getattr(pipe, "vae", None), ("enable_slicing", "enable_tiling"))):
        for n in names:
            try:
                getattr(obj, n)()
            except Exception:
                pass
    _FLUX_PIPE = pipe
    return pipe


def image_flux_klein_local(prompt, ratio, controls, refs=None, edit_base=None, chars=None):
    """FLUX.2 Klein 4B on the local GPU — free. ONE model does new scenes,
    style refs, named character refs AND instruction edits. Returns PNG bytes."""
    import io as _io
    from PIL import Image
    images = []
    if edit_base is not None:
        images.append(Image.open(_io.BytesIO(edit_base[1])).convert("RGB"))
        prompt = ("Edit the attached image. Apply ONLY this change, keeping everything else - "
                  "art style, characters, background, colors, framing - exactly identical: "
                  + prompt)
    else:
        pre = []
        for name, _mime, data in (chars or []):  # characters first, in order
            images.append(Image.open(_io.BytesIO(data)).convert("RGB"))
        if chars:
            names = ", ".join(n for n, _, _ in chars)
            pre.append(f"The first {len(chars)} attached image(s) are character references, in order: {names}. "
                       "Whenever the scene mentions one of them, draw EXACTLY that character - same face, "
                       "hair, outfit and colors, only the pose/expression changes.")
        for _mime, data in (refs or []):
            images.append(Image.open(_io.BytesIO(data)).convert("RGB"))
        if refs:
            pre.append("The remaining attached image(s) define the art style - copy it exactly; "
                       "it matches the style described in the text.")
        if pre:
            prompt = " ".join(pre) + " Scene to draw: " + prompt
    w, h = _FLUX_DIMS.get(ratio, (1024, 1024))
    if str(controls.get("size", "")).startswith("0.5"):  # half the pixels ≈ half the time
        w = max(256, round(w * 0.7071 / 16) * 16)
        h = max(256, round(h * 0.7071 / 16) * 16)
    try:
        steps = max(1, int(float(controls.get("steps", 4))))
    except (TypeError, ValueError):
        steps = 4
    def _on_step(pipe_obj, step, timestep, cb_kwargs):  # lets the Stop button abort mid-run
        if _FLUX_CANCEL.is_set():
            pipe_obj._interrupt = True
        return cb_kwargs

    kw = dict(prompt=prompt, width=w, height=h, guidance_scale=1.0, num_inference_steps=steps,
              callback_on_step_end=_on_step)
    if images:
        kw["image"] = images
    with _GPU_LOCK:  # never run local image generation and TTS at the same time
        pipe = _flux_klein_pipe()
        _FLUX_CANCEL.clear()
        out = None
        # progressively drop kwargs an older diffusers build may not know
        for drop in ((), ("callback_on_step_end",), ("callback_on_step_end", "image")):
            k2 = {a: b for a, b in kw.items() if a not in drop}
            try:
                out = pipe(**k2).images[0]
                break
            except TypeError:
                continue
        if out is None:
            out = pipe(prompt=prompt, width=w, height=h).images[0]
    if _FLUX_CANCEL.is_set():
        _FLUX_CANCEL.clear()
        raise ProviderError("Image generation cancelled.")
    buf = _io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def generate_image(provider, key, model, prompt, ratio, controls, refs=None,
                   edit_base=None, grounding=False, image_search=False, model_path=None,
                   chars=None):
    if provider == "local_flux":  # runs on the user's own GPU — no key needed
        return image_flux_klein_local(prompt, ratio, controls, refs, edit_base, chars)
    if not key:
        raise ProviderError(f"No {provider} API key configured - add it in Settings.")
    if provider == "gemini":
        return image_gemini(key, model, prompt, ratio, controls, refs, edit_base,
                            grounding, image_search, chars)
    if provider == "replicate":
        return image_flux(key, model_path or model, prompt, ratio, refs, edit_base, chars)
    raise ProviderError(f"Unknown image provider {provider}")


# ====================== model listing (live refresh) ======================

def list_models_live(provider, key):
    try:
        if provider == "openai":
            r = _check(requests.get("https://api.openai.com/v1/models",
                                    headers={"Authorization": f"Bearer {key}"}, timeout=30))
            return sorted(m["id"] for m in r.json().get("data", []))
        if provider == "claude":
            r = _check(requests.get("https://api.anthropic.com/v1/models?limit=100",
                                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=30))
            return sorted(m["id"] for m in r.json().get("data", []))
        if provider == "gemini":
            r = _check(requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                headers={"x-goog-api-key": key}, timeout=30))
            return sorted(m["name"].replace("models/", "") for m in r.json().get("models", []))
    except Exception as e:
        return {"error": str(e)}
    return []
