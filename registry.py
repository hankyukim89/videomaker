"""Model registry: models, USD pricing, per-model control schemas, default profiles.
All prices editable at runtime via settings.json overrides (Settings tab)."""

# ---------- control schema helpers ----------
def num(key, label, mn, mx, step, default):
    return {"key": key, "label": label, "type": "number", "min": mn, "max": mx, "step": step, "default": default}

def sel(key, label, options, default):
    return {"key": key, "label": label, "type": "select", "options": options, "default": default}

COMMON_TEXT = [num("max_tokens", "Max tokens", 256, 65536, 256, 8192)]
TEMP = num("temperature", "Temperature", 0, 2, 0.05, 0.9)
TEMP1 = num("temperature", "Temperature", 0, 1, 0.05, 0.9)
TOP_P = num("top_p", "Top P", 0, 1, 0.01, 1.0)
TOP_K = num("top_k", "Top K", 1, 200, 1, 40)

# ---------- TEXT MODELS ----------
# price_in / price_out = USD per 1M tokens
TEXT_MODELS = [
    # Gemini
    dict(id="gemini-2.5-flash-lite", provider="gemini", label="Gemini 2.5 Flash-Lite", price_in=0.10, price_out=0.40,
         controls=[TEMP, TOP_P, TOP_K] + COMMON_TEXT, note="cheapest, free tier available"),
    dict(id="gemini-2.5-flash", provider="gemini", label="Gemini 2.5 Flash", price_in=0.30, price_out=2.50,
         controls=[TEMP, TOP_P, TOP_K] + COMMON_TEXT),
    dict(id="gemini-3.1-flash-lite-preview", provider="gemini", label="Gemini 3.1 Flash-Lite (preview)", price_in=0.25, price_out=1.50,
         controls=[TEMP, TOP_P, TOP_K] + COMMON_TEXT),
    dict(id="gemini-3-flash-preview", provider="gemini", label="Gemini 3 Flash (preview)", price_in=0.50, price_out=3.00,
         controls=[TEMP, TOP_P, TOP_K] + COMMON_TEXT),
    dict(id="gemini-2.5-pro", provider="gemini", label="Gemini 2.5 Pro", price_in=1.25, price_out=10.00,
         controls=[TEMP, TOP_P, TOP_K] + COMMON_TEXT),
    dict(id="gemini-3.1-pro-preview", provider="gemini", label="Gemini 3.1 Pro (preview)", price_in=2.00, price_out=12.00,
         controls=[TEMP, TOP_P, TOP_K] + COMMON_TEXT),
    # OpenAI
    dict(id="gpt-4.1-nano", provider="openai", label="GPT-4.1 nano", price_in=0.10, price_out=0.40,
         controls=[TEMP, TOP_P,
                   num("frequency_penalty", "Frequency penalty", -2, 2, 0.1, 0),
                   num("presence_penalty", "Presence penalty", -2, 2, 0.1, 0)] + COMMON_TEXT),
    dict(id="gpt-4.1-mini", provider="openai", label="GPT-4.1 mini", price_in=0.40, price_out=1.60,
         controls=[TEMP, TOP_P,
                   num("frequency_penalty", "Frequency penalty", -2, 2, 0.1, 0),
                   num("presence_penalty", "Presence penalty", -2, 2, 0.1, 0)] + COMMON_TEXT),
    dict(id="gpt-4o-mini", provider="openai", label="GPT-4o mini", price_in=0.15, price_out=0.60,
         controls=[TEMP, TOP_P,
                   num("frequency_penalty", "Frequency penalty", -2, 2, 0.1, 0),
                   num("presence_penalty", "Presence penalty", -2, 2, 0.1, 0)] + COMMON_TEXT),
    dict(id="gpt-5.4-nano", provider="openai", label="GPT-5.4 nano", price_in=0.20, price_out=1.25, reasoning=True,
         controls=[sel("reasoning_effort", "Reasoning effort", ["minimal", "low", "medium", "high"], "low")] + COMMON_TEXT),
    dict(id="gpt-5.4", provider="openai", label="GPT-5.4", price_in=2.50, price_out=15.00, reasoning=True,
         controls=[sel("reasoning_effort", "Reasoning effort", ["minimal", "low", "medium", "high"], "low")] + COMMON_TEXT),
    dict(id="gpt-5.5", provider="openai", label="GPT-5.5", price_in=5.00, price_out=30.00, reasoning=True,
         controls=[sel("reasoning_effort", "Reasoning effort", ["minimal", "low", "medium", "high"], "low")] + COMMON_TEXT),
    # Anthropic
    dict(id="claude-haiku-4-5", provider="claude", label="Claude Haiku 4.5", price_in=1.00, price_out=5.00,
         controls=[TEMP1, TOP_P, TOP_K] + COMMON_TEXT),
    dict(id="claude-sonnet-4-6", provider="claude", label="Claude Sonnet 4.6", price_in=3.00, price_out=15.00,
         controls=[TEMP1, TOP_P, TOP_K] + COMMON_TEXT),
    dict(id="claude-opus-4-8", provider="claude", label="Claude Opus 4.8", price_in=5.00, price_out=25.00,
         controls=[TEMP1, TOP_P, TOP_K] + COMMON_TEXT),
    dict(id="claude-fable-5", provider="claude", label="Claude Fable 5", price_in=10.00, price_out=50.00,
         controls=[TEMP1, TOP_P] + COMMON_TEXT),
]

# ---------- TTS MODELS ----------
# price = USD per 1M characters (approximation shown to user; edge/chatterbox = free)
TTS_MODELS = [
    dict(id="edge", provider="edge", label="Edge TTS (Microsoft, free)", price_chars=0.0, timestamps=True,
         controls=[num("rate", "Speed", 0.5, 2.0, 0.05, 1.0),
                   num("pitch", "Pitch (Hz offset)", -50, 50, 1, 0),
                   num("volume", "Volume %", 50, 150, 5, 100)]),
    dict(id="chatterbox", provider="chatterbox", label="Chatterbox Turbo (free, local, voice cloning)",
         price_chars=0.0, timestamps=False,
         controls=[num("exaggeration", "Emotion exaggeration", 0, 1, 0.05, 0.5),
                   num("cfg_weight", "Pacing / adherence (CFG)", 0, 1, 0.05, 0.5),
                   num("temperature", "Temperature", 0.3, 1.5, 0.05, 0.8)]),
    dict(id="chatterbox-v1", provider="chatterbox", label="Chatterbox original (free, local — emotion slider)",
         price_chars=0.0, timestamps=False,
         controls=[num("exaggeration", "Emotion exaggeration", 0, 1, 0.05, 0.5),
                   num("cfg_weight", "Pacing / adherence (CFG)", 0, 1, 0.05, 0.5),
                   num("temperature", "Temperature", 0.3, 1.5, 0.05, 0.8)]),
    dict(id="chatterbox-mtl", provider="chatterbox",
         label="Chatterbox Multilingual (free, local — 23 languages incl. Korean, voice cloning)",
         price_chars=0.0, timestamps=False,
         controls=[sel("language_id", "Language (auto = detect from the text)",
                       ["auto", "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
                        "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"], "auto"),
                   num("exaggeration", "Emotion exaggeration", 0, 1, 0.05, 0.5),
                   num("cfg_weight", "Pacing / adherence (CFG)", 0, 1, 0.05, 0.5),
                   num("temperature", "Temperature", 0.3, 1.5, 0.05, 0.8)]),
    dict(id="chatterbox-replicate", provider="replicate_tts", label="Chatterbox Turbo (Replicate API — rented GPU, fast)",
         price_chars=40.0, timestamps=False,
         controls=[num("temperature", "Temperature", 0.3, 1.5, 0.05, 0.8)]),
    dict(id="chatterbox-v1-replicate", provider="replicate_tts", label="Chatterbox original (Replicate API — emotion slider)",
         price_chars=40.0, timestamps=False,
         controls=[num("exaggeration", "Emotion exaggeration", 0, 1, 0.05, 0.5),
                   num("cfg_weight", "Pacing / adherence (CFG)", 0, 1, 0.05, 0.5),
                   num("temperature", "Temperature", 0.3, 1.5, 0.05, 0.8)]),
    dict(id="azure", provider="azure", label="Azure Speech (word timestamps)", price_chars=16.0, timestamps=True,
         controls=[num("rate", "Speed", 0.5, 2.0, 0.05, 1.0),
                   num("pitch", "Pitch %", -50, 50, 1, 0),
                   {"key": "style", "label": "Speaking style (voice-dependent, e.g. cheerful, sad, newscast)",
                    "type": "text", "default": ""}]),
    dict(id="eleven_v3", provider="elevenlabs", label="ElevenLabs v3 (emotion tags)", price_chars=200.0, timestamps=True,
         controls=[num("stability", "Stability", 0, 1, 0.05, 0.5),
                   num("similarity_boost", "Similarity", 0, 1, 0.05, 0.75),
                   num("speed", "Speed", 0.7, 1.2, 0.05, 1.0)]),
    dict(id="eleven_multilingual_v2", provider="elevenlabs", label="ElevenLabs Multilingual v2", price_chars=180.0, timestamps=True,
         controls=[num("stability", "Stability", 0, 1, 0.05, 0.5),
                   num("similarity_boost", "Similarity", 0, 1, 0.05, 0.75),
                   num("style", "Style exaggeration", 0, 1, 0.05, 0.0),
                   num("speed", "Speed", 0.7, 1.2, 0.05, 1.0)]),
    dict(id="eleven_turbo_v2_5", provider="elevenlabs", label="ElevenLabs Turbo v2.5 (cheaper)", price_chars=90.0, timestamps=True,
         controls=[num("stability", "Stability", 0, 1, 0.05, 0.5),
                   num("similarity_boost", "Similarity", 0, 1, 0.05, 0.75),
                   num("speed", "Speed", 0.7, 1.2, 0.05, 1.0)]),
    dict(id="upload", provider="local", label="Record / upload my own audio", price_chars=0.0, timestamps=False, controls=[]),
]

# ---------- IMAGE MODELS ----------
# Gemini (Nano Banana family) only - per https://ai.google.dev/gemini-api/docs/image-generation
# price_image = USD per generated image (approx for default quality; editable in Settings)
# max_refs: total reference images; ref_objects/ref_characters: doc-stated sub-limits
# grounding: supports Grounding with Google Search; image_search: also Google Image Search
IMAGE_MODELS = [
    dict(id="gemini-2.5-flash-image", provider="gemini", label="Gemini 2.5 Flash Image (Nano Banana)",
         price_image=0.039, max_refs=3, grounding=False, can_edit=True,
         ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9"],
         controls=[]),
    dict(id="gemini-3.1-flash-image", provider="gemini", label="Gemini 3.1 Flash Image (Nano Banana 2)",
         price_image=0.067, max_refs=14, ref_objects=10, ref_characters=4, grounding=True, image_search=True, can_edit=True,
         ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9", "1:4", "4:1", "1:8", "8:1"],
         controls=[sel("imageSize", "Size", ["512", "1K", "2K", "4K"], "1K")]),
    dict(id="gemini-3-pro-image", provider="gemini", label="Gemini 3 Pro Image (Nano Banana Pro)",
         price_image=0.134, max_refs=14, ref_objects=6, ref_characters=5, grounding=True, can_edit=True,
         ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9"],
         controls=[sel("imageSize", "Size", ["1K", "2K", "4K"], "1K")]),
    # FLUX.2 Klein 4B — runs on YOUR GPU, free, one model for both new scenes and edits.
    # Needs: pip install -U diffusers transformers accelerate bitsandbytes (first run downloads ~9 GB)
    dict(id="flux2-klein-local", provider="local_flux",
         label="FLUX.2 Klein 4B (free, local GPU — gen + edit)",
         price_image=0.0, max_refs=4, can_edit=True,
         ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9"],
         controls=[sel("steps", "Steps (4 = fast draft, more = finer)", ["4", "8", "16", "28"], "4"),
                   sel("size", "Resolution", ["1MP (sharp)", "0.5MP (faster)"], "1MP (sharp)")]),
    # FLUX.2 Pro via Replicate — BFL's flagship: generation + instruction edits + up to 8 reference images
    dict(id="flux-2-pro", provider="replicate", model_path="black-forest-labs/flux-2-pro",
         label="FLUX.2 Pro (Replicate API — gen + edit + refs)",
         price_image=0.03, max_refs=8, can_edit=True,
         ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9"],
         controls=[]),
    # FLUX.1 via Replicate — great + cheap for NEW scenes; EDIT beats auto-fall back to Nano Banana
    dict(id="flux-schnell", provider="replicate", model_path="black-forest-labs/flux-schnell",
         label="FLUX Schnell (Replicate, fast & cheap)", price_image=0.003, max_refs=0, can_edit=False,
         ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9"],
         controls=[]),
    dict(id="flux-dev", provider="replicate", model_path="black-forest-labs/flux-dev",
         label="FLUX Dev (Replicate, higher quality)", price_image=0.025, max_refs=0, can_edit=False,
         ratios=["1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9"],
         controls=[]),
]

# ---------- SOUND EFFECTS ----------
# Put files named <name>.mp3 (or .wav) in the sfx/ folder next to the app.
# The Director may tag any beat with one of these; missing files are skipped at render time.
SFX_NAMES = [
    # transitions & movement
    "whoosh_fast", "whoosh_soft", "swoosh_long", "teleport", "magic_wand",
    # accents & positivity
    "pop", "pop_soft", "bubble", "bling", "ding", "chime", "sparkle", "coin", "cash_register",
    # comedy & meme
    "boing", "cartoon_slip", "record_scratch", "vine_boom", "bruh", "crickets", "airhorn",
    "sad_trombone", "drumroll", "rimshot", "laugh_track",
    # tension & impact
    "suspense_riser", "riser_short", "boom_deep", "impact_soft", "thud", "punch", "slap",
    "heartbeat", "glitch",
    # objects & UI
    "camera_shutter", "typewriter", "keyboard_typing", "click", "beep", "error_beep",
    "notification", "message_pop", "clock_tick", "page_flip",
    # environment & crowd
    "thunder", "wind_gust", "door_slam", "glass_break", "applause", "crowd_gasp",
]  # 50 sounds

# ---------- DEFAULT PROFILES ----------
DEFAULT_PROFILES = {
    "Documentary": {
        "script": {"provider": "gemini", "model": "gemini-2.5-flash-lite", "speakers": 1,
                   "preprompt": "Write a documentary-style narration script. Clear, friendly, engaging, authoritative but warm. Vivid concrete details, smooth transitions, a strong hook in the first two sentences, and a satisfying conclusion. Plain spoken prose only - no headings, no stage directions, no markdown.",
                   "controls": {"temperature": 0.9, "max_tokens": 8192}},
        "voice": {"engine": "edge", "voices": {"1": "en-US-GuyNeural"}, "controls": {"rate": 1.0, "pitch": 0, "volume": 100}},
        "images": {"model": "gemini-2.5-flash-image", "ratio": "16:9", "min_sec": 4, "max_sec": 12,
                   "preprompt": "Cinematic documentary photography, 35mm, natural light, muted realistic colors, high detail. No text, no watermarks."},
        "compile": {"resolution": "1920x1080", "motion": "kenburns_fade", "subtitles": "none"},
    },
    "Video essay": {
        "script": {"provider": "gemini", "model": "gemini-2.5-flash-lite", "speakers": 1,
                   "preprompt": "Write a YouTube video essay script. Conversational but smart, opinionated, with rhetorical questions, callbacks, and momentum. Hook the viewer hard in the first 10 seconds. Write VISUALLY: anchor ideas in concrete scenes, recurring characters and objects the viewer can picture, and return to them as the argument builds - this lets the visuals evolve with the narration. Plain spoken prose only - no headings, no markdown, no stage directions.",
                   "controls": {"temperature": 1.0, "max_tokens": 8192}},
        "director": {"pacing": "frequent"},
        "voice": {"engine": "edge", "voices": {"1": "en-US-ChristopherNeural"}, "controls": {"rate": 1.1, "pitch": 0, "volume": 100}},
        "images": {"model": "gemini-2.5-flash-image", "ratio": "16:9", "min_sec": 3, "max_sec": 10,
                   "preprompt": "Stylized editorial illustration, bold composition, dramatic lighting, modern color grade. No text, no watermarks."},
        "compile": {"resolution": "1920x1080", "motion": "kenburns_fade", "subtitles": "embedded"},
    },
    "Podcast (2 hosts)": {
        "script": {"provider": "gemini", "model": "gemini-2.5-flash-lite", "speakers": 2,
                   "preprompt": "Write a natural two-host podcast conversation. Host 1 leads and explains; Host 2 reacts, asks sharp questions, adds humor. Interruptions, banter, real chemistry. Mark every line with [Speaker 1] or [Speaker 2]. No other formatting.",
                   "controls": {"temperature": 1.1, "max_tokens": 8192}},
        "voice": {"engine": "edge", "voices": {"1": "en-US-GuyNeural", "2": "en-US-JennyNeural"}, "controls": {"rate": 1.05, "pitch": 0, "volume": 100}},
        "images": {"model": "gemini-2.5-flash-image", "ratio": "16:9", "min_sec": 6, "max_sec": 15,
                   "preprompt": "Warm podcast-studio aesthetic or topical b-roll photography, soft light. No text, no watermarks."},
        "compile": {"resolution": "1920x1080", "motion": "fade", "subtitles": "embedded"},
    },
    "Story": {
        "script": {"provider": "gemini", "model": "gemini-2.5-flash-lite", "speakers": 1,
                   "preprompt": "Write an immersive narrated story. Strong atmosphere, sensory detail, rising tension, an emotional payoff. Spoken-word pacing with short punchy sentences at key moments. Plain prose only - no headings, no markdown.",
                   "controls": {"temperature": 1.2, "max_tokens": 8192}},
        "voice": {"engine": "edge", "voices": {"1": "en-GB-RyanNeural"}, "controls": {"rate": 0.95, "pitch": -5, "volume": 100}},
        "images": {"model": "gemini-2.5-flash-image", "ratio": "16:9", "min_sec": 5, "max_sec": 14,
                   "preprompt": "Painterly cinematic concept art, moody atmospheric lighting, rich color, storybook drama. No text, no watermarks."},
        "compile": {"resolution": "1920x1080", "motion": "kenburns_fade", "subtitles": "none"},
    },
    "YouTube Short": {
        "script": {"provider": "gemini", "model": "gemini-2.5-flash-lite", "speakers": 1,
                   "preprompt": "Write a script for a vertical short video under 60 seconds (about 130 words). Instant hook in the first sentence, fast pacing, punchy facts, end with a twist or question. Plain prose only.",
                   "controls": {"temperature": 1.0, "max_tokens": 1024}},
        "director": {"pacing": "frequent"},
        "voice": {"engine": "edge", "voices": {"1": "en-US-AriaNeural"}, "controls": {"rate": 1.2, "pitch": 0, "volume": 100}},
        "images": {"model": "gemini-2.5-flash-image", "ratio": "9:16", "min_sec": 2, "max_sec": 5,
                   "preprompt": "Bold high-contrast vertical composition, vivid saturated colors, attention-grabbing. No text, no watermarks."},
        "compile": {"resolution": "1080x1920", "motion": "kenburns", "subtitles": "embedded"},
    },
}

DEFAULT_SETTINGS = {
    "keys": {"openai": "", "claude": "", "gemini": "", "elevenlabs": "", "azure": "", "replicate": ""},
    "azure_region": "eastus",
    "script_format": "",  # custom tag-format prompt; blank = built-in default
    "usd_cad": 1.40,
    "price_overrides": {},
    "align_with_whisper": False,
}
