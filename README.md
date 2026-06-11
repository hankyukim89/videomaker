# AI Video Maker

Fully local, fully customizable AI video maker. Script → Director → Voice → Images → Video, with full control at every step and live CAD cost display.

## Run it

Double-click **run.bat** (needs Python 3.9+). First run installs packages, then the app opens at http://127.0.0.1:8765. Your API keys are in `settings.json` (editable in Settings).

## Workflow

1. **Script** — pick provider (OpenAI / Claude / Gemini) and model, topic, style pre-prompt, temperature etc. Choose **image pacing**: *Normal* (varied holds) or *Frequent* (picture changes every ~3s, like fast video essays — more images, costs more). With 2+ speakers you get persona cards. Script is editable.
2. **Director** (automatic) — one cheap AI pass plans the whole video: *voice segments* (with acting directions) and *image beats*. Beats come in two kinds: **NEW scene** (fresh image) and **✎ EDIT** (Nano Banana modifies an earlier image — change an expression, add a prop, zoom in — so scenes evolve with the narration instead of jumping around). The director also places **sound effects** from the 50-sound library (`sfx/` folder — see `sfx/SFX_LIST.md` for the file names to download), writes a continuity bible, title, thumbnail prompt, and YouTube chapters. Everything editable.
3. **Voice** — **Edge TTS** (free, word timestamps), **Chatterbox Turbo** (free, runs locally, voice cloning from a short clip, expression tags like `[laugh]` — install once: `.venv\Scripts\pip install chatterbox-tts`), **Azure Speech** (word timestamps, speaking styles), or **ElevenLabs** (word timestamps, emotion tags) — or record/upload your own. Audio is generated *per segment*: re-roll any one for pennies; edits mark segments stale.
4. **Images** — Gemini **Nano Banana** models only: `gemini-2.5-flash-image` (cheap), `gemini-3.1-flash-image` (Nano Banana 2 — up to 14 reference images, 512–4K, Google **Search + Image Search grounding**), `gemini-3-pro-image` (Nano Banana Pro — highest quality). All aspect ratios incl. 1:1, 16:9, 9:16, 21:9. Tick **Ground with Google Search** for news/real-time topics. EDIT beats automatically feed the base image back in and apply only the change. Reroll keeps every version; per-beat SFX selectable.
5. **Compile** — resolution (incl. vertical), FPS, motion, image scaling, subtitles, **background music** with auto-ducking, **sound effects mixing** (toggle + volume), and a missing-image policy. Downloads: everything or per-asset.

## Projects & profiles

Everything lives in `projects/<name>/`. Profiles snapshot all settings; built-ins: Documentary, Video essay (frequent pacing), Podcast, Story, YouTube Short.

## Folders next to the app

- `sfx/` — the 50 named sound effects (see `SFX_LIST.md` inside). Missing files are skipped silently.
- `voices/` — short reference clips (5–20s) for Chatterbox voice cloning; add via the Voice tab too.

## Settings

API keys (Gemini, OpenAI, Claude, ElevenLabs, **Azure Speech key + region**), USD→CAD rate, editable pricing table, optional Whisper word-alignment (for engines without native timestamps, e.g. Chatterbox).

## Notes

- ffmpeg is bundled via `imageio-ffmpeg` — nothing to install.
- Edge TTS is free, needs no key, has exact word timestamps — best for testing.
- Word-level timestamps (Edge / Azure / ElevenLabs) give precise image cut timing; Chatterbox uses estimates unless Whisper alignment is on.
- Ken Burns motion requires "Fill screen" scaling. Edit-chain beats default to static so the picture appears to change in place.
- Claude appears only in Script/Director (no TTS/image APIs).
