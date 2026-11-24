# Image-to-Acoustics

Pick an ElevenLabs voice, type a line, and drop in a picture of a place: a
photo, a film still or a game scene. The app estimates the room's physical
acoustics from the image, then simulates the space so you hear the voice as if
it had been recorded there. You compare the result against the original and
download either as WAV or MP3.

The acoustics come from physics, not reverb presets. The image becomes room
dimensions, surface materials and furnishings. From those, a room simulation
builds an impulse response with early reflections, a frequency-dependent decay,
distance, direction and obstruction. The voice is only convolved, never
regenerated: pronunciation and delivery are untouched. `MODEL_NOTES.md` has the
maths.

## Run it

Requirements: Python 3.10+, Node 18+, FFmpeg.

```bash
cp .env.example .env      # add ELEVENLABS_API_KEY and (optionally) VISION_PROVIDER + VISION_API_KEY
./run.sh                  # creates .venv, installs, builds the UI, serves http://localhost:8000
```

Without keys the app runs in demo mode: an offline sample voice, and you pick
the closest of ten reference environments instead of AI analysis.

**Windows (PowerShell):** from the project root, copy `.env.example` to `.env`
and fill in your own API keys locally, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
cd frontend
npm.cmd ci
npm.cmd run build
cd ..\backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

**Demo montage:** with the server running and both API integrations configured,
run from the project root in a second terminal:

```powershell
.\.venv\Scripts\python.exe tools\make_montage.py --images demo-rooms --voice "George" --credit "Ari Khan" --out montage.mp4
```

This produces a video of the same generated voice across five AI-generated
environment images, with estimated acoustic metrics. The images are illustrative;
the acoustics are simulations, not measurements of real rooms.

**Development** (hot reload), in two terminals:

```bash
cd backend && uvicorn app.main:app --reload --port 8000
cd frontend && npm install && npm run dev      # http://localhost:5173, proxies /api
```

**Tests:**

```bash
cd backend && python -m pytest -q
```

There are 71 tests. They cover the acoustic engine and the API with ElevenLabs
and the vision provider mocked: caching, credit safeguards, every error path,
key secrecy, upload validation and end-to-end rendering.

## The flow

1. **Voice.** Choose a voice (with preview) and write the dialogue.
   "Generate speech" calls ElevenLabs once. The clip is cached by voice, model,
   format and text, so regenerating the same line or changing the environment
   costs no credits. You can also use your own recording.
2. **Environment.** Drop an image. The vision model returns dimensions,
   per-surface materials, furnishings, openings, likely talker and listener
   positions, assumptions and confidences. Its answer is validated: unknown
   materials are mapped or replaced, fractions are normalised, values are
   clamped, and every adjustment is listed under "Assumptions behind the
   estimate".
3. **Acoustics.** Values estimated from the image are labelled separately from
   values measured on the rendered result. You can correct the room size,
   decay time, absorption, reflections, room level, distance, obstruction and
   which way the voice faces. Drag the markers on the image to move the voice
   and the listener.
4. **Compare.** Playback is synchronised: press **A** for the original and
   **B** for the voice in this room, and **Space** to play or pause. Both clips
   are loudness-matched, so you hear the space rather than a volume change.
   Download either clip as WAV (24-bit) or MP3.

## Credentials and safeguards

- Keys live only in the server's `.env` and are never sent to the browser.
  The tests check that no response contains them.
- There's a per-clip character cap (`MAX_TTS_CHARS`), and the remaining
  ElevenLabs quota is checked *before* generating.
- Only one generation runs per cache key, so double clicks don't cost twice.
  Analyses are cached by image hash, and renders and impulse responses are
  cached too.
- ElevenLabs and vision errors (bad key, quota, rate limit, unreachable,
  malformed answer) become plain messages that say what to do.

## Layout

```
backend/
  acoustics/      physics engine (Phase 1, unchanged API + loudness-matched output)
  app/
    main.py       FastAPI routes
    tts.py        ElevenLabs SDK: voices, credits, cached synthesis, error mapping
    vision.py     prompt, Anthropic/OpenAI calls, validation and normalisation
    placement_map.py  image coordinates <-> room coordinates
    storage.py    content-addressed files
  tests/
frontend/src/
  App.tsx         state and render orchestration (debounced, abortable)
  components/     VoicePanel, Stage, AcousticsPanel, Compare
```

## Known limits

- A single image cannot fix scale. Dimensions are estimates with confidences,
  and distance from the markers is a simple perspective guess. Use the
  numeric fields and the distance slider when you know better.
- Rooms are simulated as boxes (see `MODEL_NOTES.md`). Big vaults and caves are
  approximated with extra scattering.
- Very reverberant rooms take a few seconds to render. Changes to room level
  and reflections reuse the cached impulse response and are quick.
