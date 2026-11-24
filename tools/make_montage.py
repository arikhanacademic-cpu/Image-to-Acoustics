"""Build a LinkedIn-ready 4:5 montage: the same voice line, dry and then in each room.

Requires the app to be running (./run.sh) and FFmpeg on PATH.

    python tools/make_montage.py --images path/to/rooms --voice "Brian" \
        --line "Hello? ... Is anyone there?" --credit "Your Name" --out montage.mp4

Each image's file name becomes its on-screen title (st-pauls-cathedral.jpg -> "St Pauls Cathedral").
Images are shown in alphabetical order; prefix with 1-, 2-, 3- to control the order.
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1350
BG, TEXT, MUTED, ACCENT = (23, 21, 28), (236, 232, 242), (154, 147, 167), (164, 147, 255)
IMG_H = 880
WAVE_Y, WAVE_H = 1085, 190
FONT_CANDIDATES = {
    "bold": ["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeuib.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "regular": ["C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
}


def font(kind: str, size: int):
    for p in FONT_CANDIDATES[kind]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def title_from(path: Path) -> str:
    stem = re.sub(r"^\d+[-_ .]*", "", path.stem)
    return re.sub(r"[-_]+", " ", stem).strip().title() or "Room"


def fail(msg: str):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def api(client: httpx.Client, method: str, path: str, **kw):
    r = client.request(method, path, **kw)
    if r.status_code >= 400:
        try:
            msg = r.json().get("error", r.text)
        except Exception:  # noqa: BLE001
            msg = r.text
        fail(f"{method} {path} failed ({r.status_code}): {msg}")
    return r


def frame(path_out: Path, title: str, subtitle: str, image: Path | None, credit: str, headline: str | None = None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    if image is not None:
        src = Image.open(image).convert("RGB")
        s = max(W / src.width, IMG_H / src.height)
        src = src.resize((round(src.width * s), round(src.height * s)), Image.LANCZOS)
        x0, y0 = (src.width - W) // 2, (src.height - IMG_H) // 2
        im.paste(src.crop((x0, y0, x0 + W, y0 + IMG_H)), (0, 0))
        fade = Image.new("L", (1, 220))
        for i in range(220):
            fade.putpixel((0, i), int(255 * (i / 219) ** 1.6))
        im.paste(Image.new("RGB", (W, 220), BG), (0, IMG_H - 220), fade.resize((W, 220)))
    else:
        f = font("bold", 76)
        y = 330
        for ln in (headline or title).split("\n"):
            w = d.textlength(ln, font=f)
            d.text(((W - w) / 2, y), ln, font=f, fill=TEXT)
            y += 96
    if image is not None:
        d.text((60, IMG_H - 20), title, font=font("bold", 66), fill=TEXT)
    d.text((60, IMG_H + 72 if image is not None else 860), subtitle, font=font("regular", 34), fill=MUTED)
    d.text((60, H - 62), "Image-to-Acoustics", font=font("bold", 26), fill=ACCENT)
    if credit:
        f = font("regular", 26)
        d.text((W - 60 - d.textlength(credit, font=f), H - 62), credit, font=f, fill=MUTED)
    im.save(path_out)


def segment(ffmpeg: str, png: Path, wav: Path, seconds: float, out: Path):
    """Still frame + the clip's waveform, drawn dim with a bright playhead fill (as in the app)."""
    fade = min(1.0, seconds / 3)
    afx = (f"aresample=48000,aformat=channel_layouts=stereo,atrim=0:{seconds:.3f},"
           f"afade=t=in:d=0.02,afade=t=out:st={seconds - fade:.3f}:d={fade:.3f}")
    pics = {}
    for name, col in (("dim", "0x4F4486"), ("lit", "0xA493FF")):
        pics[name] = out.with_name(out.stem + f"_{name}.png")
        r = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(wav), "-filter_complex",
                            f"{afx},pan=mono|c0=0.5*c0+0.5*c1,showwavespic=s={W - 120}x{WAVE_H}:colors={col}:scale=sqrt:draw=full",
                            "-frames:v", "1", str(pics[name])], capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"FFmpeg waveform failed on {out.name}: {r.stderr.strip()[-400:]}")
    flt = (f"[3:a]{afx}[a];"
           f"[1:v]format=rgba[d];[2:v]format=rgba[l];"
           f"[d][l]blend=all_expr='if(lt(X\\,W*T/{seconds:.3f})\\,B\\,A)'[wv];"
           f"[0:v][wv]overlay=60:{WAVE_Y}:format=auto,format=yuv420p[v]")
    cmd = [ffmpeg, "-v", "error", "-y", "-loop", "1", "-framerate", "30", "-i", str(png),
           "-loop", "1", "-framerate", "30", "-i", str(pics["dim"]), "-loop", "1", "-framerate", "30", "-i", str(pics["lit"]),
           "-i", str(wav), "-filter_complex", flt, "-map", "[v]", "-map", "[a]", "-t", f"{seconds:.3f}", "-r", "30",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "256k", "-ar", "48000", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"FFmpeg failed on {out.name}: {r.stderr.strip()[-400:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of room images (jpg/png/webp)")
    ap.add_argument("--voice", default="", help="ElevenLabs voice name (default: first voice)")
    ap.add_argument("--line", default="Hello? ... Is anyone there? I can hear my own voice coming back to me.")
    ap.add_argument("--credit", default="", help="name or handle shown bottom-right")
    ap.add_argument("--tail", type=float, default=2.2, help="seconds of room tail kept after the speech")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="montage.mp4")
    a = ap.parse_args()

    ffmpeg = shutil.which("ffmpeg") or fail("FFmpeg isn't on PATH.")
    images = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if not images:
        fail(f"No .jpg/.png/.webp images in {a.images}")
    out = Path(a.out).resolve()
    with httpx.Client(base_url=a.server, timeout=300) as c, tempfile.TemporaryDirectory() as td:
        td = Path(td)
        try:
            status = c.get("/api/status").json()
        except httpx.HTTPError:
            fail(f"The app isn't running at {a.server}. Start it with ./run.sh first.")
        voices = api(c, "GET", "/api/voices").json()["voices"]
        voice = next((v for v in voices if v["name"].lower().startswith(a.voice.lower())), None) if a.voice else voices[0]
        if voice is None:
            fail(f"No voice named '{a.voice}'. Available: {', '.join(v['name'] for v in voices[:30])}")
        print(f"Voice: {voice['name']}")
        sp = api(c, "POST", "/api/speech", json={"voice_id": voice["id"], "text": a.line}).json()
        print(f"Speech: {sp['duration']:.1f} s ({'cached, no credits used' if sp['cached'] else str(sp['characters']) + ' characters used'})")
        dry = td / "dry.wav"
        dry.write_bytes(api(c, "GET", sp["url"]).content)
        speech_len = sp["duration"]
        seg_len = speech_len + a.tail

        segments, report = [], []
        frame(td / "f0.png", "", "The original line, recorded dry.", None, a.credit,
              headline=f"One voice.\n{len(images)} places.\nSound on.")
        segment(ffmpeg, td / "f0.png", dry, speech_len + 0.6, td / "s0.mp4")
        segments.append(td / "s0.mp4")

        for i, img in enumerate(images, 1):
            title = title_from(img)
            files = {"image": (img.name, img.read_bytes())}
            data = {} if status.get("vision") else {"preset": img.stem.split("-", 1)[-1]}
            print(f"[{i}/{len(images)}] {title}: analysing...", flush=True)
            an = api(c, "POST", "/api/analyze", files=files, data=data).json()
            body = {"speech_id": sp["id"], "analysis_id": an["id"], "speaker": an["speaker"], "listener": an["listener"]}
            print(f"[{i}/{len(images)}] {title}: simulating the room...", flush=True)
            rr = api(c, "POST", "/api/render", json=body).json()
            wav = td / f"r{i}.wav"
            wav.write_bytes(api(c, "GET", rr["url"]).content)
            rt = rr["predicted"]["rt60_used"]
            rt_mid = (rt[2] + rt[3]) / 2
            rt_txt = f"{rt_mid:.1f} s" if rt_mid >= 1 else f"{round(rt_mid * 1000)} ms"
            sub = f"Reverb time {rt_txt}, estimated from this image"
            frame(td / f"f{i}.png", title, sub, img, a.credit)
            segment(ffmpeg, td / f"f{i}.png", wav, min(seg_len, sf.info(str(wav)).duration), td / f"s{i}.mp4")
            segments.append(td / f"s{i}.mp4")
            report.append({"room": title, "reverb_s": round(rt_mid, 2), "volume_m3": round(rr["predicted"]["volume"]),
                           "C50_dB": rr["measured"]["C50"], "DRR_dB": rr["measured"]["DRR"], "scene": an.get("scene")})

        frame(td / "fz.png", "", "Estimated from a single image. Simulated, not a preset.", None, a.credit,
              headline="Same voice.\nEvery room\nfrom a picture.")
        silence = td / "silence.wav"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "2.5", str(silence)], check=True)
        segment(ffmpeg, td / "fz.png", silence, 2.5, td / "sz.mp4")
        segments.append(td / "sz.mp4")

        lst = td / "list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in segments))
        r = subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                            "-movflags", "+faststart", str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"Joining segments failed: {r.stderr.strip()[-400:]}")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"\nDone: {out}  ({sum(1 for _ in segments)} segments)")
    for row in report:
        print(f"  {row['room']}: reverb {row['reverb_s']} s, volume {row['volume_m3']} m3")


if __name__ == "__main__":
    main()
