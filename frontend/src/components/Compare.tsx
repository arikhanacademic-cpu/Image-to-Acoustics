import { useCallback, useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import type { RenderResult } from "../api";
import { PauseIcon, PlayIcon, Spinner } from "./VoicePanel";

type Props = {
  originalUrl: string | null;
  result: RenderResult | null;
  rendering: boolean;
  error: string | null;
  sceneName: string;
  downloadBase: string;
};
type Side = "original" | "room";

const WAVE = {
  original: { waveColor: "#4A4457", progressColor: "#9A93A7" },
  room: { waveColor: "#4F4486", progressColor: "#A493FF" },
};

export default function Compare({ originalUrl, result, rendering, error, sceneName, downloadBase }: Props) {
  const elA = useRef<HTMLDivElement>(null), elB = useRef<HTMLDivElement>(null);
  const ws = useRef<Record<Side, WaveSurfer | null>>({ original: null, room: null });
  const [active, setActive] = useState<Side>("room");
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [dur, setDur] = useState(0);
  const [durA, setDurA] = useState(0);
  const activeRef = useRef(active); activeRef.current = active;
  const playingRef = useRef(playing); playingRef.current = playing;

  useEffect(() => {
    const make = (el: HTMLDivElement, side: Side) => WaveSurfer.create({
      container: el, height: 48, barWidth: 2, barGap: 1, barRadius: 2, cursorColor: "#ECE8F2", cursorWidth: 1,
      normalize: false, dragToSeek: true, ...WAVE[side],
    });
    const a = make(elA.current!, "original"), b = make(elB.current!, "room");
    ws.current = { original: a, room: b };
    const sync = (from: WaveSurfer, to: WaveSurfer) => from.on("interaction", (t: number) => to.setTime(Math.min(t, to.getDuration() || t)));
    sync(a, b); sync(b, a);
    b.on("timeupdate", (t: number) => setTime(t));
    b.on("ready", (d: number) => setDur(d));
    a.on("ready", (d: number) => setDurA(d));
    b.on("finish", () => { a.pause(); setPlaying(false); });
    a.on("finish", () => { if (activeRef.current === "original") { b.pause(); setPlaying(false); } });
    return () => { a.destroy(); b.destroy(); };
  }, []);

  useEffect(() => {
    const { original, room } = ws.current;
    original?.setVolume(active === "original" ? 1 : 0);
    room?.setVolume(active === "room" ? 1 : 0);
  }, [active]);

  const loadKeep = useCallback(async (side: Side, url: string | null) => {
    const w = ws.current[side];
    if (!w) return;
    if (!url) { w.empty(); return; }
    const other = ws.current[side === "room" ? "original" : "room"];
    const t = other?.getCurrentTime() ?? 0;
    const wasPlaying = playingRef.current;
    if (wasPlaying) { other?.pause(); }
    await w.load(url).catch(() => undefined);
    w.setVolume(activeRef.current === side ? 1 : 0);
    w.setTime(Math.min(t, w.getDuration()));
    if (wasPlaying) { other?.setTime(t); other?.play(); w.play(); }
  }, []);

  useEffect(() => { void loadKeep("original", originalUrl); }, [originalUrl, loadKeep]);
  useEffect(() => { void loadKeep("room", result?.url ?? null); }, [result?.url, loadKeep]);

  const ready = !!originalUrl && !!result;
  const toggle = useCallback(() => {
    const { original, room } = ws.current;
    if (!original || !room || !ready) return;
    if (playingRef.current) { original.pause(); room.pause(); setPlaying(false); return; }
    const t = room.getCurrentTime() >= room.getDuration() - 0.05 ? 0 : room.getCurrentTime();
    room.setTime(t); original.setTime(Math.min(t, original.getDuration()));
    void original.play(); void room.play(); setPlaying(true);
  }, [ready]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement)?.isContentEditable) return;
      if (e.code === "Space" && tag !== "BUTTON") { e.preventDefault(); toggle(); }
      if (e.key === "a" || e.key === "A") setActive("original");
      if (e.key === "b" || e.key === "B") setActive("room");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);

  const dl = (side: Side, ext: "wav" | "mp3") => {
    const url = side === "original" ? originalUrl : result?.url;
    if (!url) return undefined;
    const name = `${downloadBase}${side === "room" ? `-${slug(sceneName)}` : "-original"}`;
    return `${url.replace(/\.wav$/, `.${ext}`)}?download=${encodeURIComponent(name)}`;
  };

  return (
    <section className="panel" aria-labelledby="cmp-h">
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 pt-3 pb-1.5">
        <span className="step">4</span>
        <h2 id="cmp-h" className="text-[15px] font-semibold tracking-[-0.01em]">Compare</h2>
        <span className="text-2xs text-faint" aria-live="polite">
          {rendering ? "" : result ? `Levels matched, so you hear the room rather than a volume change.` : ""}
        </span>
        <span className="ml-auto flex items-center gap-2 text-xs text-muted">
          {rendering && <><Spinner /> Simulating the room</>}
          {!rendering && result && <span className="num text-faint">{result.cached ? "Ready" : `Rendered in ${result.seconds.toFixed(1)} s`}</span>}
        </span>
      </header>

      <div className="px-4 pb-3 grid grid-cols-[auto_1fr] gap-x-4 items-center">
        <div className="row-span-2 flex flex-col items-center gap-2 self-stretch justify-center pr-1">
          <button onClick={toggle} disabled={!ready} aria-label={playing ? "Pause" : "Play"}
                  className="h-12 w-12 rounded-full bg-text text-ink flex items-center justify-center disabled:opacity-30 hover:bg-white transition-colors">
            <span className="scale-[1.6]">{playing ? <PauseIcon /> : <PlayIcon />}</span>
          </button>
          <span className="text-2xs text-faint num">{fmt(time)} / {fmt(dur)}</span>
        </div>

        {(["original", "room"] as Side[]).map((side) => (
          <div key={side}
               className={`relative flex items-center gap-3 rounded-lg px-3 py-1 my-0.5 transition-colors border
                 ${active === side ? "bg-raised border-line" : "border-transparent opacity-60 hover:opacity-90"}`}>
            <button onClick={() => setActive(side)} aria-pressed={active === side}
                    className="w-[72px] sm:w-[108px] shrink-0 text-left">
              <span className={`block text-sm ${active === side ? "text-text" : "text-muted"}`}>
                {side === "original" ? "Original" : "In this room"}
              </span>
              <span className="block text-2xs text-faint">
                {side === "original" ? "Press A" : "Press B"}
              </span>
            </button>
            <div className="flex-1 min-w-0">
              {/* Same time scale on both rows: the original is drawn at its true length, so the room's tail shows. */}
              <div ref={side === "original" ? elA : elB} onPointerDown={() => setActive(side)}
                   style={side === "original" && result && dur > 0 && durA > 0 ? { width: `${Math.min(durA / dur, 1) * 100}%` } : undefined}
                   className={side === "room" && !result ? "opacity-0" : ""} />
            </div>
            {side === "room" && !result && (
              <span className="absolute inset-y-0 left-[96px] sm:left-[132px] right-3 flex items-center text-sm text-faint">
                {error ? <span className="text-danger" role="alert">{error}</span>
                  : rendering ? "Simulating reflections and decay…"
                  : "Generate speech and add an image to hear it in the space."}
              </span>
            )}
            <span className="flex flex-col sm:flex-row shrink-0 gap-1">
              {(["wav", "mp3"] as const).map((ext) => {
                const href = dl(side, ext);
                return href ? (
                  <a key={ext} href={href} download
                     className="rounded-md border border-line px-2 py-1 text-2xs text-muted hover:text-text hover:border-faint">
                    {ext.toUpperCase()}
                  </a>
                ) : <span key={ext} className="rounded-md border border-line/50 px-2 py-1 text-2xs text-faint/50">{ext.toUpperCase()}</span>;
              })}
            </span>
          </div>
        ))}
      </div>
      {result && error && <p role="alert" className="px-4 pb-3 -mt-2 text-xs text-danger">{error}</p>}
    </section>
  );
}

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "room";
