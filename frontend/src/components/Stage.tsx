import { useEffect, useRef, useState } from "react";
import type { Analysis, Pt, Status } from "../api";
import { Spinner } from "./VoicePanel";

type Props = {
  status: Status | null;
  imageUrl: string | null;
  analysis: Analysis | null;
  analysing: boolean;
  needsPreset: boolean;
  error: string | null;
  speaker: Pt | null;
  listener: Pt | null;
  facing: "towards" | "away";
  distance: number | null;
  obstruction: number;
  onFile: (f: File) => void;
  onPreset: (id: string) => void;
  onMove: (which: "speaker" | "listener", p: Pt) => void;
};

export default function Stage(p: Props) {
  const [over, setOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const area = useRef<HTMLDivElement>(null);
  const [natural, setNatural] = useState<[number, number] | null>(null);
  const [fit, setFit] = useState<[number, number] | null>(null);

  // Fit the image (and its marker overlay) to whatever space the panel has.
  useEffect(() => {
    const el = area.current;
    if (!el || !natural) return;
    const update = () => {
      const { width, height } = el.getBoundingClientRect();
      // Stacked (mobile) layout: the image sets the height. Desktop: fit inside the panel.
      const h = window.innerWidth < 1024 ? (width * natural[1]) / natural[0] : height;
      const s = Math.min(width / natural[0], h / natural[1]);
      setFit([Math.floor(natural[0] * s), Math.floor(natural[1] * s)]);
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [natural, p.imageUrl]);

  const pick = (files: FileList | null) => { const f = files?.[0]; if (f) p.onFile(f); };

  const header = (
    <header className="flex items-center gap-2.5 px-4 pt-4 pb-3">
      <span className="step">2</span>
      <h2 className="text-[15px] font-semibold tracking-[-0.01em]">Environment</h2>
      {p.imageUrl && (
        <button className="ml-auto text-xs text-muted hover:text-text underline decoration-line underline-offset-4"
                onClick={() => fileRef.current?.click()}>Replace image</button>
      )}
      <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp" className="hidden"
             onChange={(e) => { pick(e.target.files); e.target.value = ""; }} />
    </header>
  );

  if (!p.imageUrl) {
    return (
      <section className="panel flex flex-col min-h-[420px]">
        {header}
        <button type="button"
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files); }}
          className={`mx-4 mb-4 flex-1 rounded-lg border border-dashed flex flex-col items-center justify-center gap-3 text-center px-6
            transition-colors ${over ? "border-accent bg-accent/5" : "border-line hover:border-faint"}`}>
          <RoomGlyph />
          <span className="text-lg font-medium tracking-[-0.01em]">Drop an image of a place</span>
          <span className="text-sm text-muted max-w-[34ch] leading-relaxed">
            A photo, a film still or a game scene. The room’s size and surfaces are estimated from it.
          </span>
          <span className="text-2xs text-faint">JPEG, PNG or WebP, up to 12 MB</span>
          {p.error && <span role="alert" className="text-xs text-danger">{p.error}</span>}
        </button>
      </section>
    );
  }

  const drag = (which: "speaker" | "listener") => (e: React.PointerEvent) => {
    const el = e.currentTarget as HTMLElement;
    el.setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent) => {
      const r = box.current!.getBoundingClientRect();
      p.onMove(which, { x: clamp((ev.clientX - r.left) / r.width), y: clamp((ev.clientY - r.top) / r.height) });
    };
    const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
  };
  const nudge = (which: "speaker" | "listener", pt: Pt) => (e: React.KeyboardEvent) => {
    const d = e.shiftKey ? 0.05 : 0.01;
    const m: Record<string, [number, number]> = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, -d], ArrowDown: [0, d] };
    if (!m[e.key]) return;
    e.preventDefault();
    p.onMove(which, { x: clamp(pt.x + m[e.key][0]), y: clamp(pt.y + m[e.key][1]) });
  };

  const s = p.speaker, l = p.listener;
  const a = p.analysis;
  return (
    <section className="panel flex flex-col min-h-0">
      {header}
      <div className="px-4 flex-1 lg:min-h-0 flex">
      <div ref={area} className="relative flex-1" style={{ minHeight: fit && typeof window !== "undefined" && window.innerWidth < 1024 ? fit[1] : 200 }}>
        <div ref={box} className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 select-none touch-none overflow-hidden rounded-lg"
             style={fit ? { width: fit[0], height: fit[1] } : { visibility: "hidden" }}>
          <img src={p.imageUrl} alt={a?.scene || "Uploaded environment"} draggable={false}
               onLoad={(e) => setNatural([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
               className={`block h-full w-full transition-[filter] duration-500
                 ${p.analysing ? "brightness-[.55] saturate-50" : ""}`} />
          {p.analysing && (
            <div className="absolute inset-0 pointer-events-none" aria-hidden>
              <div className="scanline absolute inset-x-0 top-0 h-[10%] bg-gradient-to-b from-transparent via-accent/25 to-transparent border-b border-accent/60" />
            </div>
          )}
          {s && l && !p.analysing && (
            <>
              <svg className="absolute inset-0 h-full w-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
                <line x1={s.x * 100} y1={s.y * 100} x2={l.x * 100} y2={l.y * 100}
                      stroke="#A493FF" strokeWidth="1.5" strokeDasharray={p.obstruction > 0.05 ? "2 5" : "6 4"}
                      vectorEffect="non-scaling-stroke" strokeOpacity={1 - p.obstruction * 0.5} />
              </svg>
              {p.distance != null && (
                <span className="absolute -translate-x-1/2 -translate-y-1/2 rounded-md bg-ink/85 px-2 py-0.5 text-2xs num text-text pointer-events-none border border-line"
                      style={{ left: `${((s.x + l.x) / 2) * 100}%`, top: `${((s.y + l.y) / 2) * 100}%` }}>
                  {p.distance.toFixed(1)} m
                </span>
              )}
              <Marker pt={s} label="Voice" filled angle={Math.atan2((l.y - s.y), (l.x - s.x)) + (p.facing === "away" ? Math.PI : 0)}
                      onPointerDown={drag("speaker")} onKeyDown={nudge("speaker", s)} />
              <Marker pt={l} label="Listener" onPointerDown={drag("listener")} onKeyDown={nudge("listener", l)} />
            </>
          )}
          {p.analysing && (
            <div className="absolute inset-x-0 bottom-0 p-4 flex items-center gap-2 text-sm">
              <Spinner /> Estimating size, surfaces and furnishings
            </div>
          )}
          {p.needsPreset && !p.analysing && (
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-ink via-ink/90 to-transparent p-4 pt-10">
              <p className="text-sm mb-2.5">AI analysis isn’t configured on this server. Which reference space is closest?</p>
              <div className="flex flex-wrap gap-1.5">
                {p.status?.presets.map((pr) => (
                  <button key={pr.id} onClick={() => p.onPreset(pr.id)}
                          className="rounded-md border border-line bg-panel/90 px-2.5 py-1 text-xs hover:border-accent hover:text-text text-muted">
                    {pr.name}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      </div>
      <div className="px-4 pt-3 pb-4">
        {p.error && <p role="alert" className="text-sm text-danger">{p.error}</p>}
        {a && !p.analysing && (
          <div className="flex items-start gap-3">
            <p className="text-sm leading-relaxed text-text/90 flex-1">{a.scene}</p>
            <span className="shrink-0 text-2xs text-faint text-right leading-4 num">
              {a.source === "ai" ? <>AI estimate<br />{Math.round(a.overall_confidence * 100)}% confidence</> : <>Reference<br />environment</>}
            </span>
          </div>
        )}
        {a && !p.analysing && (
          <p className="mt-1 text-2xs text-faint">Drag the markers to place the voice and the listener. Arrow keys nudge them.</p>
        )}
      </div>
    </section>
  );
}

function Marker(props: { pt: Pt; label: string; filled?: boolean; angle?: number;
  onPointerDown: (e: React.PointerEvent) => void; onKeyDown: (e: React.KeyboardEvent) => void }) {
  const { pt, label, filled, angle } = props;
  return (
    <div role="slider" tabIndex={0} aria-label={`${label} position`}
         aria-valuetext={`${Math.round(pt.x * 100)}% across, ${Math.round(pt.y * 100)}% down`}
         onPointerDown={props.onPointerDown} onKeyDown={props.onKeyDown}
         className="absolute -translate-x-1/2 -translate-y-1/2 cursor-grab active:cursor-grabbing"
         style={{ left: `${pt.x * 100}%`, top: `${pt.y * 100}%` }}>
      <span className={`relative block h-5 w-5 rounded-full border-2 shadow-[0_0_0_4px_rgba(23,21,28,.55)]
                        ${filled ? "bg-accent border-text" : "bg-ink border-accent"}`}>
        {angle != null && (
          <span className="absolute left-1/2 top-1/2 h-0 w-0"
                style={{ transform: `rotate(${angle}rad) translateX(13px)` }}>
            <svg width="8" height="10" viewBox="0 0 8 10" className="-translate-y-1/2" aria-hidden><path d="M0 0 L8 5 L0 10 Z" fill="#ECE8F2" /></svg>
          </span>
        )}
      </span>
      <span className="absolute left-full top-1/2 -translate-y-1/2 ml-3 whitespace-nowrap rounded bg-ink/85 px-1.5 text-2xs text-text border border-line pointer-events-none">{label}</span>
    </div>
  );
}

const clamp = (v: number) => Math.min(Math.max(v, 0.02), 0.98);

const RoomGlyph = () => (
  <svg width="64" height="44" viewBox="0 0 64 44" aria-hidden className="text-faint">
    <path d="M4 40 L18 28 L46 28 L60 40 M18 28 L18 6 L46 6 L46 28 M4 4 L18 6 M60 4 L46 6" stroke="currentColor" strokeWidth="1.5" fill="none" />
    <circle cx="28" cy="20" r="2.5" fill="#A493FF" />
    <path d="M33 15 a7 7 0 0 1 0 10 M37 12 a11 11 0 0 1 0 16" stroke="#A493FF" strokeWidth="1.5" fill="none" strokeLinecap="round" />
  </svg>
);
