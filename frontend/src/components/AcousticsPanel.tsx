import type { Analysis, RenderParams, RenderResult } from "../api";
import { BANDS, fmtS, humanise } from "../api";

type Props = {
  analysis: Analysis | null;
  result: RenderResult | null;
  params: RenderParams | null;
  autoDistance: number | null;
  onChange: (patch: Partial<RenderParams>) => void;
  onReset: () => void;
};

const SURFACE_LABELS: Record<string, string> = {
  floor: "Floor", ceiling: "Ceiling", wall_x0: "Left wall", wall_x1: "Right wall", wall_y0: "Near wall", wall_y1: "Far wall",
};

export default function AcousticsPanel({ analysis, result, params, autoDistance, onChange, onReset }: Props) {
  const header = (
    <header className="flex items-center gap-2.5 px-4 pt-4 pb-3">
      <span className="step">3</span>
      <h2 className="text-[15px] font-semibold tracking-[-0.01em]">Acoustics</h2>
      {analysis && <button onClick={onReset} className="ml-auto text-xs text-muted hover:text-text underline decoration-line underline-offset-4">Reset to estimate</button>}
    </header>
  );
  if (!analysis || !params) {
    return (
      <section className="panel flex flex-col">
        {header}
        <p className="px-4 pb-5 text-sm text-muted leading-relaxed">
          Add an image to estimate the room: its size, what the surfaces are made of, and how long sound lingers in it.
        </p>
      </section>
    );
  }
  const pred = result?.predicted ?? analysis.predicted;
  const rtMid = (pred.rt60_used[2] + pred.rt60_used[3]) / 2;
  const room = analysis.room;
  const dimEdited = (k: "width" | "length" | "height") => Math.abs(params[k] - room[k].value) > 1e-6;

  return (
    <section className="panel flex flex-col min-h-0 overflow-y-auto">
      {header}
      <div className="px-4 grid grid-cols-2 gap-x-4 gap-y-3 pb-4">
        <Readout label="Reverb time" value={fmtS(rtMid)} note={pred.rt60_source === "user_override" ? "Set by you" : "Estimated"} />
        <Readout label="Room volume" value={`${Math.round(pred.volume).toLocaleString()} m³`} note="Estimated" />
        <Readout label="Speech clarity" value={result?.measured.C50 != null ? `${result.measured.C50.toFixed(1)} dB` : "–"} note="Measured, C50" />
        <Readout label="Direct to reverb" value={result?.measured.DRR != null ? `${result.measured.DRR.toFixed(1)} dB` : "–"} note="Measured" />
      </div>

      <RTChart predicted={pred.rt60_used} measured={result?.measured.T30 ?? null} />

      <Group title="Room">
        <div className="grid grid-cols-3 gap-2">
          {(["width", "length", "height"] as const).map((k) => (
            <label key={k} className="flex flex-col gap-1">
              <span className="text-2xs text-faint">{humanise(k)}{dimEdited(k) ? ", edited" : ""}</span>
              <span className="relative">
                <input type="number" min={k === "height" ? 1.8 : 1} max={300} step={0.5}
                       value={Number(params[k].toFixed(1))}
                       onChange={(e) => { const v = parseFloat(e.target.value); if (v > 0) onChange({ [k]: v }); }}
                       className={`field w-full pr-6 num ${dimEdited(k) ? "border-accent/50" : ""}`} />
                <span className="absolute right-2 top-1/2 -translate-y-1/2 text-2xs text-faint">m</span>
              </span>
            </label>
          ))}
        </div>
        <p className="text-2xs text-faint mt-1.5 num">
          {analysis.source === "ai"
            ? `Estimated from the image (${confRange(room)} confidence). Correct it if you know better.`
            : "Typical values for this kind of space. Adjust them to match your image."}
        </p>
        <dl className="mt-3 space-y-1.5">
          {Object.entries(SURFACE_LABELS).map(([k, label]) => (
            <div key={k} className="flex gap-3 text-xs">
              <dt className="w-[74px] shrink-0 text-faint">{label}</dt>
              <dd className="text-text/90">
                {room.surfaces[k].materials.map((m) => `${materialName(m.material)}${m.fraction < 0.99 ? ` ${Math.round(m.fraction * 100)}%` : ""}`).join(", ")}
              </dd>
            </div>
          ))}
          {room.objects.length > 0 && (
            <div className="flex gap-3 text-xs">
              <dt className="w-[74px] shrink-0 text-faint">Furnishing</dt>
              <dd className="text-text/90">{room.objects.map((o) => `${humanise(o.kind)} ×${Math.round(o.count)}`).join(", ")}</dd>
            </div>
          )}
        </dl>
      </Group>

      <Group title="Sound">
        <Slider label="Decay time" value={params.rt60 ?? rtMid} min={0.1} max={12} log
                display={params.rt60 == null ? `Auto, ${fmtS(rtMid)}` : fmtS(params.rt60)}
                onChange={(v) => onChange({ rt60: v })}
                onReset={params.rt60 != null ? () => onChange({ rt60: null }) : undefined} />
        <Slider label="Surface absorption" value={params.absorption_scale} min={0.25} max={4} log
                display={`${params.absorption_scale.toFixed(2)}×`} onChange={(v) => onChange({ absorption_scale: v })} />
        <Slider label="Early reflections" value={params.reflections} min={0} max={2}
                display={`${Math.round(params.reflections * 100)}%`} onChange={(v) => onChange({ reflections: v })} />
        <Slider label="Room level" value={params.wet} min={0} max={2}
                display={params.wet === 1 ? "Physical" : `${Math.round(params.wet * 100)}%`} onChange={(v) => onChange({ wet: v })} />
      </Group>

      <Group title="Voice and listener">
        <Slider label="Distance" value={params.distance ?? autoDistance ?? 3} min={0.3} max={Math.max(40, (autoDistance ?? 0) + 5)} log
                display={params.distance == null ? `From markers, ${(autoDistance ?? 0).toFixed(1)} m` : `${params.distance.toFixed(1)} m`}
                onChange={(v) => onChange({ distance: v })}
                onReset={params.distance != null ? () => onChange({ distance: null }) : undefined} />
        <Slider label="Obstruction" value={params.obstruction} min={0} max={1}
                display={params.obstruction < 0.02 ? "Clear line of sight" : params.obstruction < 0.5 ? "Partly blocked" : "Behind a wall or door"}
                onChange={(v) => onChange({ obstruction: v })} />
        <div className="flex items-center justify-between mt-2">
          <span className="text-xs text-muted">Voice faces</span>
          <div role="radiogroup" aria-label="Voice faces" className="flex rounded-lg bg-raised p-0.5 border border-line">
            {(["towards", "away"] as const).map((f) => (
              <button key={f} role="radio" aria-checked={params.facing === f} onClick={() => onChange({ facing: f })}
                      className={`px-3 py-1 text-xs rounded-md transition-colors ${params.facing === f ? "bg-panel text-text" : "text-muted hover:text-text"}`}>
                {f === "towards" ? "The listener" : "Away"}
              </button>
            ))}
          </div>
        </div>
      </Group>

      {(room.assumptions.length > 0 || analysis.corrections.length > 0) && (
        <details className="px-4 py-3 border-t border-line text-xs text-muted">
          <summary className="cursor-pointer hover:text-text">Assumptions behind the estimate</summary>
          <ul className="mt-2 space-y-1 list-disc pl-4 marker:text-faint">
            {room.assumptions.map((a, i) => <li key={`a${i}`}>{a}</li>)}
            {analysis.corrections.map((a, i) => <li key={`c${i}`} className="text-faint">Adjusted: {a}</li>)}
          </ul>
        </details>
      )}
    </section>
  );
}

const MATERIAL_NAMES: Record<string, string> = {
  carpet_thick: "Thick carpet", carpet_thin: "Thin carpet", curtains_heavy: "Heavy curtains",
  concrete_smooth: "Bare concrete", concrete_painted: "Painted concrete", stone_masonry: "Stone",
  rough_rock: "Rough rock", gypsum_board: "Plasterboard", wood_floor: "Wooden floor", wood_panelling: "Wood panelling",
  metal_sheet: "Sheet metal", glass_window: "Glass", acoustic_tile: "Acoustic ceiling tiles",
  acoustic_treatment: "Acoustic panels", grass_soil: "Grass and soil", forest_floor: "Forest floor",
  foliage_boundary: "Trees and foliage", ceramic_tile: "Ceramic tile", open: "Open air",
};
const materialName = (m: string) => MATERIAL_NAMES[m] ?? humanise(m);

function confRange(room: Analysis["room"]) {
  const c = [room.width, room.length, room.height].map((d) => Math.round(d.confidence * 100));
  const lo = Math.min(...c), hi = Math.max(...c);
  return lo === hi ? `${lo}%` : `${lo}–${hi}%`;
}

function Readout({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div>
      <div className="text-2xs text-faint">{label}</div>
      <div className="text-xl font-semibold tracking-[-0.02em] num leading-tight">{value}</div>
      <div className="text-2xs text-faint">{note}</div>
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="px-4 py-3.5 border-t border-line">
      <h3 className="text-sm font-medium mb-2.5">{title}</h3>
      {children}
    </div>
  );
}

function Slider({ label, value, min, max, log, display, onChange, onReset }: {
  label: string; value: number; min: number; max: number; log?: boolean; display: string;
  onChange: (v: number) => void; onReset?: () => void;
}) {
  const to = (v: number) => (log ? (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min)) : (v - min) / (max - min));
  const from = (t: number) => (log ? Math.exp(Math.log(min) + t * (Math.log(max) - Math.log(min))) : min + t * (max - min));
  const t = Math.min(Math.max(to(value), 0), 1);
  return (
    <div className="mb-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-xs text-muted">{label}</span>
        <span className="text-2xs text-text/85 num flex items-center gap-2">
          {display}
          {onReset && <button onClick={onReset} className="text-faint hover:text-text underline underline-offset-2">auto</button>}
        </span>
      </div>
      <input type="range" min={0} max={1000} value={Math.round(t * 1000)} aria-label={label} aria-valuetext={display}
             style={{ ["--p" as string]: `${t * 100}%` }}
             onChange={(e) => onChange(Number(from(Number(e.target.value) / 1000).toFixed(3)))} />
    </div>
  );
}

function RTChart({ predicted, measured }: { predicted: number[]; measured: (number | null)[] | null }) {
  const W = 280, H = 96, pad = 18;
  const all = [...predicted, ...(measured ?? []).filter((v): v is number => v != null)];
  const lo = Math.log(Math.max(Math.min(...all) * 0.7, 0.03)), hi = Math.log(Math.max(...all) * 1.3);
  const y = (v: number) => H - pad - ((Math.log(v) - lo) / (hi - lo)) * (H - pad - 6);
  const x = (i: number) => 8 + (i + 0.5) * ((W - 16) / 7);
  return (
    <figure className="px-4 pb-4">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
           aria-label={`Reverb time per frequency band: ${predicted.map((v, i) => `${BANDS[i]} Hz ${fmtS(v)}`).join(", ")}`}>
        {predicted.map((v, i) => (
          <rect key={i} x={x(i) - 9} y={y(v)} width={18} height={H - pad - y(v)} rx={2} fill="#2F2B38" />
        ))}
        {measured?.map((v, i) => v != null && <circle key={i} cx={x(i)} cy={y(v)} r={3.2} fill="#A493FF" />)}
        {BANDS.map((b, i) => (
          <text key={b} x={x(i)} y={H - 4} textAnchor="middle" fontSize="9" fill="#6E677B">{b}</text>
        ))}
      </svg>
      <figcaption className="flex items-center gap-4 text-2xs text-faint mt-1">
        <span className="flex items-center gap-1.5"><span className="h-2 w-2.5 rounded-sm bg-line" />Reverb time estimated from the image</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-accent" />Measured on the result</span>
      </figcaption>
    </figure>
  );
}
