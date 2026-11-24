export type Pt = { x: number; y: number };
export type Credits = { used: number; limit: number; remaining: number; resets_at: number | null };
export type Status = {
  elevenlabs: boolean; vision: boolean; vision_provider: string; max_chars: number;
  credits: Credits | null; presets: { id: string; name: string }[];
};
export type Voice = {
  id: string; name: string; category: string; description: string;
  labels: Record<string, string>; preview_url: string | null;
};
export type Speech = {
  id: string; url: string; cached: boolean; characters: number; duration: number;
  sample_rate: number; demo: boolean; credits?: Credits | null;
};
export type Dim = { value: number; confidence: number; provenance: string };
export type Surface = { materials: { material: string; fraction: number }[]; confidence: number; provenance: string };
export type Room = {
  name: string; enclosure: string; width: Dim; length: Dim; height: Dim;
  surfaces: Record<string, Surface>; objects: { kind: string; count: number; confidence: number }[];
  clutter_scattering: number; assumptions: string[];
};
export type Predicted = {
  volume: number; rt60_eyring: number[]; rt60_sabine: number[]; rt60_used: number[];
  rt60_source: string; mean_alpha: number[]; mixing_time: number;
};
export type Analysis = {
  id: string; image_url: string; image_size: [number, number]; scene: string; style: string;
  enclosure: string; room: Room; speaker: Pt; listener: Pt; obstruction: number; openings: string[];
  overall_confidence: number; corrections: string[]; predicted: Predicted; source: "ai" | "preset";
  provider: string; cached: boolean; preset?: string;
};
export type RenderResult = {
  id: string; url: string; duration: number; seconds: number; distance: number; dims: number[];
  predicted: Predicted; critical_distance: number; cached: boolean;
  measured: { T30: (number | null)[]; EDT: (number | null)[]; C50: number | null; C80: number | null; DRR: number | null };
};
export type RenderParams = {
  width: number; length: number; height: number; rt60: number | null; absorption_scale: number;
  speaker: Pt; listener: Pt; facing: "towards" | "away"; obstruction: number; distance: number | null;
  wet: number; reflections: number;
};

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new Error("Can't reach the server. Is the backend running on port 8000?");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body as T;
}

const json = (b: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b),
});

export const api = {
  status: () => call<Status>("/api/status"),
  voices: () => call<{ voices: Voice[]; demo: boolean }>("/api/voices"),
  speech: (voice_id: string, text: string) => call<Speech>("/api/speech", json({ voice_id, text })),
  uploadSpeech: (file: File) => {
    const f = new FormData();
    f.append("file", file);
    return call<Speech>("/api/speech/upload", { method: "POST", body: f });
  },
  analyze: (image: File, preset?: string) => {
    const f = new FormData();
    f.append("image", image);
    if (preset) f.append("preset", preset);
    return call<Analysis>("/api/analyze", { method: "POST", body: f });
  },
  render: (speech_id: string, analysis_id: string, p: RenderParams, signal?: AbortSignal) =>
    call<RenderResult>("/api/render", { ...json({ speech_id, analysis_id, ...p }), signal }),
};

/* Mirror of backend app/placement_map.py, for live distance while dragging. */
export function normToRoom(p: Pt, dims: number[], z: number): [number, number, number] {
  const [W, L, H] = dims;
  const depth = Math.min(Math.max((1 - p.y) / 0.7, 0), 1);
  return [Math.min(Math.max(p.x, 0.03), 0.97) * W, (0.08 + 0.84 * depth) * L, Math.min(z, H - 0.2)];
}
export function markerDistance(a: Pt, b: Pt, dims: number[]): number {
  const s = normToRoom(a, dims, 1.65), l = normToRoom(b, dims, 1.55);
  return Math.hypot(s[0] - l[0], s[1] - l[1], s[2] - l[2]);
}

export const BANDS = ["125", "250", "500", "1k", "2k", "4k", "8k"];
export const humanise = (s: string) => s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
export const fmtS = (v: number | null | undefined) =>
  v == null ? "–" : v >= 10 ? `${v.toFixed(0)} s` : v >= 1 ? `${v.toFixed(1)} s` : `${Math.round(v * 1000)} ms`;
