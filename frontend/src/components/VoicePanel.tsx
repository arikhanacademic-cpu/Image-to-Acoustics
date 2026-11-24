import { useMemo, useRef, useState } from "react";
import type { Speech, Status, Voice } from "../api";

type Props = {
  status: Status | null;
  voices: Voice[];
  voiceId: string;
  onVoice: (id: string) => void;
  text: string;
  onText: (t: string) => void;
  speech: Speech | null;
  speechText: string | null;
  busy: boolean;
  error: string | null;
  onGenerate: () => void;
  onUpload: (f: File) => void;
};

export default function VoicePanel(p: Props) {
  const [query, setQuery] = useState("");
  const [playing, setPlaying] = useState<string | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const max = p.status?.max_chars ?? 600;
  const over = p.text.length > max;
  const credits = p.status?.credits;
  const upToDate = p.speech && p.speechText === p.text.trim();

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return p.voices;
    return p.voices.filter((v) =>
      [v.name, v.description, ...Object.values(v.labels)].join(" ").toLowerCase().includes(q));
  }, [p.voices, query]);

  const preview = (v: Voice) => {
    if (!v.preview_url) return;
    audio.current?.pause();
    if (playing === v.id) return setPlaying(null);
    const a = new Audio(v.preview_url);
    a.onended = () => setPlaying(null);
    a.play().catch(() => setPlaying(null));
    audio.current = a;
    setPlaying(v.id);
  };

  return (
    <section className="panel flex flex-col min-h-0" aria-labelledby="voice-h">
      <header className="flex items-center gap-2.5 px-4 pt-4 pb-3">
        <span className="step">1</span>
        <h2 id="voice-h" className="text-[15px] font-semibold tracking-[-0.01em]">Voice</h2>
      </header>

      {p.voices.length > 8 && (
        <div className="px-4 pb-2">
          <input className="field w-full" placeholder="Search voices" value={query}
                 onChange={(e) => setQuery(e.target.value)} aria-label="Search voices" />
        </div>
      )}

      <ul role="listbox" aria-label="Voices" className="mx-2 max-h-[232px] overflow-y-auto pr-1">
        {shown.map((v) => {
          const sel = v.id === p.voiceId;
          const tags = [v.labels.accent, v.labels.gender, v.labels.age, v.labels.descriptive ?? v.labels.description]
            .filter(Boolean).slice(0, 3).join(", ");
          return (
            <li key={v.id} role="option" aria-selected={sel}
                onClick={() => p.onVoice(v.id)}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), p.onVoice(v.id))}
                tabIndex={0}
                className={`group flex items-center gap-3 rounded-lg px-2.5 py-2 cursor-pointer transition-colors
                  ${sel ? "bg-raised" : "hover:bg-raised/60"}`}>
              <span aria-hidden className={`h-7 w-[3px] rounded-full ${sel ? "bg-accent" : "bg-transparent"}`} />
              <span className="min-w-0 flex-1">
                <span className="block text-sm truncate">{v.name}</span>
                <span className="block text-2xs text-faint truncate">
                  {tags || v.description || v.category}
                </span>
              </span>
              {v.preview_url && (
                <button type="button" onClick={(e) => (e.stopPropagation(), preview(v))}
                        aria-label={playing === v.id ? `Stop ${v.name} preview` : `Preview ${v.name}`}
                        className="h-7 w-7 shrink-0 rounded-full border border-line text-muted hover:text-text hover:border-faint
                                   flex items-center justify-center">
                  {playing === v.id ? <StopIcon /> : <PlayIcon />}
                </button>
              )}
            </li>
          );
        })}
        {!shown.length && <li className="px-3 py-4 text-sm text-faint">No voices match “{query}”.</li>}
      </ul>

      <div className="mt-3 border-t border-line px-4 pt-4 pb-4 flex flex-col gap-3">
        <label htmlFor="dialogue" className="text-sm text-muted">Dialogue</label>
        <textarea id="dialogue" rows={5} value={p.text} onChange={(e) => p.onText(e.target.value)}
                  className="field resize-none leading-relaxed"
                  placeholder="Type the line you want spoken in the scene." />
        <div className="flex items-center justify-between text-2xs num">
          <span className={over ? "text-danger" : "text-faint"}>
            {p.text.trim().length} / {max} characters
          </span>
          {credits && (
            <span className="text-faint" title="Remaining ElevenLabs characters this period">
              {credits.remaining.toLocaleString()} left on account
            </span>
          )}
        </div>
        <button className={`${upToDate ? "btn-quiet" : "btn-primary"} w-full`} disabled={p.busy || over || !p.text.trim() || !p.voiceId || !!upToDate}
                onClick={p.onGenerate}>
          {p.busy ? <><Spinner /> Generating speech</> : upToDate ? <><Check /> Speech ready</> : "Generate speech"}
        </button>
        {p.speech && (
          <p className="text-2xs text-faint num" aria-live="polite">
            {p.speech.duration.toFixed(1)} s clip{p.speech.demo ? ", offline demo voice" : ""}.
            {" "}{p.speech.cached ? "From cache, no characters used." : p.speech.characters ? `${p.speech.characters} characters used.` : "Your recording."}
          </p>
        )}
        {p.error && <p role="alert" className="text-xs text-danger leading-snug">{p.error}</p>}
        <button type="button" onClick={() => fileRef.current?.click()}
                className="self-start text-xs text-muted underline decoration-line underline-offset-4 hover:text-text">
          Use my own recording instead
        </button>
        <input ref={fileRef} type="file" accept="audio/*" className="hidden"
               onChange={(e) => { const f = e.target.files?.[0]; if (f) p.onUpload(f); e.target.value = ""; }} />
      </div>
    </section>
  );
}

export const PlayIcon = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden><path d="M2 1 L9 5 L2 9 Z" fill="currentColor" /></svg>
);
export const PauseIcon = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden><rect x="1.5" y="1" width="2.5" height="8" rx=".5" fill="currentColor" /><rect x="6" y="1" width="2.5" height="8" rx=".5" fill="currentColor" /></svg>
);
export const StopIcon = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden><rect x="1.5" y="1.5" width="7" height="7" rx="1" fill="currentColor" /></svg>
);
export const Spinner = () => (
  <svg className="animate-spin" width="14" height="14" viewBox="0 0 24 24" aria-hidden>
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity=".25" strokeWidth="3" fill="none" />
    <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" fill="none" strokeLinecap="round" />
  </svg>
);
export const Check = () => (
  <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden><path d="M2 6.5 L5 9 L10 3" stroke="#A493FF" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>
);
