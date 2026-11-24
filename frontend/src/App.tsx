import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, markerDistance } from "./api";
import type { Analysis, Pt, RenderParams, RenderResult, Speech, Status, Voice } from "./api";
import VoicePanel from "./components/VoicePanel";
import Stage from "./components/Stage";
import AcousticsPanel from "./components/AcousticsPanel";
import Compare from "./components/Compare";

const DEFAULT_LINE = "Hello? Is anyone there? I can hear my own voice coming back to me.";

const paramsFrom = (a: Analysis): RenderParams => ({
  width: a.room.width.value, length: a.room.length.value, height: a.room.height.value,
  rt60: null, absorption_scale: 1, speaker: a.speaker, listener: a.listener,
  facing: "towards", obstruction: a.obstruction, distance: null, wet: 1, reflections: 1,
});

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [voiceId, setVoiceId] = useState("");
  const [text, setText] = useState(DEFAULT_LINE);
  const [speech, setSpeech] = useState<Speech | null>(null);
  const [speechText, setSpeechText] = useState<string | null>(null);
  const [speechBusy, setSpeechBusy] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [analysing, setAnalysing] = useState(false);
  const [needsPreset, setNeedsPreset] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);

  const [params, setParams] = useState<RenderParams | null>(null);
  const [result, setResult] = useState<RenderResult | null>(null);
  const [rendering, setRendering] = useState(false);
  const [renderError, setRenderError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.status(), api.voices()])
      .then(([s, v]) => { setStatus(s); setVoices(v.voices); if (v.voices[0]) setVoiceId(v.voices[0].id); })
      .catch((e: Error) => setBootError(e.message));
  }, []);

  // ------------------------------------------------------------- speech
  const generate = async () => {
    setSpeechBusy(true); setSpeechError(null);
    try {
      const s = await api.speech(voiceId, text);
      setSpeech(s); setSpeechText(text.trim());
      if (s.credits && status) setStatus({ ...status, credits: s.credits });
    } catch (e) { setSpeechError((e as Error).message); }
    finally { setSpeechBusy(false); }
  };
  const upload = async (f: File) => {
    setSpeechBusy(true); setSpeechError(null);
    try { const s = await api.uploadSpeech(f); setSpeech(s); setSpeechText(null); }
    catch (e) { setSpeechError((e as Error).message); }
    finally { setSpeechBusy(false); }
  };

  // ------------------------------------------------------------ analysis
  const analyse = async (f: File, preset?: string) => {
    setAnalysing(true); setImageError(null); setNeedsPreset(false);
    try {
      const a = await api.analyze(f, preset);
      setAnalysis(a); setParams(paramsFrom(a)); setResult(null);
    } catch (e) {
      setImageError((e as Error).message);
      if (!status?.vision) setNeedsPreset(true);
    } finally { setAnalysing(false); }
  };
  const onFile = (f: File) => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setFile(f); setImageUrl(URL.createObjectURL(f)); setAnalysis(null); setParams(null); setResult(null);
    setImageError(null);
    if (status?.vision) void analyse(f); else setNeedsPreset(true);
  };

  // -------------------------------------------------------------- render
  const seq = useRef(0);
  const abort = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!speech || !analysis || !params) return;
    const id = ++seq.current;
    const t = setTimeout(async () => {
      abort.current?.abort();
      const ctl = new AbortController(); abort.current = ctl;
      setRendering(true); setRenderError(null);
      try {
        const r = await api.render(speech.id, analysis.id, params, ctl.signal);
        if (id === seq.current) setResult(r);
      } catch (e) {
        if ((e as Error).name !== "AbortError" && id === seq.current) setRenderError((e as Error).message);
      } finally { if (id === seq.current) setRendering(false); }
    }, 450);
    return () => clearTimeout(t);
  }, [speech, analysis, params]);

  const patch = useCallback((p: Partial<RenderParams>) => setParams((cur) => (cur ? { ...cur, ...p } : cur)), []);
  const move = useCallback((which: "speaker" | "listener", pt: Pt) => patch({ [which]: pt }), [patch]);

  const dims = params ? [params.width, params.length, params.height] : null;
  const autoDistance = params && dims ? markerDistance(params.speaker, params.listener, dims) : null;
  const shownDistance = params?.distance ?? autoDistance;
  const voice = voices.find((v) => v.id === voiceId);
  const downloadBase = useMemo(() => (voice?.name || "voice").toLowerCase().replace(/[^a-z0-9]+/g, "-"), [voice]);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="flex flex-wrap items-center gap-x-6 gap-y-2 px-5 lg:px-6 py-3.5 border-b border-line">
        <div className="flex items-center gap-3">
          <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden><path d="M5 22 L11 12 L17 19 L21 8 L27 22" stroke="#A493FF" strokeWidth="2.6" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>
          <div>
            <h1 className="text-base font-semibold tracking-[-0.02em] leading-tight">Image-to-Acoustics</h1>
            <p className="text-2xs text-faint leading-tight">Hear a voice inside the place in a picture</p>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-4 text-2xs text-muted">
          <StatusDot ok={!!status?.elevenlabs} label={status?.elevenlabs ? "ElevenLabs connected" : "Offline demo voice"} />
          <StatusDot ok={!!status?.vision} label={status?.vision ? `Image analysis by ${cap(status.vision_provider)}` : "Image analysis off"} />
        </div>
      </header>

      {bootError && (
        <div role="alert" className="mx-5 mt-4 rounded-lg border border-danger/40 bg-danger/10 px-4 py-3 text-sm">{bootError}</div>
      )}

      <main className="flex-1 p-4 lg:p-5 grid gap-4 lg:gap-5 grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)_330px] lg:grid-rows-[minmax(0,1fr)_auto] lg:h-[calc(100vh-61px)] lg:min-h-[620px] lg:flex-none">
        <VoicePanel status={status} voices={voices} voiceId={voiceId} onVoice={setVoiceId} text={text} onText={setText}
                    speech={speech} speechText={speechText} busy={speechBusy} error={speechError}
                    onGenerate={generate} onUpload={upload} />
        <Stage status={status} imageUrl={imageUrl} analysis={analysis} analysing={analysing} needsPreset={needsPreset}
               error={imageError} speaker={params?.speaker ?? null} listener={params?.listener ?? null}
               facing={params?.facing ?? "towards"} distance={shownDistance} obstruction={params?.obstruction ?? 0}
               onFile={onFile} onPreset={(id) => file && analyse(file, id)} onMove={move} />
        <AcousticsPanel analysis={analysis} result={result} params={params} autoDistance={autoDistance}
                        onChange={patch} onReset={() => analysis && setParams(paramsFrom(analysis))} />
        <div className="lg:col-span-3">
          <Compare originalUrl={speech?.url ?? null} result={result} rendering={rendering} error={renderError}
                   sceneName={analysis?.preset ?? analysis?.scene ?? "room"} downloadBase={downloadBase} />
        </div>
      </main>
    </div>
  );
}

function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-accent" : "bg-warn"}`} aria-hidden />
      {label}
    </span>
  );
}
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
