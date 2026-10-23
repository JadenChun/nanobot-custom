import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";

const VOICE_RECORDING_MAX_MS = 120_000;
const VOICE_RECORDING_MIN_MS = 650;
const VOICE_HOLD_START_MS = 140;
const VOICE_WAVEFORM_BAR_COUNT = 64;
const VOICE_WAVEFORM_SILENT_HEIGHT = 3;
const VOICE_WAVEFORM_MIN_HEIGHT = 7;
const VOICE_WAVEFORM_MAX_HEIGHT = 34;
const VOICE_MIN_LEVEL = 0.018;
const VOICE_WAVEFORM_IDLE_LEVELS = Array.from(
  { length: VOICE_WAVEFORM_BAR_COUNT },
  () => VOICE_WAVEFORM_SILENT_HEIGHT,
);
const VOICE_MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
] as const;

export type VoiceRecorderState = "idle" | "recording" | "transcribing";
export type VoiceRecorderErrorKey =
  | "failed"
  | "insecureContext"
  | "noDevice"
  | "notConfigured"
  | "permission"
  | "tooLong"
  | "tooShort"
  | "unsupported";

interface VoiceRecorderOptions {
  disabled?: boolean;
  onClearError: () => void;
  onError: (key: VoiceRecorderErrorKey) => void;
  onTranscript: (text: string) => void;
  onTranscribeAudio?: (dataUrl: string, options?: { durationMs?: number }) => Promise<string>;
  /** When true, convert recorded audio to WAV before sending (needed for providers that don't support WebM). */
  wantsWav?: boolean;
}

export function useVoiceRecorder({
  disabled,
  onClearError,
  onError,
  onTranscript,
  onTranscribeAudio,
  wantsWav = false,
}: VoiceRecorderOptions) {
  const captureRef = useRef<VoiceCapture | null>(null);
  const audioRef = useRef<VoiceAudioState | null>(null);
  const maxTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const holdTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const holdActiveRef = useRef(false);
  const startPendingRef = useRef(false);
  const requestEpochRef = useRef(0);
  const recordingIdRef = useRef(0);
  const mountedRef = useRef(false);
  const stopAfterStartRef = useRef(false);
  const suppressClickRef = useRef(false);
  const suppressClickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shortcutActiveRef = useRef(false);
  const [state, setState] = useState<VoiceRecorderState>("idle");
  const [elapsedMs, setElapsedMs] = useState(0);
  const [levels, setLevels] = useState<number[]>(VOICE_WAVEFORM_IDLE_LEVELS);

  const clearSuppressClickTimer = useCallback(() => clearTimer(suppressClickTimerRef), []);

  const suppressNextClick = useCallback(() => {
    clearSuppressClickTimer();
    suppressClickRef.current = true;
    suppressClickTimerRef.current = setTimeout(() => {
      suppressClickRef.current = false;
      suppressClickTimerRef.current = null;
    }, 500);
  }, [clearSuppressClickTimer]);

  const stopWaveform = useCallback(() => {
    const audio = audioRef.current;
    audioRef.current = null;
    if (!audio) return;
    if (audio.frame !== null) {
      try {
        cancelAnimationFrame(audio.frame);
      } catch {
        // Animation frames may already be cancelled by the host.
      }
    }
    try {
      audio.source.disconnect();
    } catch {
      // The browser may have disconnected the source while hiding the webview.
    }
    try {
      audio.analyser.disconnect();
    } catch {
      // The browser may have disconnected the analyser while hiding the webview.
    }
    void audio.context.close().catch(() => undefined);
  }, []);

  const startWaveform = useCallback((stream: MediaStream) => {
    const AudioContextCtor = audioContextConstructor();
    if (!AudioContextCtor) return;
    stopWaveform();
    setLevels(VOICE_WAVEFORM_IDLE_LEVELS);
    try {
      const context = new AudioContextCtor();
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.68;
      source.connect(analyser);
      const audio: VoiceAudioState = {
        analyser,
        context,
        data: new Uint8Array(analyser.fftSize),
        frame: null,
        source,
      };
      const tick = () => {
        const current = audioRef.current;
        if (!current) return;
        if (current.context.state !== "running") {
          void current.context.resume().catch(() => undefined);
          current.frame = requestAnimationFrame(tick);
          return;
        }
        current.analyser.getByteTimeDomainData(current.data);
        const level = voiceLevelFromSamples(current.data);
        setLevels((currentLevels) => [
          ...currentLevels.slice(1),
          waveformHeightFromLevel(level),
        ]);
        current.frame = requestAnimationFrame(tick);
      };
      audioRef.current = audio;
      void context.resume().catch(() => undefined);
      audio.frame = requestAnimationFrame(tick);
    } catch {
      stopWaveform();
    }
  }, [stopWaveform]);

  const cleanupRecording = useCallback((capture?: VoiceCapture) => {
    if (capture && captureRef.current !== capture) return;
    clearTimer(holdTimerRef);
    clearTimer(maxTimerRef);
    const activeCapture = captureRef.current;
    captureRef.current = null;
    stopWaveform();
    if (activeCapture) stopMediaStream(activeCapture.stream);
    startPendingRef.current = false;
    shortcutActiveRef.current = false;
  }, [stopWaveform]);

  const stopRecording = useCallback(() => {
    const capture = captureRef.current;
    if (!capture) return;
    stopMediaRecorder(capture.recorder);
  }, []);

  const stopRecordingWhenReady = useCallback(() => {
    const capture = captureRef.current;
    if (capture && capture.recorder.state !== "inactive") {
      stopRecording();
    } else if (startPendingRef.current) {
      stopAfterStartRef.current = true;
    }
  }, [stopRecording]);

  const startRecording = useCallback(async () => {
    if (!onTranscribeAudio || state !== "idle" || startPendingRef.current) return;
    onClearError();
    if (window.isSecureContext === false) {
      onError("insecureContext");
      return;
    }
    const mediaDevices = navigator.mediaDevices;
    const MediaRecorderCtor = mediaRecorderConstructor();
    if (!mediaDevices?.getUserMedia || !MediaRecorderCtor) {
      onError("unsupported");
      return;
    }
    const requestEpoch = ++requestEpochRef.current;
    const recordingId = ++recordingIdRef.current;
    startPendingRef.current = true;
    let stream: MediaStream | null = null;
    let capture: VoiceCapture | null = null;
    try {
      stream = await mediaDevices.getUserMedia({ audio: true });
      if (
        requestEpoch !== requestEpochRef.current
        || recordingId !== recordingIdRef.current
        || !mountedRef.current
      ) {
        stopMediaStream(stream);
        return;
      }
      const recorder = new MediaRecorderCtor(stream, mediaRecorderOptions(MediaRecorderCtor));
      capture = {
        chunks: [],
        recorder,
        startedAt: Date.now(),
        stream,
      };
      captureRef.current = capture;
      startPendingRef.current = false;
      setElapsedMs(0);
      startWaveform(stream);
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) capture?.chunks.push(event.data);
      };
      recorder.onstop = () => {
        const chunks = capture?.chunks.splice(0) ?? [];
        const durationMs = Math.max(0, Date.now() - (capture?.startedAt ?? Date.now()));
        const mimeType = recorder.mimeType || "audio/webm";
        const isCurrentRecording = recordingIdRef.current === recordingId;
        if (capture) cleanupRecording(capture);
        if (!mountedRef.current) return;
        if (chunks.length === 0) {
          if (isCurrentRecording) setState("idle");
          return;
        }
        if (durationMs < VOICE_RECORDING_MIN_MS) {
          if (isCurrentRecording) {
            setState("idle");
            onError("tooShort");
          }
          return;
        }
        if (isCurrentRecording) setState("transcribing");
        const blob = new Blob(chunks, { type: mimeType });
        const audioPromise = wantsWav ? convertBlobToWav(blob) : blobToDataUrl(blob);
        void audioPromise
          .then((dataUrl) => onTranscribeAudio(dataUrl, { durationMs }))
          .then((text) => {
            if (mountedRef.current) onTranscript(text);
          })
          .catch((error) => {
            if (mountedRef.current && recordingIdRef.current === recordingId) {
              onError(transcriptionErrorKey(error));
            }
          })
          .finally(() => {
            if (mountedRef.current && recordingIdRef.current === recordingId) setState("idle");
          });
      };
      recorder.start();
      setState("recording");
      onClearError();
      maxTimerRef.current = setTimeout(stopRecording, VOICE_RECORDING_MAX_MS);
    } catch (error) {
      if (stream) stopMediaStream(stream);
      if (
        requestEpoch !== requestEpochRef.current
        || recordingId !== recordingIdRef.current
        || !mountedRef.current
      ) return;
      if (capture) stopMediaRecorder(capture.recorder);
      cleanupRecording(capture ?? undefined);
      setState("idle");
      onError(recordingErrorKey(error));
    }
  }, [
    cleanupRecording,
    onClearError,
    onError,
    onTranscribeAudio,
    onTranscript,
    startWaveform,
    state,
    stopRecording,
    wantsWav,
  ]);

  const startRecordingWithDeferredStop = useCallback(() => {
    stopAfterStartRef.current = false;
    void startRecording().then(() => {
      if (!stopAfterStartRef.current) return;
      stopAfterStartRef.current = false;
      stopRecording();
    });
  }, [startRecording, stopRecording]);

  const beginPress = useCallback((event: ReactPointerEvent<HTMLButtonElement>) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    if (!onTranscribeAudio || disabled || state !== "idle") return;
    clearTimer(holdTimerRef);
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Some embedded runtimes do not expose pointer capture for toolbar buttons.
    }
    holdTimerRef.current = setTimeout(() => {
      holdTimerRef.current = null;
      holdActiveRef.current = true;
      suppressNextClick();
      startRecordingWithDeferredStop();
    }, VOICE_HOLD_START_MS);
  }, [disabled, onTranscribeAudio, startRecordingWithDeferredStop, state, suppressNextClick]);

  const endPress = useCallback(() => {
    const wasHoldRecording = holdActiveRef.current;
    clearTimer(holdTimerRef);
    if (!wasHoldRecording) return;
    holdActiveRef.current = false;
    suppressNextClick();
    stopRecordingWhenReady();
  }, [stopRecordingWhenReady, suppressNextClick]);

  const handleClick = useCallback(() => {
    if (suppressClickRef.current) {
      clearSuppressClickTimer();
      suppressClickRef.current = false;
      return;
    }
    if (state === "recording") stopRecording();
    else void startRecording();
  }, [clearSuppressClickTimer, startRecording, state, stopRecording]);

  const beginShortcutHold = useCallback(() => {
    if (!onTranscribeAudio || disabled || state !== "idle" || shortcutActiveRef.current) return;
    shortcutActiveRef.current = true;
    startRecordingWithDeferredStop();
  }, [disabled, onTranscribeAudio, startRecordingWithDeferredStop, state]);

  const endShortcutHold = useCallback(() => {
    if (!shortcutActiveRef.current) return;
    shortcutActiveRef.current = false;
    stopRecordingWhenReady();
  }, [stopRecordingWhenReady]);

  const clearCaptureOnHide = useCallback(() => {
    const capture = captureRef.current;
    const permissionPending = startPendingRef.current;
    requestEpochRef.current += 1;
    clearTimer(holdTimerRef);
    clearTimer(maxTimerRef);
    clearSuppressClickTimer();
    holdActiveRef.current = false;
    suppressClickRef.current = false;
    stopAfterStartRef.current = false;
    shortcutActiveRef.current = false;
    startPendingRef.current = false;

    if (capture) {
      captureRef.current = null;
      stopWaveform();
      setElapsedMs(0);
      setState("idle");
      stopMediaRecorder(capture.recorder);
      stopMediaStream(capture.stream);
    } else if (permissionPending) {
      recordingIdRef.current += 1;
      setState("idle");
    }
  }, [clearSuppressClickTimer, stopWaveform]);

  useEffect(() => {
    if (state !== "recording") {
      setElapsedMs(0);
      return;
    }
    const updateElapsed = () => {
      setElapsedMs(Math.max(0, Date.now() - (captureRef.current?.startedAt ?? Date.now())));
    };
    updateElapsed();
    const interval = window.setInterval(updateElapsed, 250);
    return () => window.clearInterval(interval);
  }, [state]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      requestEpochRef.current += 1;
      recordingIdRef.current += 1;
      clearTimer(holdTimerRef);
      clearTimer(maxTimerRef);
      clearSuppressClickTimer();
      const capture = captureRef.current;
      cleanupRecording(capture ?? undefined);
      if (capture) stopMediaRecorder(capture.recorder);
    };
  }, [cleanupRecording, clearSuppressClickTimer]);

  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.visibilityState === "hidden") clearCaptureOnHide();
    };
    window.addEventListener("nanobot:desktop-hide", clearCaptureOnHide);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.removeEventListener("nanobot:desktop-hide", clearCaptureOnHide);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [clearCaptureOnHide]);

  useEffect(() => () => clearSuppressClickTimer(), [clearSuppressClickTimer]);

  return {
    beginShortcutHold,
    beginPress,
    buttonDisabled: disabled || state === "transcribing",
    elapsedLabel: formatVoiceElapsed(elapsedMs),
    endShortcutHold,
    endPress,
    handleClick,
    isRecording: state === "recording",
    levels,
    state,
  };
}

interface VoiceAudioState {
  analyser: AnalyserNode;
  context: AudioContext;
  data: Uint8Array<ArrayBuffer>;
  frame: number | null;
  source: MediaStreamAudioSourceNode;
}

interface VoiceCapture {
  chunks: BlobPart[];
  recorder: MediaRecorder;
  startedAt: number;
  stream: MediaStream;
}

function stopMediaStream(stream: MediaStream) {
  stream.getTracks().forEach((track) => track.stop());
}

function stopMediaRecorder(recorder: MediaRecorder) {
  if (recorder.state === "inactive") return;
  try {
    recorder.stop();
  } catch {
    // The browser may have stopped the recorder when its tracks ended.
  }
}

function clearTimer(ref: { current: ReturnType<typeof setTimeout> | null }) {
  if (ref.current !== null) {
    clearTimeout(ref.current);
    ref.current = null;
  }
}

function mediaRecorderOptions(MediaRecorderCtor: MediaRecorderConstructor): MediaRecorderOptions | undefined {
  const mimeType = VOICE_MIME_CANDIDATES.find((type) => MediaRecorderCtor.isTypeSupported?.(type));
  return mimeType ? { mimeType } : undefined;
}

type MediaRecorderConstructor = typeof MediaRecorder;

function mediaRecorderConstructor(): MediaRecorderConstructor | undefined {
  if (typeof window === "undefined") return undefined;
  const browserWindow = window as Window & {
    MediaRecorder?: MediaRecorderConstructor;
  };
  return browserWindow.MediaRecorder;
}

function formatVoiceElapsed(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

function audioContextConstructor(): typeof AudioContext | undefined {
  if (typeof window === "undefined") return undefined;
  return window.AudioContext
    ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
}

function voiceLevelFromSamples(samples: ArrayLike<number>): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const centered = (samples[index] - 128) / 128;
    sum += centered * centered;
  }
  const rms = Math.sqrt(sum / samples.length);
  return Math.min(1, Math.pow(rms * 4.2, 0.72));
}

function waveformHeightFromLevel(level: number): number {
  if (level < VOICE_MIN_LEVEL) return VOICE_WAVEFORM_SILENT_HEIGHT;
  const activeLevel = Math.min(1, (level - VOICE_MIN_LEVEL) / (1 - VOICE_MIN_LEVEL));
  return Math.round(
    VOICE_WAVEFORM_MIN_HEIGHT
      + activeLevel * (VOICE_WAVEFORM_MAX_HEIGHT - VOICE_WAVEFORM_MIN_HEIGHT),
  );
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error("invalid_data_url"));
    };
    reader.onerror = () => reject(reader.error ?? new Error("read_failed"));
    reader.readAsDataURL(blob);
  });
}

/**
 * Convert any browser-recorded audio blob (typically webm/opus) to WAV
 * using the Web Audio API. This avoids sending unsupported formats
 * (e.g. webm) to ASR providers that only accept wav/mp3/mpeg.
 */
async function convertBlobToWav(blob: Blob): Promise<string> {
  const AudioCtx = audioContextConstructor();
  if (!AudioCtx) return blobToDataUrl(blob);

  const arrayBuffer = await blob.arrayBuffer();
  const ctx = new AudioCtx();
  try {
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    const wavBlob = audioBufferToWav(audioBuffer);
    return blobToDataUrl(wavBlob);
  } finally {
    void ctx.close();
  }
}

/**
 * Encode an AudioBuffer as a 16-bit PCM WAV Blob.
 */
function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitsPerSample = 16;

  // Interleave channels
  const channels: Float32Array[] = [];
  for (let ch = 0; ch < numChannels; ch++) {
    channels.push(buffer.getChannelData(ch));
  }
  const length = channels[0].length;
  const interleaved = new Int16Array(length * numChannels);
  for (let i = 0; i < length; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const sample = Math.max(-1, Math.min(1, channels[ch][i]));
      interleaved[i * numChannels + ch] = sample < 0
        ? sample * 0x8000
        : sample * 0x7FFF;
    }
  }

  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = interleaved.byteLength;
  const headerSize = 44;
  const totalSize = headerSize + dataSize;

  const buffer2 = new ArrayBuffer(totalSize);
  const view = new DataView(buffer2);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, totalSize - 8, true);
  writeString(view, 8, "WAVE");

  // fmt sub-chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // sub-chunk size
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data sub-chunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  new Int16Array(buffer2, headerSize).set(interleaved);

  return new Blob([buffer2], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

function transcriptionErrorKey(error: unknown): VoiceRecorderErrorKey {
  const detail = error instanceof Error ? error.message : "";
  if (detail === "not_configured") return "notConfigured";
  if (detail === "duration") return "tooLong";
  return "failed";
}

function recordingErrorKey(error: unknown): VoiceRecorderErrorKey {
  const name = error instanceof Error ? error.name : "";
  if (name === "NotFoundError") return "noDevice";
  return "permission";
}
