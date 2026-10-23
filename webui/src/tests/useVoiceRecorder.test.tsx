import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useVoiceRecorder } from "@/hooks/useVoiceRecorder";

const initialMediaDevices = navigator.mediaDevices;
const initialVisibilityDescriptor = Object.getOwnPropertyDescriptor(document, "visibilityState");

interface FakeTrack {
  stop: ReturnType<typeof vi.fn>;
}

interface FakeStream extends MediaStream {
  track: FakeTrack;
}

class FakeMediaRecorder {
  static instances: FakeMediaRecorder[] = [];
  static isTypeSupported = vi.fn(() => true);

  readonly stop = vi.fn();
  readonly stream: MediaStream;
  state: RecordingState = "inactive";
  mimeType = "audio/webm";
  ondataavailable: ((event: BlobEvent) => void) | null = null;
  onstop: (() => void) | null = null;

  constructor(stream: MediaStream) {
    this.stream = stream;
    FakeMediaRecorder.instances.push(this);
    this.stop.mockImplementation(() => {
      this.state = "inactive";
      this.emitStopWithAudio();
    });
  }

  start() {
    this.state = "recording";
  }

  emitStopWithAudio() {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["voice"], { type: this.mimeType }) } as BlobEvent);
    this.onstop?.();
  }
}

function makeStream(): FakeStream {
  const track = { stop: vi.fn() };
  return {
    track,
    getTracks: () => [track as unknown as MediaStreamTrack],
  } as unknown as FakeStream;
}

function installMediaDevices(getUserMedia = vi.fn(async () => makeStream())) {
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia },
  });
  FakeMediaRecorder.instances = [];
  vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
  return getUserMedia;
}

function renderRecorder(onTranscribeAudio = vi.fn(async () => "transcript")) {
  const onClearError = vi.fn();
  const onError = vi.fn();
  const onTranscript = vi.fn();
  const hook = renderHook(() => useVoiceRecorder({
    onClearError,
    onError,
    onTranscript,
    onTranscribeAudio,
  }));
  return { ...hook, onClearError, onError, onTranscript, onTranscribeAudio };
}

function hideDesktop() {
  window.dispatchEvent(new Event("nanobot:desktop-hide"));
}

function setVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    value: state,
  });
  document.dispatchEvent(new Event("visibilitychange"));
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
  if (initialMediaDevices) {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: initialMediaDevices,
    });
  } else {
    Reflect.deleteProperty(navigator, "mediaDevices");
  }
  if (initialVisibilityDescriptor) {
    Object.defineProperty(document, "visibilityState", initialVisibilityDescriptor);
  } else {
    Reflect.deleteProperty(document, "visibilityState");
  }
});

describe("useVoiceRecorder desktop privacy cleanup", () => {
  it("stops an active capture on hide and finishes its transcript into the draft callback", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(0));
    const getUserMedia = installMediaDevices();
    const source = {
      connect: vi.fn(),
      disconnect: vi.fn(),
    } as unknown as MediaStreamAudioSourceNode;
    const analyser = {
      fftSize: 0,
      smoothingTimeConstant: 0,
      disconnect: vi.fn(),
      getByteTimeDomainData: vi.fn(),
    } as unknown as AnalyserNode;
    const audioContext = {
      state: "running",
      createMediaStreamSource: vi.fn(() => source),
      createAnalyser: vi.fn(() => analyser),
      resume: vi.fn(async () => undefined),
      close: vi.fn(async () => undefined),
    };
    const cancelFrame = vi.fn();
    vi.stubGlobal("AudioContext", vi.fn(() => audioContext));
    vi.stubGlobal("requestAnimationFrame", vi.fn(() => 17));
    vi.stubGlobal("cancelAnimationFrame", cancelFrame);
    const recorder = renderRecorder();

    await act(async () => {
      recorder.result.current.handleClick();
      await Promise.resolve();
    });
    expect(recorder.result.current.state).toBe("recording");
    const stream = FakeMediaRecorder.instances[0].stream as FakeStream;
    const mediaRecorder = FakeMediaRecorder.instances[0];
    vi.setSystemTime(new Date(700));

    act(() => hideDesktop());

    expect(mediaRecorder.stop).toHaveBeenCalledTimes(1);
    expect(stream.track.stop).toHaveBeenCalledTimes(1);
    expect(getUserMedia).toHaveBeenCalledTimes(1);
    expect(recorder.result.current.state).toBe("transcribing");
    expect(vi.getTimerCount()).toBe(0);
    expect(cancelFrame).toHaveBeenCalledWith(17);
    expect(source.disconnect).toHaveBeenCalledTimes(1);
    expect(analyser.disconnect).toHaveBeenCalledTimes(1);
    expect(audioContext.close).toHaveBeenCalledTimes(1);
    vi.useRealTimers();
    await waitFor(() => expect(recorder.onTranscript).toHaveBeenCalledWith("transcript"));
  });

  it("fences a pending permission request through hide and reopen", async () => {
    let resolveStream: ((stream: MediaStream) => void) | undefined;
    const getUserMedia = installMediaDevices(vi.fn(() => new Promise((resolve) => {
      resolveStream = resolve as (stream: MediaStream) => void;
    })));
    const recorder = renderRecorder();

    act(() => recorder.result.current.handleClick());
    expect(getUserMedia).toHaveBeenCalledTimes(1);

    setVisibility("hidden");
    setVisibility("visible");
    const lateStream = makeStream();
    await act(async () => {
      resolveStream?.(lateStream);
      await Promise.resolve();
    });

    expect(lateStream.track.stop).toHaveBeenCalledTimes(1);
    expect(FakeMediaRecorder.instances).toHaveLength(0);
    expect(recorder.result.current.state).toBe("idle");
  });

  it("stops an old permission result without disturbing a newer capture", async () => {
    const pendingStreams: Array<(stream: MediaStream) => void> = [];
    installMediaDevices(vi.fn(() => new Promise((resolve) => {
      pendingStreams.push(resolve as (stream: MediaStream) => void);
    })));
    const recorder = renderRecorder();

    act(() => recorder.result.current.handleClick());
    act(() => hideDesktop());
    setVisibility("visible");
    act(() => recorder.result.current.handleClick());

    const currentStream = makeStream();
    await act(async () => {
      pendingStreams[1](currentStream);
      await Promise.resolve();
    });
    expect(recorder.result.current.state).toBe("recording");

    const staleStream = makeStream();
    await act(async () => {
      pendingStreams[0](staleStream);
      await Promise.resolve();
    });

    expect(staleStream.track.stop).toHaveBeenCalledTimes(1);
    expect(FakeMediaRecorder.instances).toHaveLength(1);
    expect(FakeMediaRecorder.instances[0].stream).toBe(currentStream);
    expect(currentStream.track.stop).not.toHaveBeenCalled();
    expect(recorder.result.current.state).toBe("recording");
  });

  it("keeps too-short handling when hiding an active recording", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(0));
    installMediaDevices();
    const recorder = renderRecorder();

    await act(async () => {
      recorder.result.current.handleClick();
      await Promise.resolve();
    });
    vi.setSystemTime(new Date(400));

    act(() => hideDesktop());

    expect(recorder.result.current.state).toBe("idle");
    expect(recorder.onError).toHaveBeenCalledWith("tooShort");
    vi.useRealTimers();
  });

  it("cancels a pending hold and treats native and browser hide notifications idempotently", () => {
    vi.useFakeTimers();
    const getUserMedia = installMediaDevices();
    const recorder = renderRecorder();
    const button = document.createElement("button");

    act(() => recorder.result.current.beginPress({
      button: 0,
      currentTarget: button,
      pointerId: 1,
      pointerType: "touch",
    } as unknown as React.PointerEvent<HTMLButtonElement>));
    act(() => vi.advanceTimersByTime(80));
    act(() => {
      hideDesktop();
      setVisibility("hidden");
    });
    act(() => vi.advanceTimersByTime(500));

    expect(getUserMedia).not.toHaveBeenCalled();
    expect(recorder.result.current.state).toBe("idle");
  });

  it("clears hold-click suppression and fences permission after a hold starts", async () => {
    vi.useFakeTimers();
    let resolveStream: ((stream: MediaStream) => void) | undefined;
    const getUserMedia = installMediaDevices(vi.fn(() => new Promise((resolve) => {
      resolveStream = resolve as (stream: MediaStream) => void;
    })));
    const recorder = renderRecorder();
    const button = document.createElement("button");

    act(() => recorder.result.current.beginPress({
      button: 0,
      currentTarget: button,
      pointerId: 1,
      pointerType: "touch",
    } as unknown as React.PointerEvent<HTMLButtonElement>));
    act(() => vi.advanceTimersByTime(140));
    expect(getUserMedia).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(1);

    act(() => hideDesktop());
    expect(vi.getTimerCount()).toBe(0);

    const lateStream = makeStream();
    await act(async () => {
      resolveStream?.(lateStream);
      await Promise.resolve();
    });

    expect(lateStream.track.stop).toHaveBeenCalledTimes(1);
    expect(FakeMediaRecorder.instances).toHaveLength(0);
    expect(recorder.result.current.state).toBe("idle");
  });

  it("does not let an old stop event clear a newer recording", async () => {
    vi.useFakeTimers();
    const firstStream = makeStream();
    const secondStream = makeStream();
    const getUserMedia = installMediaDevices();
    getUserMedia.mockResolvedValueOnce(firstStream).mockResolvedValueOnce(secondStream);
    const recorder = renderRecorder();

    await act(async () => {
      recorder.result.current.handleClick();
      await Promise.resolve();
    });
    const oldRecorder = FakeMediaRecorder.instances[0];
    oldRecorder.stop.mockImplementation(() => {
      oldRecorder.state = "inactive";
    });
    act(() => hideDesktop());

    await act(async () => {
      recorder.result.current.handleClick();
      await Promise.resolve();
    });
    expect(recorder.result.current.state).toBe("recording");

    vi.setSystemTime(new Date(700));
    act(() => oldRecorder.emitStopWithAudio());

    expect(recorder.result.current.state).toBe("recording");
    expect(secondStream.track.stop).not.toHaveBeenCalled();
  });

  it("stops a permission result that resolves after unmount", async () => {
    let resolveStream: ((stream: MediaStream) => void) | undefined;
    installMediaDevices(vi.fn(() => new Promise((resolve) => {
      resolveStream = resolve as (stream: MediaStream) => void;
    })));
    const recorder = renderRecorder();

    act(() => recorder.result.current.handleClick());
    recorder.unmount();
    const lateStream = makeStream();
    await act(async () => {
      resolveStream?.(lateStream);
      await Promise.resolve();
    });

    expect(lateStream.track.stop).toHaveBeenCalledTimes(1);
    expect(FakeMediaRecorder.instances).toHaveLength(0);
  });
});
