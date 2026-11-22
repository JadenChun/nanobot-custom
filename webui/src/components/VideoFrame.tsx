import { useEffect, useRef } from "react";

export function drawVideoFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement): boolean {
  const context = canvas.getContext("2d");
  if (!context || !video.videoWidth || !video.videoHeight) return false;
  const scale = Math.min(1, 384 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  try {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return true;
  } catch {
    return false;
  }
}

export function VideoFrame({ src, onFrame, onError }: {
  src: string;
  onFrame: (video: HTMLVideoElement) => void;
  onError: () => void;
}) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    video.src = src;
    return () => {
      // A paused preview can hold an open byte range until the video is unloaded.
      video.removeAttribute("src");
      video.load();
    };
  }, [src]);

  useEffect(() => {
    const timeout = window.setTimeout(onError, 30_000);
    return () => window.clearTimeout(timeout);
  }, [onError]);

  return (
    <video
      ref={ref}
      muted
      playsInline
      preload="auto"
      aria-hidden
      className="pointer-events-none absolute inset-0 h-full w-full opacity-0"
      onLoadedData={(event) => onFrame(event.currentTarget)}
      onError={onError}
    />
  );
}
