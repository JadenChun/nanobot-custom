import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { MediaLoadingIndicator } from "@/components/AssetPreview";
import { drawVideoFrame, VideoFrame } from "@/components/VideoFrame";
import { queueMobilePreview, useMobilePreviewVisibility } from "@/lib/mobile-media";
import { cn } from "@/lib/utils";

interface MobileVideoPlayerProps {
  src: string;
  label: string;
  className?: string;
  containerClassName?: string;
}

export function MobileVideoPlayer({ src, label, className, containerClassName }: MobileVideoPlayerProps) {
  const { t } = useTranslation();
  const container = useRef<HTMLSpanElement>(null);
  const visible = useMobilePreviewVisibility(container, true);
  const [admitted, setAdmitted] = useState(false);
  const [poster, setPoster] = useState<string>();
  const [previewFailed, setPreviewFailed] = useState(false);
  const [playback, setPlayback] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const previewing = visible && playback === "idle" && !previewFailed && !poster;
  const failPreview = useCallback(() => setPreviewFailed(true), []);

  useEffect(() => {
    if (previewing) return queueMobilePreview(() => setAdmitted(true));
  }, [previewing]);

  function capturePoster(video: HTMLVideoElement) {
    const canvas = document.createElement("canvas");
    if (!drawVideoFrame(video, canvas)) {
      failPreview();
      return;
    }
    try {
      setPoster(canvas.toDataURL("image/jpeg", 0.85));
    } catch {
      // A cross-origin video can still play even if its frames cannot be exported.
      failPreview();
    }
  }

  return (
    <span ref={container} className={cn("relative block", containerClassName)}>
      <video
        src={src}
        poster={poster}
        controls
        playsInline
        preload="none"
        aria-label={label}
        onPlay={() => setPlayback("loading")}
        onWaiting={() => setPlayback("loading")}
        onPlaying={() => setPlayback("ready")}
        onPause={() => setPlayback((status) => status === "loading" ? "ready" : status)}
        onError={() => setPlayback("error")}
        className={cn("bg-black object-contain", className)}
      />
      {previewing && admitted ? <VideoFrame src={src} onFrame={capturePoster} onError={failPreview} /> : null}
      {previewing || playback === "loading" ? <MediaLoadingIndicator /> : null}
      {playback === "error" ? (
        <span role="status" className="pointer-events-none absolute inset-0 grid place-items-center text-sm text-white">
          {t("message.attachmentUnavailable", { defaultValue: "Attachment unavailable" })}
        </span>
      ) : null}
    </span>
  );
}
