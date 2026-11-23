import { useCallback, useEffect, useRef, useState } from "react";
import { CircleAlert, Loader2, Play } from "lucide-react";
import { useTranslation } from "react-i18next";

import { VideoFrame, drawVideoFrame } from "@/components/VideoFrame";
import { queueMobilePreview, useMobilePreviewVisibility } from "@/lib/mobile-media";
import { isMobileRuntimeActive } from "@/lib/mobile-runtime";
import type { UIImage } from "@/lib/types";
import { cn } from "@/lib/utils";

export function MediaLoadingIndicator() {
  const { t } = useTranslation();
  return (
    <span className="pointer-events-none absolute inset-0 grid place-items-center" aria-busy="true">
      <Loader2 className="h-5 w-5 animate-spin text-muted-foreground motion-reduce:animate-none" aria-hidden />
      <span className="sr-only">{t("message.mediaLoading", { defaultValue: "Loading media…" })}</span>
    </span>
  );
}

interface AssetPreviewProps {
  asset: UIImage;
  compact?: boolean;
  className?: string;
}

export function AssetPreview(props: AssetPreviewProps) {
  return <AssetPreviewContent key={props.asset.url} {...props} />;
}

function AssetPreviewContent({ asset, compact = false, className }: AssetPreviewProps) {
  const { t } = useTranslation();
  const mobile = isMobileRuntimeActive();
  const container = useRef<HTMLSpanElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [admitted, setAdmitted] = useState(!mobile);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const failPreview = useCallback(() => setStatus("error"), []);
  const url = asset.url;
  const inViewport = useMobilePreviewVisibility(container, mobile && Boolean(url));
  const visible = !mobile || inViewport;
  const video = asset.kind === "video";
  const failed = !url || status === "error";
  const showMedia = url && admitted && !failed;
  const mediaClassName = cn("h-full w-full object-cover", className);

  useEffect(() => {
    if (mobile && visible && url && status === "loading") {
      return queueMobilePreview(() => setAdmitted(true));
    }
  }, [mobile, visible, url, status]);

  return (
    <span ref={container} className="relative block h-full w-full">
      {showMedia && video && mobile ? (
        <>
          <canvas ref={canvas} className={cn(mediaClassName, status !== "ready" && "opacity-0")} aria-hidden />
          {status === "loading" ? (
            <VideoFrame
              src={url}
              onFrame={(element) => {
                const target = canvas.current;
                setStatus(target && drawVideoFrame(element, target) ? "ready" : "error");
              }}
              onError={failPreview}
            />
          ) : null}
        </>
      ) : showMedia && video ? (
        <video
          src={url}
          muted
          playsInline
          preload="metadata"
          aria-label={compact ? undefined : asset.name}
          onLoadedData={() => setStatus("ready")}
          onError={failPreview}
          className={mediaClassName}
        />
      ) : showMedia ? (
        <img
          src={url}
          alt={compact ? "" : (asset.name ?? "")}
          loading={mobile ? "eager" : "lazy"}
          decoding="async"
          draggable={false}
          onLoad={() => setStatus("ready")}
          onError={failPreview}
          className={cn(mediaClassName, status !== "ready" && "opacity-0")}
        />
      ) : null}
      {failed ? (
        <span className="absolute inset-0 flex flex-col items-center justify-center gap-1 px-2 text-center text-muted-foreground" title={asset.name}>
          <CircleAlert className="h-4 w-4" aria-hidden />
          <span className={cn("text-[10px]", compact && "sr-only")}>
            {t("thread.generatedAssets.previewUnavailable", { defaultValue: "Preview unavailable" })}
          </span>
        </span>
      ) : video && (status === "ready" || !mobile) ? (
        <span className="pointer-events-none absolute inset-0 grid place-items-center bg-black/20">
          <Play className={cn("fill-white text-white/90", compact ? "h-4 w-4" : "h-6 w-6")} aria-hidden />
        </span>
      ) : visible && status === "loading" ? (
        <MediaLoadingIndicator />
      ) : null}
    </span>
  );
}
