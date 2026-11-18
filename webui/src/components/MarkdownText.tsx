import {
  Component,
  Suspense,
  lazy,
  memo,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import { cn } from "@/lib/utils";

interface MarkdownTextProps {
  children: string;
  className?: string;
  streaming?: boolean;
  preserveStreamingLayout?: boolean;
  revealOnHistoryEntry?: boolean;
  onOpenFilePreview?: (path: string) => void;
}

const loadMarkdownRenderer = () => import("@/components/MarkdownTextRenderer");
const LazyMarkdownRenderer = lazy(loadMarkdownRenderer);

const MemoizedMarkdownRenderer = memo(function MemoizedMarkdownRenderer({
  source,
  className,
  highlightCode,
  streaming,
  revealOnHistoryEntry,
  onOpenFilePreview,
}: {
  source: string;
  className?: string;
  highlightCode: boolean;
  streaming: boolean;
  revealOnHistoryEntry: boolean;
  onOpenFilePreview?: (path: string) => void;
}) {
  return (
    <LazyMarkdownRenderer
      className={className}
      highlightCode={highlightCode}
      streaming={streaming}
      revealOnHistoryEntry={revealOnHistoryEntry}
      onOpenFilePreview={onOpenFilePreview}
    >
      {source}
    </LazyMarkdownRenderer>
  );
});

class MarkdownRendererBoundary extends Component<
  { children: ReactNode; fallback: ReactNode; resetKey: string },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidUpdate(previous: Readonly<{ resetKey: string }>) {
    if (this.state.failed && previous.resetKey !== this.props.resetKey) {
      this.setState({ failed: false });
    }
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

export function preloadMarkdownText(): Promise<void> {
  return loadMarkdownRenderer().then(() => undefined);
}

/** Lazy boundary for the heavier GFM, math, and code renderer. */
export function MarkdownText({
  children,
  className,
  streaming = false,
  preserveStreamingLayout = false,
  revealOnHistoryEntry = false,
  onOpenFilePreview,
}: MarkdownTextProps) {
  const renderedSource = children;
  const [initialRevealSource] = useState(() => (
    revealOnHistoryEntry && !streaming && renderedSource.length <= 2_048
      ? renderedSource
      : null
  ));
  const animateHistory = initialRevealSource !== null
    && renderedSource === initialRevealSource;
  const renderPhase = streaming ? "streaming" : "complete";
  const highlightCode = !streaming;
  const renderWithStreamingLayout = streaming || preserveStreamingLayout;

  useEffect(() => {
    if (streaming) void preloadMarkdownText();
  }, [streaming]);

  const plainFallback = (
    <div
      className={cn(
        "whitespace-pre-wrap break-words leading-relaxed text-foreground/92",
        streaming && "streaming-text-fallback",
        className,
      )}
    >
      {renderedSource}
    </div>
  );

  return (
    <MarkdownRendererBoundary resetKey={renderPhase} fallback={plainFallback}>
      <Suspense fallback={plainFallback}>
        <MemoizedMarkdownRenderer
          source={renderedSource}
          className={className}
          highlightCode={highlightCode}
          streaming={renderWithStreamingLayout}
          revealOnHistoryEntry={animateHistory}
          onOpenFilePreview={onOpenFilePreview}
        />
      </Suspense>
    </MarkdownRendererBoundary>
  );
}
