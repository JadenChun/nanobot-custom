import { Component, type ErrorInfo, type ReactNode } from "react";

const CHUNK_LOAD_ERROR_PATTERNS = [
  /failed to fetch dynamically imported module/i,
  /importing a module script failed/i,
  /error loading dynamically imported module/i,
  /loading (?:css )?chunk .* failed/i,
  /chunkloaderror/i,
  /unable to preload css/i,
];

export function isChunkLoadError(error: unknown): boolean {
  if (!error) return false;
  const name = error instanceof Error ? error.name : "";
  const message =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : "";
  const combined = `${name} ${message}`;
  return CHUNK_LOAD_ERROR_PATTERNS.some((pattern) => pattern.test(combined));
}

function resetKeysChanged(
  prevKeys: readonly unknown[] | undefined,
  nextKeys: readonly unknown[] | undefined,
): boolean {
  if (prevKeys === nextKeys) return false;
  if (!prevKeys || !nextKeys || prevKeys.length !== nextKeys.length) return true;
  for (let i = 0; i < prevKeys.length; i += 1) {
    if (!Object.is(prevKeys[i], nextKeys[i])) return true;
  }
  return false;
}

export interface ErrorBoundaryFallbackProps {
  error: Error;
  isChunkError: boolean;
  reset: () => void;
}

interface ErrorBoundaryProps {
  children: ReactNode;
  resetKeys?: readonly unknown[];
  onBack?: () => void;
  onError?: (error: Error, info: ErrorInfo) => void;
  onReload?: () => void;
  backLabel?: string;
  reloadLabel?: string;
  retryLabel?: string;
  title?: string;
  chunkErrorTitle?: string;
  chunkErrorDescription?: string;
  fallback?: (props: ErrorBoundaryFallbackProps) => ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: unknown): ErrorBoundaryState {
    return {
      error: error instanceof Error ? error : new Error(String(error ?? "Unknown error")),
    };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[nanobot-webui] Uncaught render error:", error, info.componentStack);
    this.props.onError?.(error, info);
  }

  componentDidUpdate(prevProps: ErrorBoundaryProps): void {
    if (this.state.error && resetKeysChanged(prevProps.resetKeys, this.props.resetKeys)) {
      this.setState({ error: null });
    }
  }

  private handleReset = (): void => {
    this.setState({ error: null });
  };

  private handleBack = (): void => {
    this.setState({ error: null });
    this.props.onBack?.();
  };

  private handleReload = (): void => {
    if (this.props.onReload) {
      this.props.onReload();
      return;
    }
    if (typeof window !== "undefined") {
      window.location.reload();
    }
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) {
      return this.props.children;
    }

    const chunkError = isChunkLoadError(error);
    if (this.props.fallback) {
      return this.props.fallback({
        error,
        isChunkError: chunkError,
        reset: this.handleReset,
      });
    }

    const heading = chunkError
      ? (this.props.chunkErrorTitle ?? "This view could not be loaded")
      : (this.props.title ?? "Something went wrong");
    const description = chunkError
      ? (this.props.chunkErrorDescription
        ?? "A newer version of the WebUI may have been built while this tab was open. Reload to load the latest files, or return to your active chat.")
      : (error.message || "An unexpected error occurred while rendering this view.");

    return (
      <div
        role="alert"
        className="flex h-full w-full flex-1 items-center justify-center bg-background px-6 py-10 text-foreground"
      >
        <div className="w-full max-w-md rounded-xl border border-border bg-card p-6 shadow-sm">
          <h2 className="text-base font-semibold text-foreground">{heading}</h2>
          <p className="mt-2 break-words text-sm text-muted-foreground">{description}</p>
          <div className="mt-5 flex flex-wrap items-center gap-2">
            {this.props.onBack ? (
              <button
                type="button"
                onClick={this.handleBack}
                className="inline-flex h-9 items-center justify-center rounded-md border border-border bg-background px-3.5 text-sm font-medium text-foreground transition-colors hover:bg-muted"
              >
                {this.props.backLabel ?? "Back to chat"}
              </button>
            ) : null}
            <button
              type="button"
              onClick={this.handleReload}
              className="inline-flex h-9 items-center justify-center rounded-md bg-primary px-3.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
            >
              {this.props.reloadLabel ?? "Reload"}
            </button>
            {!chunkError ? (
              <button
                type="button"
                onClick={this.handleReset}
                className="inline-flex h-9 items-center justify-center rounded-md border border-border bg-background px-3.5 text-sm font-medium text-foreground transition-colors hover:bg-muted"
              >
                {this.props.retryLabel ?? "Try again"}
              </button>
            ) : null}
          </div>
        </div>
      </div>
    );
  }
}
