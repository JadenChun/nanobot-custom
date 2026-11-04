const DEFAULT_HTTP_TIMEOUT_MS = 20_000;

import { getMobileRuntime, mobileOwnerCapability, mobileStandardDispatchCapability } from "./mobile-runtime";

export async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs: number = DEFAULT_HTTP_TIMEOUT_MS,
): Promise<Response> {
  const mobile = getMobileRuntime();
  if (mobile) {
    return mobileFetch(mobile, input, init);
  }
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    return fetch(input, init);
  }

  const controller = typeof AbortController !== "undefined"
    ? new AbortController()
    : null;
  const externalSignal = init.signal;
  const abortFromExternal = () => controller?.abort();
  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  if (controller && externalSignal) {
    if (externalSignal.aborted) {
      controller.abort();
    } else {
      externalSignal.addEventListener("abort", abortFromExternal, { once: true });
    }
  }

  try {
    const request = fetch(input, {
      ...init,
      signal: controller?.signal ?? externalSignal,
    });
    const timeout = new Promise<Response>((_, reject) => {
      timeoutId = setTimeout(() => {
        reject(new Error(`Request timed out after ${timeoutMs}ms`));
        controller?.abort();
      }, timeoutMs);
    });
    return await Promise.race([request, timeout]);
  } finally {
    if (timeoutId !== undefined) clearTimeout(timeoutId);
    externalSignal?.removeEventListener("abort", abortFromExternal);
  }
}


async function mobileFetch(
  runtime: ReturnType<typeof getMobileRuntime> & object,
  input: RequestInfo | URL,
  init: RequestInit,
): Promise<Response> {
  const method = (init.method ?? "GET").toUpperCase();
  if (method !== "GET") {
    throw new Error("mobile operation unavailable over the read bridge");
  }
  const raw = typeof input === "string" ? input : input instanceof URL ? input.pathname : input.url;
  const url = new URL(raw, typeof window === "undefined" ? "https://mobile.local" : window.location.origin);
  if (url.pathname === "/api/sessions") {
    const body = await runtime.read("/mobile/v1/read");
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  if (url.pathname === "/api/settings") {
    if (mobileOwnerCapability()) {
      return mobileFeatureRequest(runtime, `${url.pathname}${url.search}`);
    }
    return mobileJsonRequest(runtime, "mobile_model_settings", "/mobile/v1/model-settings");
  }
  if (url.pathname === "/api/workspaces") {
    return mobileJsonRequest(runtime, "mobile_workspaces", "/mobile/v1/workspaces");
  }
  if (isMobileChatApiPath(url.pathname)) {
    return mobileApiRequest(runtime, `${url.pathname}${url.search}`);
  }
  const thread = /^\/api\/sessions\/([^/]+)\/webui-thread$/.exec(url.pathname);
  if (thread) {
    const key = decodeURIComponent(thread[1]);
    return mobileJsonRequest(
      runtime,
      "mobile_thread",
      `/mobile/v1/sessions/${encodeURIComponent(key)}/thread${url.search}`,
      { allowNotFound: true },
    );
  }
  const media = /^\/api\/media\/([A-Za-z0-9_-]+)\/([A-Za-z0-9_-]+)$/.exec(url.pathname);
  if (media) {
    const response = await runtime.request({
      operation: "mobile_media",
      method: "GET",
      path: `/mobile/v1/media/${media[1]}/${media[2]}`,
      headers: init.headers as Record<string, string> | undefined,
    });
    if (response.status < 200 || response.status >= 300) {
      throw new Error(`mobile media failed: HTTP ${response.status}`);
    }
    return new Response(decodeBase64Bytes(response.bodyBase64).buffer as ArrayBuffer, {
      status: response.status,
      headers: response.headers,
    });
  }
  if (isStandardReadPath(url.pathname)) {
    return mobileFeatureRequest(runtime, `${url.pathname}${url.search}`);
  }
  throw new Error(`mobile operation unavailable for ${url.pathname}`);
}

function isStandardReadPath(path: string): boolean {
  return path.startsWith("/api/");
}

async function mobileStandardDispatch(
  runtime: ReturnType<typeof getMobileRuntime> & object,
  path: string,
): Promise<Awaited<ReturnType<typeof runtime.request>> | null> {
  if (!mobileOwnerCapability() || !mobileStandardDispatchCapability()) return null;
  const response = await runtime.request({
    operation: "mobile_dispatch",
    method: "GET",
    path: `/mobile/v1/dispatch?path=${encodeURIComponent(path)}`,
  });
  if ([400, 404, 405, 503].includes(response.status)) return null;
  return response;
}

async function mobileFeatureRequest(
  runtime: ReturnType<typeof getMobileRuntime> & object,
  path: string,
): Promise<Response> {
  const dispatched = await mobileStandardDispatch(runtime, path);
  const response = dispatched ?? await runtime.request({
    operation: "mobile_feature",
    method: "GET",
    path: `/mobile/v1/feature?path=${encodeURIComponent(path)}`,
  });
  if (response.status === 403) {
    throw new Error("owner_required: this action needs the full owner capability");
  }
  if (response.status < 200 || response.status >= 300) {
    throw new Error(`mobile feature failed: HTTP ${response.status}`);
  }
  return new Response(decodeBase64Text(response.bodyBase64), {
    status: response.status,
    headers: { "Content-Type": "application/json" },
  });
}

function isMobileChatApiPath(path: string): boolean {
  return (
    path === "/api/commands"
    || path === "/api/webui/sidebar-state"
    || /^\/api\/sessions\/[^/]+\/(file-preview|automations)$/.test(path)
  );
}

async function mobileApiRequest(
  runtime: ReturnType<typeof getMobileRuntime> & object,
  path: string,
): Promise<Response> {
  const dispatched = await mobileStandardDispatch(runtime, path);
  const response = dispatched ?? await runtime.request({
    operation: "mobile_api",
    method: "GET",
    path: `/mobile/v1/api?path=${encodeURIComponent(path)}`,
  });
  return new Response(decodeBase64Text(response.bodyBase64), {
    status: response.status,
    headers: { "Content-Type": "application/json" },
  });
}

async function mobileJsonRequest(
  runtime: ReturnType<typeof getMobileRuntime> & object,
  operation: string,
  path: string,
  options: { allowNotFound?: boolean } = {},
): Promise<Response> {
  const response = await runtime.request({ operation, method: "GET", path });
  if (response.status === 404 && options.allowNotFound) {
    return new Response("", { status: 404 });
  }
  if (response.status < 200 || response.status >= 300) {
    throw new Error(`${operation} failed: HTTP ${response.status}`);
  }
  return new Response(decodeBase64Text(response.bodyBase64), {
    status: response.status,
    headers: { "Content-Type": "application/json" },
  });
}

function decodeBase64Text(value: string): string {
  return new TextDecoder().decode(decodeBase64Bytes(value));
}

function decodeBase64Bytes(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}
