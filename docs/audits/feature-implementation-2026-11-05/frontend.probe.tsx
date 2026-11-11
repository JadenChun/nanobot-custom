import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useGeneratedAssetLibrary } from '@/hooks/useGeneratedAssetLibrary';
import { useSessionGeneratedAssets } from '@/hooks/useSessionGeneratedAssets';
import { useSkillsCatalog } from '@/hooks/useSkills';
import { useClipboardAndDrop } from '@/hooks/useClipboardAndDrop';
import { fetchSessionGeneratedAssets, fetchAllGeneratedAssets, fetchSkills } from '@/lib/api';

vi.mock('@/lib/api', () => ({ fetchSessionGeneratedAssets: vi.fn(), fetchAllGeneratedAssets: vi.fn(), fetchSkills: vi.fn() }));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}
function page(id: string, next: string | null = null) {
  return { schemaVersion: 1, sessionKey: id, total: 2, next,
    groups: [{ id, turnId: id, toolCallId: id, toolName: 'generate_image', sequence: 1, artifacts: [] }] };
}
beforeEach(() => vi.resetAllMocks());
afterEach(() => cleanup());

describe('feature audit: desired behavior probes', () => {
  it('does not append an old chat page after switching sessions', async () => {
    const more = deferred<ReturnType<typeof page>>();
    vi.mocked(fetchSessionGeneratedAssets)
      .mockResolvedValueOnce(page('a-first', 'a-cursor'))
      .mockReturnValueOnce(more.promise)
      .mockResolvedValueOnce(page('b-first'));
    const { result, rerender } = renderHook(
      ({ sessionKey }) => useSessionGeneratedAssets(true, 'token', sessionKey),
      { initialProps: { sessionKey: 'websocket:a' } },
    );
    await waitFor(() => expect(result.current.groups[0]?.id).toBe('a-first'));
    act(() => result.current.loadMore());
    rerender({ sessionKey: 'websocket:b' });
    await waitFor(() => expect(result.current.groups[0]?.id).toBe('b-first'));
    await act(async () => { more.resolve(page('a-old-more', 'a-next')); await more.promise; });
    expect(result.current.groups.map((group) => group.id)).toEqual(['b-first']);
    expect(result.current.next).toBeNull();
  });

  it('does not append a stale page after refreshing the same chat', async () => {
    const more = deferred<ReturnType<typeof page>>();
    vi.mocked(fetchSessionGeneratedAssets)
      .mockResolvedValueOnce(page('old-first', 'old-cursor'))
      .mockReturnValueOnce(more.promise)
      .mockResolvedValueOnce(page('refreshed-first'));
    const { result } = renderHook(() => useSessionGeneratedAssets(true, 'token', 'websocket:a'));
    await waitFor(() => expect(result.current.groups[0]?.id).toBe('old-first'));
    act(() => result.current.loadMore());
    act(() => result.current.refresh());
    await waitFor(() => expect(result.current.groups[0]?.id).toBe('refreshed-first'));
    await act(async () => { more.resolve(page('stale-more')); await more.promise; });
    expect(result.current.groups.map((group) => group.id)).toEqual(['refreshed-first']);
  });

  it('does not append a stale global library page after refresh', async () => {
    const more = deferred<ReturnType<typeof page>>();
    vi.mocked(fetchAllGeneratedAssets)
      .mockResolvedValueOnce(page('library-first', 'old-cursor'))
      .mockReturnValueOnce(more.promise)
      .mockResolvedValueOnce(page('library-refreshed'));
    const { result } = renderHook(() => useGeneratedAssetLibrary(true, 'token'));
    await waitFor(() => expect(result.current.groups[0]?.id).toBe('library-first'));
    act(() => result.current.loadMore());
    act(() => result.current.refresh());
    await waitFor(() => expect(result.current.groups[0]?.id).toBe('library-refreshed'));
    await act(async () => { more.resolve(page('library-stale')); await more.promise; });
    expect(result.current.groups.map((group) => group.id)).toEqual(['library-refreshed']);
  });

  it('keeps the last successful skills catalog when a refresh fails', async () => {
    const skill = { name: 'audit-skill', description: 'Local audit fixture', source: 'workspace', available: true };
    vi.mocked(fetchSkills)
      .mockResolvedValueOnce({ skills: [skill], catalog_complete: true })
      .mockRejectedValueOnce(new Error('HTTP 503'));
    const firstTokenGetter = () => 'token';
    const { result, rerender } = renderHook(
      ({ getToken }) => useSkillsCatalog(getToken),
      { initialProps: { getToken: firstTokenGetter } },
    );
    await waitFor(() => expect(result.current.skills).toEqual([skill]));
    rerender({ getToken: () => 'token' });
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(fetchSkills).toHaveBeenCalledTimes(2);
    expect(result.current.skills).toEqual([skill]);
  });

  it('does not report a complete empty skills catalog after an HTTP failure', async () => {
    vi.mocked(fetchSkills).mockRejectedValue(new Error('HTTP 503'));
    const getToken = () => 'token';
    const { result } = renderHook(() => useSkillsCatalog(getToken));
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(fetchSkills).toHaveBeenCalledOnce();
    expect(result.current.catalog_complete).toBe(false);
  });

  it('preserves plain text pasted alongside a supported attachment', () => {
    const onFiles = vi.fn();
    const { result } = renderHook(() => useClipboardAndDrop(onFiles));
    const file = new File(['image'], 'reference.png', { type: 'image/png' });
    const preventDefault = vi.fn();
    const event = {
      preventDefault,
      clipboardData: {
        items: [{ kind: 'file', getAsFile: () => file }],
        getData: (format: string) => format === 'text/plain' ? 'Please explain this image' : '',
      },
    };
    act(() => result.current.onPaste(event as never));
    expect(onFiles).toHaveBeenCalledWith([file]);
    expect(preventDefault).not.toHaveBeenCalled();
  });
});
