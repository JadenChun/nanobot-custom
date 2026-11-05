import { fetchAllGeneratedAssets } from "@/lib/api";
import { usePagedGeneratedAssets } from "@/hooks/usePagedGeneratedAssets";

const fetchLibraryPage = (token: string, options: { cursor?: string; limit: number }) =>
  fetchAllGeneratedAssets(token, "", options);

export function useGeneratedAssetLibrary(
  active: boolean,
  token: string,
  refreshKey: string | number = 0,
) {
  return usePagedGeneratedAssets(active, token, "library", refreshKey, fetchLibraryPage);
}
