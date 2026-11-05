import { useCallback, useEffect, useRef, useState } from "react";

import { fetchSkills } from "@/lib/api";
import { isSkillsPayload, skillsEventMatchesScope, SKILLS_CHANGED_EVENT } from "@/lib/skill-events";
import type { SkillsPayload, SkillSummary } from "@/lib/types";

interface CatalogState {
  token: string;
  catalog: SkillsPayload;
  loading: boolean;
  error: string | null;
}

export function useSkillsCatalog(getToken: () => string) {
  // getToken is stable in ClientProvider; its returned credentials are not.
  const token = getToken();
  const tokenRef = useRef(token);
  tokenRef.current = token;
  const versionRef = useRef(0);
  const [state, setState] = useState<CatalogState>({
    token, catalog: { skills: [] }, loading: true, error: null,
  });

  const refresh = useCallback(() => {
    const version = ++versionRef.current;
    setState((current) => ({
      token,
      catalog: current.token === token ? current.catalog : { skills: [] },
      loading: true,
      error: null,
    }));
    void fetchSkills(token)
      .then((payload) => {
        if (version !== versionRef.current || tokenRef.current !== token) return;
        setState((current) => ({
          token,
          catalog: mergeSkillsCatalog(current.catalog, payload),
          loading: false,
          error: null,
        }));
      })
      .catch(() => {
        if (version !== versionRef.current || tokenRef.current !== token) return;
        setState((current) => ({
          ...current,
          loading: false,
          error: "Unable to load skills. Check your connection and retry.",
        }));
      });
  }, [token]);

  useEffect(() => {
    const onSkillsChanged = (event: Event) => {
      const scopeMatches = skillsEventMatchesScope(event, token);
      if (scopeMatches === undefined) {
        // Unscoped invalidations can refresh, but cannot supply private rows.
        refresh();
        return;
      }
      if (!scopeMatches) return;
      const payload = (event as CustomEvent<unknown>).detail;
      if (tokenRef.current === token && isSkillsPayload(payload)) {
        versionRef.current += 1;
        setState((current) => ({
          token,
          catalog: mergeSkillsCatalog(
            current.token === token ? current.catalog : { skills: [] }, payload,
          ),
          loading: false,
          error: null,
        }));
      }
    };
    refresh();
    window.addEventListener(SKILLS_CHANGED_EVENT, onSkillsChanged);
    return () => {
      versionRef.current += 1;
      window.removeEventListener(SKILLS_CHANGED_EVENT, onSkillsChanged);
    };
  }, [token, refresh]);

  const current = state.token === token ? state : {
    token, catalog: { skills: [] }, loading: true, error: null,
  };
  return {
    ...current.catalog,
    catalog_complete: current.catalog.catalog_complete ?? false,
    loading: current.loading,
    error: current.error,
    refresh,
  };
}

export function useSkills(getToken: () => string): SkillSummary[] {
  return useSkillsCatalog(getToken).skills;
}

function mergeSkillsCatalog(current: SkillsPayload, next: SkillsPayload): SkillsPayload {
  const hasCompleteness = typeof next.catalog_complete === "boolean";
  const catalogComplete = hasCompleteness
    ? next.catalog_complete
    : current.catalog_complete ?? true;
  return {
    skills: catalogComplete === false
      ? next.skills.map((skill) =>
        skill.source !== "workspace" && skill.verification === undefined
          ? {
              ...skill,
              verification: "unverified",
              available: false,
              unavailable_reason:
                skill.unavailable_reason
                ?? "A higher-priority configured skill library could not be checked.",
            }
          : skill,
      )
      : next.skills,
    catalog_complete: catalogComplete,
    catalog_notice: hasCompleteness
      ? next.catalog_complete === true
        ? undefined
        : next.catalog_notice ?? current.catalog_notice
      : current.catalog_notice,
  };
}
