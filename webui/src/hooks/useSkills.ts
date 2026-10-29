import { useEffect, useState } from "react";

import { fetchSkills } from "@/lib/api";
import { isSkillsPayload, SKILLS_CHANGED_EVENT } from "@/lib/skill-events";
import type { SkillsPayload, SkillSummary } from "@/lib/types";

export function useSkillsCatalog(getToken: () => string): SkillsPayload {
  const [catalog, setCatalog] = useState<SkillsPayload>({
    skills: [],
    catalog_complete: true,
  });

  useEffect(() => {
    let cancelled = false;
    let payloadVersion = 0;
    const refresh = () => {
      const version = payloadVersion;
      fetchSkills(getToken())
        .then((payload) => {
          if (!cancelled && version === payloadVersion) {
            setCatalog((current) => mergeSkillsCatalog(current, payload));
          }
        })
        .catch(() => {
          if (!cancelled && version === payloadVersion) {
            setCatalog((current) => ({ ...current, skills: [] }));
          }
        });
    };
    const onSkillsChanged = (event: Event) => {
      const payload = (event as CustomEvent<unknown>).detail;
      if (!cancelled && isSkillsPayload(payload)) {
        payloadVersion += 1;
        setCatalog((current) => mergeSkillsCatalog(current, payload));
      }
    };

    refresh();
    window.addEventListener(SKILLS_CHANGED_EVENT, onSkillsChanged);
    return () => {
      cancelled = true;
      window.removeEventListener(SKILLS_CHANGED_EVENT, onSkillsChanged);
    };
  }, [getToken]);

  return catalog;
}

export function useSkills(getToken: () => string): SkillSummary[] {
  return useSkillsCatalog(getToken).skills;
}

function mergeSkillsCatalog(current: SkillsPayload, next: SkillsPayload): SkillsPayload {
  const hasCompleteness = typeof next.catalog_complete === "boolean";
  const catalogComplete = hasCompleteness
    ? next.catalog_complete
    : current.catalog_complete;
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
