// Shapes of the JSON returned by the Component 2 API (/c2/teams/{team_id}/summary)

export type Severity = "none" | "minor" | "moderate" | "significant" | "no_plan" | "no_evidence" | "unknown";

export interface DriftDimension {
  dimension: "target" | "algorithm" | "metric";
  planned: string | null;
  seen_top: string | null;
  drift_score: number | null;
  severity: Severity;
  explanation: string;
}

export interface Member {
  member_id: string;
  stage_mix: Record<string, number>;
  chat: { participation: number; responsiveness: number; social_impact: number };
  dominant_role: string | null;
  role_trend: { week: string; dominant: string | null }[];
}

export interface Summary {
  team_id: string;
  generated_at: string;
  window: { start: string; end: string };
  stage_breakdown: Record<string, number>;
  lagging_stages: string[];
  drift: { overall_severity: Severity; dimensions: DriftDimension[] };
  members: Member[];
}

export interface TeamInfo {
  team_id: string;
  activities: number;
  first_activity: string;
  last_activity: string;
}