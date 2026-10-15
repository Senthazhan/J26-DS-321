// import { PagePlaceholder } from "@/components/page-placeholder"

// export default function Page() {
//   return <PagePlaceholder id="behavior" />
// }

"use client";
// Behavior tab (Component 2): open http://localhost:3000/behavior
import { useEffect, useState } from "react";

import Dashboard from "./Dashboard";
import styles from "./c2.module.css";
import type { Summary, TeamInfo } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function BehaviorPage() {
  const [teams, setTeams] = useState<TeamInfo[]>([]);
  const [team, setTeam] = useState("");
  const [days, setDays] = useState(30);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  // 1) Load the list of teams once
  useEffect(() => {
    fetch(`${API}/c2/teams`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`API error ${r.status}`))))
      .then((list: TeamInfo[]) => {
        setTeams(list);
        if (list.length) setTeam(list[0].team_id);
        else setLoading(false);
      })
      .catch(() => {
        setError(`Could not reach the Component 2 API at ${API}. Is "uvicorn main:app" running?`);
        setLoading(false);
      });
  }, []);

  // 2) Load the chosen team's summary whenever the team or window changes
  useEffect(() => {
    if (!team) return;
    setLoading(true);
    setError("");
    fetch(`${API}/c2/teams/${encodeURIComponent(team)}/summary?days=${days}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`API error ${r.status}`))))
      .then((s: Summary) => setSummary(s))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [team, days]);

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <h1 className={styles.title}>Behavior</h1>
        <p className={styles.subtitle}>Lifecycle stages, plan drift and member roles</p>
        <div className={styles.filters}>
          <label>
            Team
            <select value={team} onChange={(e) => setTeam(e.target.value)} disabled={!teams.length}>
              {teams.map((t) => (
                <option key={t.team_id} value={t.team_id}>{t.team_id}</option>
              ))}
            </select>
          </label>
          <label>
            Window
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              <option value={14}>Last 14 days</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
            </select>
          </label>
        </div>
      </header>

      {error && <p className={`${styles.state} ${styles.error}`} role="alert">{error}</p>}
      {!error && loading && <p className={styles.state}>Loading…</p>}
      {!error && !loading && summary && <Dashboard summary={summary} />}
      {!error && !loading && !summary && <p className={styles.state}>No teams found in the database yet.</p>}
    </div>
  );
}