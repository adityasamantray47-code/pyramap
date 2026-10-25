import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { api } from "./api";
import DangerChip from "./DangerChip";
import StateAreaMap from "./StateAreaMap";
import TrendChart from "./TrendChart";
import { DANGER_HEX } from "./dangerColors";

function issuedDateLine(status) {
  if (!status?.data_through) return null;
  const d = new Date(status.data_through + "T00:00:00");
  return d.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
}

const avg = (nums) => (nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0);

function trendWord(before, after) {
  if (after > before * 1.3) return "rising";
  if (after < before * 0.7) return "easing";
  return "holding fairly steady";
}

function outlookWord(cat) {
  if (cat === "above") return "above-normal";
  if (cat === "below") return "below-normal";
  return "near-normal";
}

// Turns the raw weekly numbers into a few plain-language sentences, read
// alongside the charts below rather than instead of them.
function summarizeHistory(history, soilPct) {
  const fire = history.fire || [];
  const dc = history.dc || [];
  const lines = [];
  if (fire.length === 0) return lines;

  const half = Math.max(1, Math.floor(fire.length / 2));
  const earlyAvg = avg(fire.slice(0, half));
  const recentAvg = avg(fire.slice(-half));
  lines.push(
    `Satellite fire detections here have been ${trendWord(earlyAvg, recentAvg)} over the last ${fire.length} weeks — ` +
      `about ${Math.round(recentAvg)} per week recently, versus ${Math.round(earlyAvg)} earlier in that stretch.`
  );

  const dryWord = soilPct < 25 ? "unusually dry, which makes fire spread more easily" : soilPct < 45 ? "on the dry side" : "fairly moist, which resists fire";
  lines.push(`The ground's root-zone moisture is around ${soilPct}% — ${dryWord}.`);

  const tempCat = history.temp_cat?.at(-1);
  const precipCat = history.precip_cat?.at(-1);
  if (tempCat && precipCat) {
    lines.push(
      `NOAA's latest monthly outlook calls for ${outlookWord(tempCat)} temperatures and ${outlookWord(precipCat)} rainfall here over the next month.`
    );
  }

  if (dc.length >= 2) {
    const first = dc[0];
    const last = dc.at(-1);
    lines.push(
      `Our model's estimated fire-danger probability has gone from ${Math.round(first * 100)}% to ${Math.round(last * 100)}% ` +
        `over this period, ${trendWord(first, last)}.`
    );
  }
  return lines;
}

export default function App() {
  const [states, setStates] = useState([]);
  const [status, setStatus] = useState(null);
  const [selected, setSelected] = useState(null);
  const [outlook, setOutlook] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [selectedArea, setSelectedArea] = useState(null);
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.states().then(setStates).catch((e) => setError(e.message));
    api.dataStatus().then(setStatus).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    setSelectedArea(null);
    setHistory(null);
    Promise.all([api.outlook(selected), api.metrics(selected)])
      .then(([o, m]) => {
        setOutlook(o);
        setMetrics(m);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [selected]);

  useEffect(() => {
    if (!selectedArea) return;
    api.areaHistory(selectedArea).then(setHistory).catch(() => setHistory(null));
  }, [selectedArea]);

  const selectedStateMeta = useMemo(() => states.find((s) => s.code === selected), [states, selected]);
  const selectedAreaMeta = useMemo(() => outlook.find((o) => o.id === selectedArea), [outlook, selectedArea]);

  return (
    <div style={{ background: "var(--bg)", minHeight: "100vh" }}>
      <div className="masthead">
        <div className="masthead-inner">
          <div className="brand-mark">🔥</div>
          <div>
            <p className="brand-title">Long-Lead Wildfire Outlook</p>
            <p className="brand-sub">A 30-day-ahead wildfire danger outlook for the United States</p>
          </div>
        </div>
      </div>

      {status && (
        <div className="issued-bar">
          <div className="issued-bar-inner">
            <span>
              Based on data through <b>{issuedDateLine(status) || "—"}</b>
            </span>
            <span>·</span>
            <span>
              <b className="num">{status.states_with_real_data.length}</b> states on live NASA/NOAA data,{" "}
              <b className="num">{status.states_with_synthetic_data.length}</b> on practice data
            </span>
          </div>
        </div>
      )}

      <main className="page-wrap">
        <div className="mb-6">
          <p className="h-small">Public outlook · updated weekly</p>
          <h1 className="h-section" style={{ fontSize: "1.7rem" }}>
            Where fire danger is building, 30 days out
          </h1>
          <p className="lede">
            Built from NASA satellite fire detections, NASA weather &amp; soil-moisture data, and NOAA's
            monthly climate outlook. Pick a state to see its forecast areas below.
          </p>
          <div className="scale">
            <span className="lo">Low</span>
            <span className="mo">Moderate</span>
            <span className="hi">High</span>
            <span className="vh">Very High</span>
            <span className="ex">Extreme</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[1fr_320px] gap-6 items-start">
          <div className="card map-card">
            <p className="hint">Click a state, or choose one from the list.</p>
            <MapContainer center={[39.5, -98.35]} zoom={4} style={{ height: 460, width: "100%" }}>
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution="&copy; OpenStreetMap contributors"
              />
              {states.map((s) => (
                <CircleMarker
                  key={s.code}
                  center={[s.lat, s.lon]}
                  radius={selected === s.code ? 10 : 6}
                  pathOptions={{
                    color: s.has_real_data ? "#ff6b35" : "#78787f",
                    fillColor: s.has_real_data ? "#ff6b35" : "#78787f",
                    fillOpacity: 0.75,
                    weight: selected === s.code ? 3 : 1,
                  }}
                  eventHandlers={{ click: () => setSelected(s.code) }}
                >
                  <Tooltip>
                    {s.name}
                    {s.has_real_data ? "" : " (practice data)"}
                  </Tooltip>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>

          <div className="card">
            <p className="h-small" style={{ padding: "12px 14px 0" }}>
              Choose a state
            </p>
            <div className="state-list">
              {states.map((s) => (
                <button
                  key={s.code}
                  onClick={() => setSelected(s.code)}
                  className={`state-list-item${selected === s.code ? " active" : ""}`}
                >
                  {s.name} {!s.has_real_data && <span className="tag">(practice)</span>}
                </button>
              ))}
            </div>
          </div>
        </div>

        {error && <p style={{ color: "var(--fd-extreme)", marginTop: 16 }}>{error}</p>}
        {loading && <p style={{ marginTop: 16, color: "var(--ink-2)" }}>Loading outlook for {selectedStateMeta?.name}…</p>}

        {!loading && selected && outlook.length > 0 && (
          <section style={{ marginTop: 28 }}>
            <p className="h-small">{selectedStateMeta?.name}</p>
            <h2 className="h-section">30-day outlook by forecast area</h2>
            {metrics && (
              <p className="lede" style={{ marginTop: 6, fontSize: "0.88rem" }}>
                In tests on past data, this forecast told high-risk weeks apart from low-risk ones about{" "}
                {Math.round((metrics.metrics.roc_auc ?? 0.5) * 100)}% as well as a perfect model would (ROC-AUC{" "}
                {metrics.metrics.roc_auc ?? "n/a"}), using {metrics.source === "real" ? "real NASA/NOAA" : "practice"}{" "}
                data. Mainly driven by: {outlook[0]?.drivers?.join(", ") || "seasonal fire history"}.
              </p>
            )}

            <div className="card map-card" style={{ marginTop: 14 }}>
              <p className="hint">
                Each shape is one forecast area within {selectedStateMeta?.name} — click it (or a row below) to see
                what's happening there.
              </p>
              <StateAreaMap areas={outlook} selectedArea={selectedArea} onSelectArea={setSelectedArea} />
            </div>

            <div className="card" style={{ marginTop: 14, overflowX: "auto" }}>
              <table className="outlook-table">
                <thead>
                  <tr>
                    <th>Area</th>
                    <th>Danger</th>
                    <th>Probability</th>
                    <th>Root soil wetness</th>
                  </tr>
                </thead>
                <tbody>
                  {outlook.map((row) => (
                    <tr
                      key={row.id}
                      onClick={() => setSelectedArea(row.id)}
                      style={row.id === selectedArea ? { background: "var(--surface-2)" } : undefined}
                    >
                      <td className="area-id">
                        {row.region || row.id}
                        <div style={{ fontWeight: 400, fontSize: "0.76rem", color: "var(--ink-3)" }}>{row.id}</div>
                      </td>
                      <td>
                        <DangerChip danger_class={row.danger_class} />
                      </td>
                      <td className="num">{Math.round(row.danger_prob * 100)}%</td>
                      <td className="num">
                        {Math.round(row.root_soil_wetness * 100)}%
                        {row.root_soil_wetness < 0.25 ? " (drier, riskier)" : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="hint" style={{ padding: "0 14px 12px" }}>
                Click a row (or the map above) to see the full picture for that area.
              </p>
            </div>

            {selectedArea && history && selectedAreaMeta && (
              <div className="card history-panel">
                <h3>
                  {selectedAreaMeta.region || selectedArea}{" "}
                  <span style={{ fontWeight: 400, fontSize: "0.85rem", color: "var(--ink-3)" }}>
                    ({selectedArea})
                  </span>
                </h3>

                <div style={{ marginBottom: 14 }}>
                  <DangerChip danger_class={selectedAreaMeta.danger_class} />{" "}
                  <span className="num" style={{ marginLeft: 8, color: "var(--ink-2)" }}>
                    {Math.round(selectedAreaMeta.danger_prob * 100)}% estimated chance of a fire week in the next 30
                    days
                  </span>
                </div>

                <p className="h-small" style={{ marginBottom: 6 }}>
                  Why the model thinks this
                </p>
                {(selectedAreaMeta.reasons || []).length > 0 ? (
                  <ul style={{ margin: "0 0 14px", paddingLeft: 18, color: "var(--ink-2)", fontSize: "0.92rem" }}>
                    {selectedAreaMeta.reasons.map((r, i) => (
                      <li key={i} style={{ marginBottom: 6 }}>
                        {r}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="lede" style={{ marginBottom: 14, fontSize: "0.92rem" }}>
                    Conditions here are close to this area's own normal range for the season — nothing stands out as
                    unusually dry, hot, or active right now.
                  </p>
                )}

                <p className="h-small" style={{ marginBottom: 6 }}>
                  In plain terms
                </p>
                {summarizeHistory(history, Math.round(selectedAreaMeta.root_soil_wetness * 100)).map((line, i) => (
                  <p key={i} className="lede" style={{ marginTop: i === 0 ? 0 : 8, fontSize: "0.92rem" }}>
                    {line}
                  </p>
                ))}

                <p className="h-small" style={{ marginTop: 14, marginBottom: -4 }}>
                  The real data behind that
                </p>
                <TrendChart
                  title="Weekly satellite fire detections"
                  dates={history.dates}
                  values={history.fire}
                  color="#ff6b35"
                  formatValue={(v) => Math.round(v)}
                />
                <TrendChart
                  title="Modeled 30-day fire-danger probability"
                  dates={history.dates.slice(-history.dc.length)}
                  values={history.dc.map((v) => v * 100)}
                  color={DANGER_HEX[selectedAreaMeta.danger_class] || "#f94144"}
                  formatValue={(v) => `${Math.round(v)}%`}
                />
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
