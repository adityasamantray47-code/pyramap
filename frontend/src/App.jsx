import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { api } from "./api";
import DangerChip from "./DangerChip";

function issuedDateLine(status) {
  if (!status?.data_through) return null;
  const d = new Date(status.data_through + "T00:00:00");
  return d.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
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
                Model performance on held-out data: ROC-AUC {metrics.metrics.roc_auc ?? "n/a"}, Brier skill vs.
                climatology {metrics.metrics.brier_skill_score ?? "n/a"} · source: {metrics.source}
              </p>
            )}

            <div className="card" style={{ marginTop: 14, overflowX: "auto" }}>
              <table className="outlook-table">
                <thead>
                  <tr>
                    <th>Area</th>
                    <th>Danger</th>
                    <th>Probability</th>
                    <th>Root soil wetness</th>
                    <th>Why</th>
                  </tr>
                </thead>
                <tbody>
                  {outlook.map((row) => (
                    <tr key={row.id} onClick={() => setSelectedArea(row.id)}>
                      <td className="area-id">{row.id}</td>
                      <td>
                        <DangerChip danger_class={row.danger_class} />
                      </td>
                      <td className="num">{Math.round(row.danger_prob * 100)}%</td>
                      <td className="num">
                        {Math.round(row.root_soil_wetness * 100)}%
                        {row.root_soil_wetness < 0.25 ? " (drier, riskier)" : ""}
                      </td>
                      <td className="why">{row.drivers.join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {selectedArea && history && (
              <div className="card history-panel">
                <h3>
                  {selectedArea} · last {history.dates.length} weeks
                </h3>
                <div className="history-row">
                  <b>Weekly fire detections</b>
                  <span>{history.fire.join(", ")}</span>
                </div>
                <div className="history-row">
                  <b>NOAA temp. outlook probability</b>
                  <span>{history.t30.join(", ")}</span>
                </div>
                <div className="history-row">
                  <b>NOAA precip. outlook probability</b>
                  <span>{history.p30.join(", ")}</span>
                </div>
                <div className="history-row">
                  <b>Modeled danger probability</b>
                  <span>{history.dc.join(", ")}</span>
                </div>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
