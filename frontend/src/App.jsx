import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { api } from "./api";
import DangerChip from "./DangerChip";

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
    <div style={{ background: "var(--paper)", minHeight: "100vh" }}>
      <header style={{ background: "var(--forest)", color: "#fff" }} className="px-4 py-3">
        <div className="max-w-6xl mx-auto flex items-center justify-between flex-wrap gap-2">
          <div>
            <strong className="text-lg font-extrabold">Long-Lead Wildfire Outlook</strong>
            <div className="text-sm opacity-85">30-day-ahead wildfire danger for the US</div>
          </div>
          {status && (
            <div className="text-sm bg-white/10 rounded px-3 py-1">
              Based on data through <b>{status.data_through || "—"}</b>
              {" · "}
              {status.states_with_real_data.length} states with live data,{" "}
              {status.states_with_synthetic_data.length} using practice data
            </div>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 md:grid-cols-[1fr_320px] gap-6 items-start">
          <div
            className="rounded border overflow-hidden"
            style={{ borderColor: "var(--rule)", background: "var(--surface)" }}
          >
            <MapContainer center={[39.5, -98.35]} zoom={4} style={{ height: 480, width: "100%" }}>
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; OpenStreetMap contributors'
              />
              {states.map((s) => (
                <CircleMarker
                  key={s.code}
                  center={[s.lat, s.lon]}
                  radius={selected === s.code ? 10 : 6}
                  pathOptions={{
                    color: s.has_real_data ? "#1f5136" : "#8b968f",
                    fillColor: s.has_real_data ? "#1f5136" : "#8b968f",
                    fillOpacity: 0.7,
                  }}
                  eventHandlers={{ click: () => setSelected(s.code) }}
                >
                  <Tooltip>{s.name}{s.has_real_data ? "" : " (practice data)"}</Tooltip>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>

          <div
            className="rounded border p-3 max-h-[480px] overflow-y-auto"
            style={{ borderColor: "var(--rule)", background: "var(--surface)" }}
          >
            <div className="text-xs font-bold uppercase tracking-wide mb-2" style={{ color: "var(--ink-2)" }}>
              Choose a state
            </div>
            <ul>
              {states.map((s) => (
                <li key={s.code}>
                  <button
                    onClick={() => setSelected(s.code)}
                    className="w-full text-left px-2 py-1.5 rounded text-sm"
                    style={{
                      background: selected === s.code ? "var(--forest-soft)" : "transparent",
                      fontWeight: selected === s.code ? 700 : 400,
                    }}
                  >
                    {s.name} {!s.has_real_data && <span className="opacity-60">(practice)</span>}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {error && <p className="text-red-600 mt-4">{error}</p>}
        {loading && <p className="mt-4">Loading outlook for {selectedStateMeta?.name}…</p>}

        {!loading && selected && outlook.length > 0 && (
          <section className="mt-6">
            <h2 className="text-xl font-bold mb-1">{selectedStateMeta?.name}: 30-day outlook</h2>
            {metrics && (
              <p className="text-sm mb-3" style={{ color: "var(--ink-2)" }}>
                Model: ROC-AUC {metrics.metrics.roc_auc ?? "n/a"}, Brier skill vs. climatology{" "}
                {metrics.metrics.brier_skill_score ?? "n/a"} · source: {metrics.source}
              </p>
            )}
            <div className="overflow-x-auto rounded border" style={{ borderColor: "var(--rule)" }}>
              <table className="w-full text-sm" style={{ background: "var(--surface)" }}>
                <thead>
                  <tr className="text-left border-b" style={{ borderColor: "var(--rule)" }}>
                    <th className="p-2">Area</th>
                    <th className="p-2">Danger</th>
                    <th className="p-2">Probability</th>
                    <th className="p-2">Root soil wetness</th>
                    <th className="p-2">Why</th>
                  </tr>
                </thead>
                <tbody>
                  {outlook.map((row) => (
                    <tr
                      key={row.id}
                      className="border-b cursor-pointer hover:bg-black/5"
                      style={{ borderColor: "var(--rule)" }}
                      onClick={() => setSelectedArea(row.id)}
                    >
                      <td className="p-2 font-mono">{row.id}</td>
                      <td className="p-2">
                        <DangerChip danger_class={row.danger_class} />
                      </td>
                      <td className="p-2">{Math.round(row.danger_prob * 100)}%</td>
                      <td className="p-2">
                        {Math.round(row.root_soil_wetness * 100)}%
                        {row.root_soil_wetness < 0.25 ? " (drier, riskier)" : ""}
                      </td>
                      <td className="p-2 text-xs" style={{ color: "var(--ink-2)" }}>
                        {row.drivers.join(", ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {selectedArea && history && (
              <div className="mt-4 rounded border p-3" style={{ borderColor: "var(--rule)", background: "var(--surface)" }}>
                <h3 className="font-bold mb-2">{selectedArea} · last {history.dates.length} weeks</h3>
                <div className="text-xs grid grid-cols-1 gap-1" style={{ color: "var(--ink-2)" }}>
                  <div>Weekly fire detections: {history.fire.join(", ")}</div>
                  <div>CPC temp outlook probability: {history.t30.join(", ")}</div>
                  <div>CPC precip outlook probability: {history.p30.join(", ")}</div>
                  <div>Modeled danger probability over time: {history.dc.join(", ")}</div>
                </div>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
