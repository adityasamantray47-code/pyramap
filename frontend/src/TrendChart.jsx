// Small hand-rolled SVG line chart - no charting library needed for one series.
export default function TrendChart({ title, dates, values, color, formatValue, height = 110 }) {
  if (!values || values.length < 2) return null;

  const w = 600;
  const pad = 8;
  const max = Math.max(...values, 0.0001);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const stepX = (w - pad * 2) / (values.length - 1);

  const coords = values.map((v, i) => [pad + i * stepX, pad + (height - pad * 2) * (1 - (v - min) / range)]);
  const linePath = coords.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${coords.at(-1)[0]},${height - pad} L${coords[0][0]},${height - pad} Z`;

  return (
    <div style={{ marginTop: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span className="h-small" style={{ color: "var(--ink-3)" }}>
          {title}
        </span>
        <span className="num" style={{ fontSize: "0.8rem", color: "var(--ink-2)" }}>
          now: {formatValue(values.at(-1))}
        </span>
      </div>
      <svg viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" style={{ width: "100%", height, display: "block" }}>
        <path d={areaPath} fill={color} opacity="0.18" stroke="none" />
        <path d={linePath} fill="none" stroke={color} strokeWidth="2.5" />
      </svg>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--ink-3)" }}>
        <span>{dates[0]}</span>
        <span>{dates.at(-1)}</span>
      </div>
    </div>
  );
}
