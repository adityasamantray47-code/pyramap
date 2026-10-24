const COLORS = {
  Low: "var(--fd-low)",
  Moderate: "var(--fd-moderate)",
  High: "var(--fd-high)",
  "Very High": "var(--fd-veryhigh)",
  Extreme: "var(--fd-extreme)",
};

export default function DangerChip({ danger_class }) {
  const bg = COLORS[danger_class] || "var(--ink-2)";
  const darkText = danger_class === "High";
  return (
    <span
      className="inline-flex items-center rounded px-2 py-0.5 text-xs font-bold uppercase tracking-wide"
      style={{ background: bg, color: darkText ? "#3b2f00" : "#fff" }}
    >
      {danger_class}
    </span>
  );
}
