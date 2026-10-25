const COLORS = {
  Low: "var(--fd-low)",
  Moderate: "var(--fd-moderate)",
  High: "var(--fd-high)",
  "Very High": "var(--fd-veryhigh)",
  Extreme: "var(--fd-extreme)",
};

const DARK_TEXT = { Low: "lo", Moderate: "mo" };

export default function DangerChip({ danger_class }) {
  const bg = COLORS[danger_class] || "var(--ink-3)";
  const cls = DARK_TEXT[danger_class] || "";
  return (
    <span className={`chip${cls ? " " + cls : ""}`} style={{ background: bg }}>
      {danger_class}
    </span>
  );
}
