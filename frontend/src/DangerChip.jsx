import { DANGER_DARK_TEXT, DANGER_HEX } from "./dangerColors";

export default function DangerChip({ danger_class }) {
  const bg = DANGER_HEX[danger_class] || "#78787f";
  const isDark = DANGER_DARK_TEXT.has(danger_class);
  return (
    <span className={`chip${isDark ? " lo" : ""}`} style={{ background: bg }}>
      {danger_class}
    </span>
  );
}
