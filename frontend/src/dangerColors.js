// Hex values matching the --fd-* custom properties in index.css. Duplicated
// as plain hex (not var()) because Leaflet's SVG renderer needs real colors.
export const DANGER_HEX = {
  Low: "#90be6d",
  Moderate: "#f9c74f",
  High: "#ff6b35",
  "Very High": "#f3722c",
  Extreme: "#f94144",
};

export const DANGER_DARK_TEXT = new Set(["Low", "Moderate"]);
