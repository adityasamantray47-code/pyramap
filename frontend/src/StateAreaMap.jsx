import { MapContainer, TileLayer, Rectangle, Tooltip, useMap } from "react-leaflet";
import { useEffect } from "react";
import { DANGER_HEX } from "./dangerColors";

function FitToAreas({ areas }) {
  const map = useMap();
  useEffect(() => {
    if (!areas.length) return;
    const lats = areas.flatMap((a) => [a.bbox[0], a.bbox[2]]);
    const lons = areas.flatMap((a) => [a.bbox[1], a.bbox[3]]);
    map.fitBounds(
      [
        [Math.min(...lats), Math.min(...lons)],
        [Math.max(...lats), Math.max(...lons)],
      ],
      { padding: [16, 16] }
    );
  }, [areas, map]);
  return null;
}

// Shows each forecast area as the actual rectangle of land it represents,
// colored by danger level and clickable - so "TX-6" is a place you can see
// and point at, not just a row in a table.
export default function StateAreaMap({ areas, selectedArea, onSelectArea }) {
  return (
    <MapContainer center={[39.5, -98.35]} zoom={5} style={{ height: 380, width: "100%" }} scrollWheelZoom={false}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
      <FitToAreas areas={areas} />
      {areas.map((area) => {
        const [latMin, lonMin, latMax, lonMax] = area.bbox;
        const isSelected = area.id === selectedArea;
        return (
          <Rectangle
            key={area.id}
            bounds={[
              [latMin, lonMin],
              [latMax, lonMax],
            ]}
            pathOptions={{
              color: isSelected ? "#ffffff" : DANGER_HEX[area.danger_class] || "#78787f",
              weight: isSelected ? 3 : 1,
              fillColor: DANGER_HEX[area.danger_class] || "#78787f",
              fillOpacity: isSelected ? 0.75 : 0.5,
            }}
            eventHandlers={{ click: () => onSelectArea(area.id) }}
          >
            <Tooltip sticky>
              <b>{area.region}</b> ({area.id}) · {area.danger_class} · {Math.round(area.danger_prob * 100)}%
            </Tooltip>
          </Rectangle>
        );
      })}
    </MapContainer>
  );
}
