const BASE = "/api";

async function get(path, params = {}) {
  const url = new URL(BASE + path, window.location.origin);
  Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export const api = {
  states: () => get("/states"),
  dataStatus: () => get("/data-status"),
  outlook: (state) => get("/30-day-outlook", { state }),
  metrics: (state) => get("/model-metrics", { state }),
  areaHistory: (areaId, days = 120) => get("/area-history", { area_id: areaId, days }),
};
