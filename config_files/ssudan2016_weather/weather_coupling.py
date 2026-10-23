"""
Single-scale weather coupling for Flee (drop-in).

Adds precipitation- and river-discharge-driven changes to link distances INSIDE
the single-scale simulation, so the M0-M3 counterfactuals can be run WITHOUT the
multiscale macro-micro coupling.

Mechanism (reimplements the ICCS thresholds for the single-scale model):

  * Precipitation links -- any link with a column in precipitation.csv:
        tp <= X1      -> x1        (normal)
        X1 < tp <= X2 -> x2        (slowed)
        tp >  X2      -> xCLOSED   (impassable)
    X1, X2 are the q1/q2 percentiles (default 0.15 / 0.75). Preferred source is
    the 40-year monthly climatology 40yrs_tp.csv (percentiles of the monthly `tp`
    at the location nearest the link midpoint) -- this matches the ICCS method.
    If 40yrs_tp.csv is absent, X1/X2 fall back to percentiles of the link's own
    precipitation.csv series (stopgap).

  * Crossing link(s) -- endpoints in `crossings` (default Tonga-Malakal):
        discharge >  Q -> closed; 0.75*Q < discharge <= Q -> slowed (x2);
        else open. Q (closure threshold) default 8000; discharge from the
    nearest GloFAS grid cell to the link midpoint.

  * Every other link is unaffected.

Toggles: M0 both off | M1 precip only | M2 discharge only | M3 both on.
"""
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from flee import pflee

CLOSED = 10000.0
_daylog = []  # per-day diagnostics rows


class _Config:
    ready = False
    use_precipitation = True
    use_discharge = True
    q1 = 0.15
    q2 = 0.75
    Q = 8000.0
    n_days = 0
    precip = None
    coords = {}
    crossings = {frozenset(("Tonga", "Malakal"))}
    _grid = None
    _disc_by_cell = {}
    _tp40_grid = None      # np.array of (lat, lon) for 40-yr climatology locations
    _tp40_x = None         # np.array of (X1, X2) aligned with _tp40_grid
    tp40_used = False
    channel_cell = {}  # frozenset(pair) -> (lat,lon) channel-snapped discharge cell


def configure(input_csv_dir, start_date, n_days,
              use_precipitation=True, use_discharge=True,
              q1=0.15, q2=0.75, Q=8000.0, crossings=None):
    c = _Config
    c.use_precipitation = bool(use_precipitation)
    c.use_discharge = bool(use_discharge)
    c.q1, c.q2, c.Q = float(q1), float(q2), float(Q)
    c.n_days = int(n_days)
    if crossings:
        c.crossings = {frozenset(p) for p in crossings}
    else:
        rpath = os.path.join(input_csv_dir, "routes.csv")
        if os.path.exists(rpath):
            rdf = pd.read_csv(rpath)
            rdf.columns = [str(col).lstrip("#").strip().lower() for col in rdf.columns]
            if "link_type" in rdf.columns:
                n1, n2 = rdf.columns[0], rdf.columns[1]
                cr = {frozenset((str(r[n1]).strip(), str(r[n2]).strip()))
                      for _, r in rdf.iterrows()
                      if str(r.get("link_type", "")).strip().lower() == "crossing"}
                if cr:
                    c.crossings = cr

    c.precip = pd.read_csv(os.path.join(input_csv_dir, "precipitation.csv"))

    ldf = pd.read_csv(os.path.join(input_csv_dir, "locations.csv"))
    ldf.columns = [str(col).lstrip("#").strip().lower() for col in ldf.columns]
    name_c = ldf.columns[0]
    lat_c = "latitude" if "latitude" in ldf.columns else ldf.columns[3]
    lon_c = "longitude" if "longitude" in ldf.columns else ldf.columns[4]
    c.coords = {}
    for _, r in ldf.iterrows():
        try:
            c.coords[str(r[name_c]).strip()] = (float(r[lat_c]), float(r[lon_c]))
        except (ValueError, TypeError):
            pass

    # --- 40-year climatology thresholds (preferred) ---
    c._tp40_grid = None; c._tp40_x = None; c.tp40_used = False
    tp_path = os.path.join(input_csv_dir, "40yrs_tp.csv")
    if os.path.exists(tp_path):
        h = pd.read_csv(tp_path)
        h.columns = [str(col).strip().lower() for col in h.columns]
        latk = "latitude" if "latitude" in h.columns else "lat"
        lonk = "longitude" if "longitude" in h.columns else "lon"
        grid, xx = [], []
        for (la, lo), g in h.groupby([latk, lonk]):
            tp = pd.to_numeric(g["tp"], errors="coerce").dropna().to_numpy()
            if tp.size == 0:
                continue
            grid.append((float(la), float(lo)))
            xx.append((float(np.quantile(tp, c.q1)), float(np.quantile(tp, c.q2))))
        if grid:
            c._tp40_grid = np.array(grid)
            c._tp40_x = np.array(xx)
            c.tp40_used = True

    # --- river discharge grid, indexed by day ---
    ddf = pd.read_csv(os.path.join(input_csv_dir, "river_discharge.csv"))
    ddf.columns = [str(col).strip().lower() for col in ddf.columns]
    ddf = ddf.rename(columns={"latitude": "lat", "longitude": "lon"})
    base = datetime.strptime(str(start_date), "%Y-%m-%d")
    date_to_day = {(base + timedelta(days=i)).strftime("%Y-%m-%d"): i for i in range(c.n_days)}
    ddf = ddf[ddf["time"].isin(date_to_day)].copy()
    ddf["day"] = ddf["time"].map(date_to_day)
    c._disc_by_cell = {}
    for (la, lo), g in ddf.groupby(["lat", "lon"]):
        arr = np.zeros(c.n_days); arr[g["day"].to_numpy()] = g["dis24"].to_numpy()
        c._disc_by_cell[(float(la), float(lo))] = arr
    c._grid = np.array(list(c._disc_by_cell.keys())) if c._disc_by_cell else None

    # --- channel-snapped discharge cell per crossing (provenance mapping) ---
    c.channel_cell = {}
    mp = os.path.join(input_csv_dir, "crossing_discharge_mapping.csv")
    if os.path.exists(mp):
        md = pd.read_csv(mp); md.columns = [str(x).strip().lower() for x in md.columns]
        for _, r in md.iterrows():
            try:
                c.channel_cell[frozenset((str(r["name1"]).strip(), str(r["name2"]).strip()))] = \
                    (float(r["channel_cell_lat"]), float(r["channel_cell_lon"]))
            except Exception:
                pass

    # reset diagnostics state for this run
    Link_weather_coupling.instances = []
    _daylog.clear()
    c.ready = True


def _nearest(grid, lat, lon):
    d = (grid[:, 0] - lat) ** 2 + (grid[:, 1] - lon) ** 2
    return int(np.argmin(d))


def _nearest_cell(lat, lon):
    c = _Config
    if c._grid is None or len(c._grid) == 0:
        return None
    la, lo = c._grid[_nearest(c._grid, lat, lon)]
    return (float(la), float(lo))


def _thresholds_40yr(lat, lon):
    c = _Config
    i = _nearest(c._tp40_grid, lat, lon)
    return float(c._tp40_x[i, 0]), float(c._tp40_x[i, 1])


class Link_weather_coupling(pflee.Link):
    current_day = 0
    instances = []

    def __init__(self, startpoint, endpoint, distance, forced_redirection=False, attributes={}):
        super().__init__(startpoint, endpoint, distance, forced_redirection, attributes)
        self._base_distance = float(distance)
        self._pair = frozenset((startpoint.name, endpoint.name))
        self._is_crossing = self._pair in _Config.crossings
        self._mult = self._build_multipliers(startpoint.name, endpoint.name)
        Link_weather_coupling.instances.append(self)

    def _build_multipliers(self, a, b):
        c = _Config
        n = max(c.n_days, 1)
        la1, lo1 = c.coords.get(a, (0.0, 0.0))
        la2, lo2 = c.coords.get(b, (0.0, 0.0))
        mid_lat, mid_lon = (la1 + la2) / 2.0, (lo1 + lo2) / 2.0

        # crossing -> discharge (use channel-snapped cell if provided)
        if frozenset((a, b)) in c.crossings:
            if not c.use_discharge:
                return None
            ch = c.channel_cell.get(frozenset((a, b)))
            cell = _nearest_cell(ch[0], ch[1]) if ch is not None else _nearest_cell(mid_lat, mid_lon)
            dis = c._disc_by_cell.get(cell, np.zeros(n)) if cell else np.zeros(n)
            # Graduated rule (parallel to precipitation): Q2 = c.Q is the closure
            # threshold and Q1 = 0.75*Q2 the slowdown threshold.
            #   discharge <= Q1        -> open (x1)
            #   Q1 < discharge <= Q2   -> slowed (x2, effective distance doubled)
            #   discharge >  Q2        -> closed
            Q2 = c.Q
            Q1 = 0.75 * c.Q
            d = dis[:n]
            return np.where(d > Q2, CLOSED, np.where(d > Q1, 2.0, 1.0))

        # precipitation
        if not c.use_precipitation:
            return None
        col = None
        for cand in ("%s - %s" % (a, b), "%s - %s" % (b, a)):
            if cand in c.precip.columns:
                col = cand; break
        if col is None:
            return None
        full = pd.to_numeric(c.precip[col], errors="coerce").to_numpy()
        if c._tp40_grid is not None:                         # preferred: 40-yr climatology
            X1, X2 = _thresholds_40yr(mid_lat, mid_lon)
        else:                                                # fallback: own series
            valid = full[~np.isnan(full)]
            if valid.size == 0:
                return None
            X1 = float(np.quantile(valid, c.q1)); X2 = float(np.quantile(valid, c.q2))
        s = np.nan_to_num(full[:n], nan=0.0)
        if s.size < n:
            s = np.concatenate([s, np.zeros(n - s.size)])
        return np.where(s > X2, CLOSED, np.where(s > X1, 2.0, 1.0))

    def get_distance(self) -> float:
        if self._mult is None:
            return self._base_distance
        d = Link_weather_coupling.current_day
        if d < 0: d = 0
        elif d >= self._mult.size: d = self._mult.size - 1
        return self._base_distance * float(self._mult[d])


def sample_day(day):
    """Record per-day link-state diagnostics and crossing loads. Call once per
    timestep AFTER e.evolve()."""
    state = {}  # undirected pair -> [mult, load, is_crossing]
    for lk in Link_weather_coupling.instances:
        m = 1.0 if lk._mult is None else float(lk._mult[min(day, lk._mult.size - 1)])
        p = tuple(sorted(lk._pair))
        ld = int(getattr(lk, "numAgents", 0) or 0)
        if p in state:
            state[p][0] = max(state[p][0], m); state[p][1] += ld
            state[p][2] = state[p][2] or lk._is_crossing
        else:
            state[p] = [m, ld, lk._is_crossing]
    slowed = sum(1 for v in state.values() if (not v[2]) and 1.0 < v[0] < CLOSED)
    pclosed = sum(1 for v in state.values() if (not v[2]) and v[0] >= CLOSED)
    dclosed = sum(1 for v in state.values() if v[2] and v[0] >= CLOSED)
    row = {"day": day, "precip_slowed_links": slowed,
           "precip_closed_links": pclosed, "discharge_closed_crossings": dclosed}
    for p, v in state.items():
        if v[2]:
            nm = "%s-%s" % p
            row["load[%s]" % nm] = v[1]; row["closed[%s]" % nm] = int(v[0] >= CLOSED)
    _daylog.append(row)


def write_diagnostics(prefix):
    """Write <prefix>_daily.csv (per-day link state + crossing loads) and
    <prefix>_crossings.csv (per-crossing closure days + cumulative usage)."""
    if _daylog:
        pd.DataFrame(_daylog).to_csv(prefix + "_daily.csv", index=False)
    byp = {}
    for lk in Link_weather_coupling.instances:
        if not lk._is_crossing:
            continue
        p = tuple(sorted(lk._pair))
        e = byp.setdefault(p, {"cum": 0, "load": 0, "mult": None})
        e["cum"] += int(getattr(lk, "cumNumAgents", 0) or 0)
        e["load"] += int(getattr(lk, "numAgents", 0) or 0)
        if lk._mult is not None:
            e["mult"] = lk._mult
    rows = []
    for p, e in byp.items():
        mult = e["mult"]; cd = 0
        if mult is not None:
            cd = int(sum(1 for d in range(max(_Config.n_days, 1))
                         if mult[min(d, mult.size - 1)] >= CLOSED))
        rows.append({"crossing": "%s-%s" % p, "closure_days": cd,
                     "cum_agents_used": e["cum"], "final_load": e["load"]})
    if rows:
        pd.DataFrame(rows).to_csv(prefix + "_crossings.csv", index=False)
