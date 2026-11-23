import { useState, useEffect, useCallback, useRef } from "react";

// ─────────────────────────────────────────────────────────────────
// API CLIENT — mirrors the FastAPI backend
// In production, set VITE_API_URL=http://your-backend:8000
// ─────────────────────────────────────────────────────────────────
const API_BASE = (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_URL) || "http://localhost:8000";
const API = `${API_BASE}/api/v1`;

const apiClient = {
  async predict(features) {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(features),
    });
    if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
    return res.json();
  },
  async health() {
    const res = await fetch(`${API}/health`);
    return res.json();
  },
  async readiness() {
    const res = await fetch(`${API}/health/ready`);
    return res.json();
  },
  async getMetrics() {
    const res = await fetch(`${API}/metrics`);
    return res.json();
  },
  async listModels() {
    const res = await fetch(`${API}/models`);
    return res.json();
  },
  async getActiveModel() {
    const res = await fetch(`${API}/models/active`);
    return res.json();
  },
  async promoteModel(version) {
    const res = await fetch(`${API}/models/promote/${version}`, { method: "POST" });
    return res.json();
  },
  async triggerTraining(config) {
    const res = await fetch(`${API}/models/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    return res.json();
  },
  async datasetStats() {
    const res = await fetch(`${API}/dataset/stats`);
    return res.json();
  },
  async datasetTracks(genre = null, limit = 20) {
    const url = genre
      ? `${API}/dataset/tracks?genre=${genre}&limit=${limit}`
      : `${API}/dataset/tracks?limit=${limit}`;
    const res = await fetch(url);
    return res.json();
  },
};

// ─────────────────────────────────────────────────────────────────
// CONSTANTS & CONFIG
// ─────────────────────────────────────────────────────────────────
const GENRE_COLORS = {
  Pop: "#FF6B9D", "Hip-Hop": "#FFD93D", Electronic: "#4ECDC4",
  Rock: "#FF6348", "R&B": "#A29BFE", Country: "#6BCB77",
};
const GENRES = Object.keys(GENRE_COLORS);

const FEATURE_CONFIG = {
  danceability: { label: "Danceability", emoji: "🕺", min: 0, max: 1, step: 0.01, desc: "Rhythmic suitability for dancing" },
  energy:       { label: "Energy",       emoji: "⚡", min: 0, max: 1, step: 0.01, desc: "Perceptual intensity & activity" },
  valence:      { label: "Valence",      emoji: "😊", min: 0, max: 1, step: 0.01, desc: "Musical positiveness / mood" },
  tempo:        { label: "Tempo (BPM)",  emoji: "🥁", min: 60, max: 200, step: 1,    desc: "Beats per minute" },
  acousticness: { label: "Acousticness", emoji: "🎸", min: 0, max: 1, step: 0.01, desc: "Acoustic vs electronic content" },
  speechiness:  { label: "Speechiness",  emoji: "🎤", min: 0, max: 1, step: 0.01, desc: "Spoken word presence" },
};

const PRESET_TRACKS = [
  { name: "Levels (Avicii)", features: { danceability: 0.67, energy: 0.95, valence: 0.89, tempo: 126, acousticness: 0.01, speechiness: 0.03 } },
  { name: "HUMBLE. (Kendrick)", features: { danceability: 0.91, energy: 0.62, valence: 0.42, tempo: 150, acousticness: 0.05, speechiness: 0.22 } },
  { name: "Anti-Hero (T.Swift)", features: { danceability: 0.64, energy: 0.64, valence: 0.42, tempo: 97, acousticness: 0.07, speechiness: 0.05 } },
  { name: "Mr. Brightside", features: { danceability: 0.34, energy: 0.92, valence: 0.25, tempo: 148, acousticness: 0.00, speechiness: 0.05 } },
  { name: "All of Me (Legend)", features: { danceability: 0.39, energy: 0.18, valence: 0.36, tempo: 120, acousticness: 0.86, speechiness: 0.03 } },
  { name: "Jolene (Parton)", features: { danceability: 0.47, energy: 0.44, valence: 0.48, tempo: 112, acousticness: 0.62, speechiness: 0.03 } },
];

// ─────────────────────────────────────────────────────────────────
// UTILITY COMPONENTS
// ─────────────────────────────────────────────────────────────────
function StatusDot({ status }) {
  const color = status === "ready" ? "#6BCB77" : status === "degraded" ? "#FFD93D" : "#FF6348";
  return (
    <span style={{
      display: "inline-block", width: 8, height: 8, borderRadius: "50%",
      background: color, boxShadow: `0 0 6px ${color}`, marginRight: 6,
    }} />
  );
}

function Card({ children, style = {}, glow }) {
  return (
    <div style={{
      background: "#0a0a1a", borderRadius: 16, padding: 20,
      border: `1px solid ${glow ? glow + "44" : "#1a1a2e"}`,
      boxShadow: glow ? `0 0 20px ${glow}22` : "none",
      ...style,
    }}>
      {children}
    </div>
  );
}

function Pill({ label, color }) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 10px", borderRadius: 20,
      background: color + "22", color, fontSize: 11, fontFamily: "monospace",
      border: `1px solid ${color}44`,
    }}>
      {label}
    </span>
  );
}

function ProgressBar({ value, max = 1, color, animated }) {
  const pct = Math.round((value / max) * 100);
  return (
    <div style={{ position: "relative", background: "#111", borderRadius: 4, height: 8, overflow: "hidden" }}>
      <div style={{
        position: "absolute", left: 0, top: 0, bottom: 0,
        width: `${pct}%`, background: color, borderRadius: 4,
        transition: animated ? "width 0.7s cubic-bezier(0.34,1.56,0.64,1)" : "none",
      }} />
    </div>
  );
}

function MetricCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: "#0d0d20", borderRadius: 12, padding: "14px 16px",
      border: `1px solid ${color}33`, textAlign: "center",
    }}>
      <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: "'Space Mono', monospace" }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: "#aaa", marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: "#444", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// RADAR CHART
// ─────────────────────────────────────────────────────────────────
function RadarChart({ features, color, size = 180 }) {
  const cx = size / 2, cy = size / 2, r = size * 0.36;
  const feats = ["danceability", "energy", "valence", "acousticness", "speechiness"];
  const n = feats.length;

  const normalize = (f, v) => {
    const cfg = FEATURE_CONFIG[f];
    return (v - cfg.min) / (cfg.max - cfg.min);
  };

  const pts = feats.map((f, i) => {
    const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
    const norm = normalize(f, features[f] || 0);
    return { x: cx + r * norm * Math.cos(angle), y: cy + r * norm * Math.sin(angle), f, norm };
  });

  const outerRing = feats.map((_, i) => {
    const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
  });

  const gridRings = [0.25, 0.5, 0.75, 1].map(s =>
    feats.map((_, i) => {
      const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
      return `${cx + r * s * Math.cos(angle)},${cy + r * s * Math.sin(angle)}`;
    }).join(" ")
  );

  return (
    <svg viewBox={`0 0 ${size} ${size}`} style={{ width: "100%", maxWidth: size }}>
      {gridRings.map((pts, i) => (
        <polygon key={i} points={pts} fill="none" stroke="#ffffff0d" strokeWidth="1" />
      ))}
      {outerRing.map((p, i) => (
        <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="#ffffff0d" strokeWidth="1" />
      ))}
      <polygon
        points={pts.map(p => `${p.x},${p.y}`).join(" ")}
        fill={color + "33"} stroke={color} strokeWidth="2"
        style={{ transition: "all 0.4s ease" }}
      />
      {pts.map((p, i) => {
        const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
        const lx = cx + (r + 16) * Math.cos(angle);
        const ly = cy + (r + 16) * Math.sin(angle);
        return (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r="3.5" fill={color} style={{ transition: "all 0.4s ease" }} />
            <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle"
              style={{ fontSize: 9, fill: "#666", fontFamily: "monospace" }}>
              {FEATURE_CONFIG[p.f].emoji}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────
// CONFUSION MATRIX
// ─────────────────────────────────────────────────────────────────
function ConfusionMatrix({ matrix, classes }) {
  if (!matrix?.length) return <div style={{ color: "#444", fontSize: 12 }}>No confusion matrix data</div>;
  const max = Math.max(...matrix.flat());
  const cellSize = Math.min(48, Math.floor(320 / classes.length));

  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ display: "inline-block" }}>
        <div style={{ display: "flex", marginBottom: 2, marginLeft: cellSize + 8 }}>
          {classes.map(c => (
            <div key={c} style={{
              width: cellSize, textAlign: "center", fontSize: 9,
              color: GENRE_COLORS[c], fontFamily: "monospace",
              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            }}>
              {c.slice(0, 4)}
            </div>
          ))}
        </div>
        {matrix.map((row, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", marginBottom: 2 }}>
            <div style={{
              width: cellSize, fontSize: 9, color: GENRE_COLORS[classes[i]],
              fontFamily: "monospace", marginRight: 8, textAlign: "right",
              whiteSpace: "nowrap", overflow: "hidden",
            }}>
              {classes[i].slice(0, 4)}
            </div>
            {row.map((val, j) => {
              const intensity = max > 0 ? val / max : 0;
              const isCorrect = i === j;
              const bg = isCorrect
                ? `rgba(78, 205, 196, ${0.1 + intensity * 0.8})`
                : `rgba(255, 107, 107, ${intensity * 0.7})`;
              return (
                <div key={j} title={`${classes[i]} → ${classes[j]}: ${(val * 100).toFixed(0)}%`}
                  style={{
                    width: cellSize, height: cellSize, display: "flex",
                    alignItems: "center", justifyContent: "center",
                    background: bg, fontSize: 9, color: "#ddd",
                    fontFamily: "monospace", border: "1px solid #0d0d20",
                    cursor: "default",
                  }}>
                  {(val * 100).toFixed(0)}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <p style={{ fontSize: 9, color: "#333", marginTop: 8, fontFamily: "monospace" }}>
        Rows = True Label · Cols = Predicted · Green = Correct · Red = Error
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// FEATURE SLIDER
// ─────────────────────────────────────────────────────────────────
function FeatureSlider({ name, value, onChange }) {
  const cfg = FEATURE_CONFIG[name];
  const norm = (value - cfg.min) / (cfg.max - cfg.min);
  const display = name === "tempo" ? Math.round(value) : value.toFixed(2);

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 14 }}>{cfg.emoji}</span>
          <span style={{ fontSize: 12, color: "#ccc", fontWeight: 600 }}>{cfg.label}</span>
        </div>
        <span style={{
          fontFamily: "monospace", fontSize: 12,
          background: `rgba(78,205,196,${0.1 + norm * 0.3})`,
          color: "#4ECDC4", padding: "1px 8px", borderRadius: 4,
          transition: "background 0.2s",
        }}>
          {display}
        </span>
      </div>
      <input
        type="range" min={cfg.min} max={cfg.max} step={cfg.step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: "#4ECDC4", cursor: "pointer", height: 4 }}
      />
      <p style={{ fontSize: 10, color: "#444", marginTop: 2 }}>{cfg.desc}</p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// API DEMO PANEL (shows the actual HTTP request/response)
// ─────────────────────────────────────────────────────────────────
function ApiPanel({ features, result, loading }) {
  const reqBody = JSON.stringify(features, null, 2);
  const resBody = result ? JSON.stringify({
    predicted_genre: result.predicted_genre,
    confidence: result.confidence,
    all_genres: result.all_genres?.slice(0, 3),
    model_version: result.model_version,
  }, null, 2) : null;

  return (
    <div style={{ fontFamily: "monospace", fontSize: 11 }}>
      <div style={{ marginBottom: 12 }}>
        <div style={{ color: "#4ECDC4", marginBottom: 6 }}>
          POST {API}/predict
        </div>
        <div style={{
          background: "#060612", borderRadius: 8, padding: 12,
          border: "1px solid #1a1a2e", color: "#888", overflowX: "auto",
        }}>
          <pre style={{ margin: 0 }}>{reqBody}</pre>
        </div>
      </div>

      {loading && (
        <div style={{ color: "#FFD93D", padding: 12, textAlign: "center" }}>
          ⟳ Waiting for response...
        </div>
      )}

      {result && (
        <div>
          <div style={{ color: "#6BCB77", marginBottom: 6 }}>
            HTTP 200 OK — {result.model_version}
          </div>
          <div style={{
            background: "#060612", borderRadius: 8, padding: 12,
            border: "1px solid #1a1a2e", color: "#888", overflowX: "auto",
          }}>
            <pre style={{ margin: 0 }}>{resBody}</pre>
          </div>
        </div>
      )}

      {!result && !loading && (
        <div style={{ color: "#333", padding: 12, textAlign: "center" }}>
          Submit a prediction to see the API response
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────────
export default function SoundMLDashboard() {
  const [tab, setTab] = useState("classify");

  // API state
  const [apiStatus, setApiStatus] = useState({ status: "checking" });
  const [metrics, setMetrics] = useState(null);
  const [modelList, setModelList] = useState(null);
  const [datasetStats, setDatasetStats] = useState(null);
  const [datasetTracks, setDatasetTracks] = useState(null);
  const [filterGenre, setFilterGenre] = useState(null);

  // Classifier state
  const [features, setFeatures] = useState({
    danceability: 0.65, energy: 0.78, valence: 0.60,
    tempo: 128, acousticness: 0.05, speechiness: 0.06,
  });
  const [result, setResult] = useState(null);
  const [predLoading, setPredLoading] = useState(false);
  const [predError, setPredError] = useState(null);
  const [autoPredict, setAutoPredict] = useState(false);

  // Model management
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainMsg, setTrainMsg] = useState(null);

  const debounceRef = useRef(null);

  // ── Fetch API status on mount ────────────────────────────────────────────
  useEffect(() => {
    const checkApi = async () => {
      try {
        const [health, ready] = await Promise.all([apiClient.health(), apiClient.readiness()]);
        setApiStatus({ ...health, ready: ready.status });
      } catch (e) {
        setApiStatus({ status: "offline", error: e.message });
      }
    };
    checkApi();
    const interval = setInterval(checkApi, 15000);
    return () => clearInterval(interval);
  }, []);

  // ── Fetch metrics and models when tab changes ────────────────────────────
  useEffect(() => {
    if (tab === "metrics" && !metrics) {
      apiClient.getMetrics().then(setMetrics).catch(() => setMetrics({ error: "Failed to load" }));
    }
    if (tab === "registry" && !modelList) {
      apiClient.listModels().then(setModelList).catch(() => setModelList({ error: "Failed to load" }));
    }
    if (tab === "dataset" && !datasetStats) {
      Promise.all([apiClient.datasetStats(), apiClient.datasetTracks(null, 30)])
        .then(([stats, tracks]) => { setDatasetStats(stats); setDatasetTracks(tracks); })
        .catch(() => setDatasetStats({ error: "Failed to load" }));
    }
  }, [tab]);

  // ── Auto-predict on feature change ──────────────────────────────────────
  useEffect(() => {
    if (!autoPredict) return;
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => { handlePredict(); }, 300);
  }, [features, autoPredict]);

  // ── Filter dataset tracks ────────────────────────────────────────────────
  useEffect(() => {
    if (tab !== "dataset") return;
    apiClient.datasetTracks(filterGenre, 40)
      .then(setDatasetTracks)
      .catch(() => {});
  }, [filterGenre]);

  const handlePredict = useCallback(async () => {
    setPredLoading(true);
    setPredError(null);
    try {
      const res = await apiClient.predict(features);
      setResult(res);
    } catch (e) {
      setPredError(e.message);
      setResult(null);
    } finally {
      setPredLoading(false);
    }
  }, [features]);

  const handlePromote = async (version) => {
    if (!confirm(`Promote ${version} to production?`)) return;
    await apiClient.promoteModel(version);
    const updated = await apiClient.listModels();
    setModelList(updated);
    setMetrics(null); // Refresh metrics
  };

  const handleTrain = async () => {
    setTrainLoading(true);
    setTrainMsg(null);
    try {
      const res = await apiClient.triggerTraining({ models: null, tune: false, synthetic_per_genre: 50, auto_promote: true });
      setTrainMsg(res.message);
    } catch (e) {
      setTrainMsg(`Error: ${e.message}`);
    } finally {
      setTrainLoading(false);
    }
  };

  const loadPreset = (preset) => {
    setFeatures(preset.features);
    setResult(null);
  };

  const isOffline = apiStatus.status === "offline";
  const genreColor = result ? (GENRE_COLORS[result.predicted_genre] || "#4ECDC4") : "#4ECDC4";

  // ─────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────
  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(160deg, #04040f 0%, #0a0a20 60%, #060610 100%)",
      fontFamily: "'DM Sans', system-ui, sans-serif",
      color: "#e0e0e0",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
        input[type=range] { -webkit-appearance: none; height: 4px; border-radius: 2px; background: #1a1a2e; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #4ECDC4; cursor: pointer; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .fade-in { animation: fadeIn 0.35s ease forwards; }
        .tab-btn { padding: 8px 18px; border-radius: 8px; border: 1px solid #1a1a2e; cursor: pointer; font-size: 12px; font-weight: 600; font-family: 'Space Mono', monospace; transition: all 0.15s; background: transparent; color: #666; }
        .tab-btn:hover { border-color: #4ECDC4; color: #4ECDC4; }
        .tab-btn.active { background: #4ECDC4; color: #04040f; border-color: #4ECDC4; }
        .preset-chip { padding: 5px 12px; border-radius: 20px; border: 1px solid #1a1a2e; background: #0a0a1a; font-size: 11px; cursor: pointer; color: #666; transition: all 0.15s; white-space: nowrap; }
        .preset-chip:hover { border-color: #4ECDC4; color: #4ECDC4; transform: translateY(-1px); }
        .action-btn { padding: 10px 22px; border-radius: 10px; border: none; cursor: pointer; font-size: 13px; font-weight: 700; font-family: 'DM Sans', sans-serif; transition: all 0.15s; }
        .action-btn:hover { filter: brightness(1.15); transform: translateY(-1px); }
        .action-btn:active { transform: translateY(0); }
        tr:hover td { background: #0d0d22 !important; }
      `}</style>

      {/* ── HEADER ── */}
      <div style={{
        background: "linear-gradient(90deg, #060618, #0d0d28)",
        borderBottom: "1px solid #1a1a30",
        padding: "0 24px", height: 62,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        position: "sticky", top: 0, zIndex: 100, backdropFilter: "blur(12px)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "linear-gradient(135deg, #4ECDC4 0%, #A29BFE 100%)",
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
          }}>🎵</div>
          <div>
            <div style={{ fontFamily: "'Space Mono',monospace", fontWeight: 700, fontSize: 18, letterSpacing: -0.5 }}>
              Sound<span style={{ color: "#4ECDC4" }}>ML</span>
              <span style={{ fontSize: 10, color: "#555", marginLeft: 8, fontWeight: 400 }}>v1.0</span>
            </div>
            <div style={{ fontSize: 9, color: "#444", fontFamily: "monospace" }}>
              Production ML System · Spotify Audio Classifier
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 6 }}>
          {[
            ["classify", "🎯 Classify"],
            ["metrics", "📊 Metrics"],
            ["registry", "🗄️ Registry"],
            ["dataset", "📦 Dataset"],
            ["api", "📡 API"],
          ].map(([key, label]) => (
            <button key={key} className={`tab-btn ${tab === key ? "active" : ""}`} onClick={() => setTab(key)}>
              {label}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, fontFamily: "monospace" }}>
          <StatusDot status={isOffline ? "offline" : apiStatus.ready === "ready" ? "ready" : "degraded"} />
          <span style={{ color: "#555" }}>
            {isOffline ? "API Offline" : apiStatus.ready === "ready" ? "API Ready" : "No Model"}
          </span>
        </div>
      </div>

      {/* ── OFFLINE BANNER ── */}
      {isOffline && (
        <div style={{
          background: "#FF634822", borderBottom: "1px solid #FF634844",
          padding: "10px 24px", fontSize: 12, color: "#FF6348",
          display: "flex", alignItems: "center", gap: 8,
        }}>
          ⚠️ <strong>API Offline</strong> — The frontend is fully functional for exploration.
          Start the backend: <code style={{ fontFamily: "monospace", background: "#0a0a1a", padding: "2px 6px", borderRadius: 4 }}>
            uvicorn backend.api.main:app --reload
          </code>
        </div>
      )}

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "24px 24px" }}>

        {/* ════════════════════════════════════════════════════════
            TAB: CLASSIFY
        ════════════════════════════════════════════════════════ */}
        {tab === "classify" && (
          <div className="fade-in">
            <div style={{ marginBottom: 16 }}>
              <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>🎯 Genre Classifier</h2>
              <p style={{ color: "#555", fontSize: 13 }}>
                Adjust audio features and classify — or load a preset track. Results come from the live FastAPI backend.
              </p>
            </div>

            {/* Presets */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20 }}>
              <span style={{ fontSize: 11, color: "#444", fontFamily: "monospace", alignSelf: "center" }}>LOAD PRESET →</span>
              {PRESET_TRACKS.map(p => (
                <button key={p.name} className="preset-chip" onClick={() => loadPreset(p)}>{p.name}</button>
              ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 20 }}>

              {/* Left: sliders */}
              <Card>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
                  <h3 style={{ fontSize: 14, fontWeight: 700 }}>🎛️ Audio Features</h3>
                  <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#555", cursor: "pointer" }}>
                    <input type="checkbox" checked={autoPredict} onChange={e => setAutoPredict(e.target.checked)}
                      style={{ accentColor: "#4ECDC4" }} />
                    Auto-predict
                  </label>
                </div>

                {Object.keys(FEATURE_CONFIG).map(name => (
                  <FeatureSlider key={name} name={name} value={features[name]}
                    onChange={v => setFeatures(prev => ({ ...prev, [name]: v }))} />
                ))}

                <button
                  className="action-btn"
                  onClick={handlePredict}
                  disabled={predLoading}
                  style={{
                    width: "100%", marginTop: 8,
                    background: predLoading ? "#1a1a2e" : "linear-gradient(90deg, #4ECDC4, #A29BFE)",
                    color: predLoading ? "#555" : "#04040f",
                  }}>
                  {predLoading ? "⟳ Classifying..." : "🔍 Classify Genre"}
                </button>

                {predError && (
                  <div style={{ marginTop: 10, padding: 10, background: "#FF634822", borderRadius: 8, fontSize: 12, color: "#FF6348" }}>
                    {isOffline ? "⚠️ API offline — start backend to get live predictions" : `Error: ${predError}`}
                  </div>
                )}
              </Card>

              {/* Right: result */}
              <div>
                {result ? (
                  <div className="fade-in">
                    <Card glow={genreColor} style={{ marginBottom: 16 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                        <div>
                          <p style={{ fontSize: 10, color: "#555", fontFamily: "monospace", marginBottom: 4 }}>
                            PREDICTED GENRE · {result.model_version}
                          </p>
                          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                            <div style={{ width: 10, height: 10, borderRadius: "50%", background: genreColor, boxShadow: `0 0 10px ${genreColor}` }} />
                            <span style={{ fontSize: 30, fontWeight: 800, color: genreColor, fontFamily: "'Space Mono',monospace" }}>
                              {result.predicted_genre}
                            </span>
                          </div>
                          <p style={{ fontSize: 12, color: "#555", marginBottom: 8 }}>
                            {result.confidence_pct}% confidence · {result.model_key?.toUpperCase()}
                          </p>
                          <p style={{ fontSize: 12, color: "#888", fontStyle: "italic" }}>{result.explanation}</p>
                        </div>
                        <RadarChart features={features} color={genreColor} size={160} />
                      </div>

                      {/* Genre confidence bars */}
                      <div style={{ marginTop: 16 }}>
                        <p style={{ fontSize: 10, color: "#444", fontFamily: "monospace", marginBottom: 10 }}>ALL GENRE PROBABILITIES</p>
                        {result.all_genres?.map(g => (
                          <div key={g.genre} style={{ marginBottom: 8 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                              <span style={{ fontSize: 12, color: g.genre === result.predicted_genre ? GENRE_COLORS[g.genre] : "#666", fontWeight: g.genre === result.predicted_genre ? 700 : 400 }}>
                                {g.genre}
                              </span>
                              <span style={{ fontSize: 11, fontFamily: "monospace", color: "#555" }}>{g.percentage}%</span>
                            </div>
                            <ProgressBar value={g.probability} color={GENRE_COLORS[g.genre]} animated />
                          </div>
                        ))}
                      </div>
                    </Card>
                  </div>
                ) : (
                  <Card style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 300 }}>
                    <div style={{ fontSize: 40, marginBottom: 12 }}>🎵</div>
                    <p style={{ color: "#444", fontSize: 14, textAlign: "center" }}>
                      {predLoading ? "Classifying..." : "Hit Classify Genre to get a prediction"}
                    </p>
                    <RadarChart features={features} color="#333" size={140} />
                  </Card>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ════════════════════════════════════════════════════════
            TAB: METRICS
        ════════════════════════════════════════════════════════ */}
        {tab === "metrics" && (
          <div className="fade-in">
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>📊 Model Performance Metrics</h2>
            <p style={{ color: "#555", fontSize: 13, marginBottom: 20 }}>Live metrics from the production model via GET /api/v1/metrics</p>

            {!metrics && <div style={{ color: "#555" }}>Loading metrics...</div>}
            {metrics?.error && (
              <Card>
                <p style={{ color: "#FF6348" }}>⚠️ {metrics.error}</p>
                <p style={{ color: "#555", fontSize: 12, marginTop: 8 }}>Start the backend and run training first.</p>
              </Card>
            )}
            {metrics && !metrics.error && (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12, marginBottom: 20 }}>
                  <MetricCard label="Accuracy" value={`${((metrics.performance?.accuracy || 0) * 100).toFixed(1)}%`} color="#4ECDC4" sub="Overall" />
                  <MetricCard label="Macro F1" value={((metrics.performance?.macro_f1 || 0)).toFixed(3)} color="#A29BFE" />
                  <MetricCard label="Precision" value={((metrics.performance?.macro_precision || 0)).toFixed(3)} color="#FF6B9D" />
                  <MetricCard label="Recall" value={((metrics.performance?.macro_recall || 0)).toFixed(3)} color="#FFD93D" />
                  <MetricCard label="Top-2 Acc" value={metrics.performance?.top2_accuracy ? `${(metrics.performance.top2_accuracy * 100).toFixed(1)}%` : "N/A"} color="#6BCB77" />
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                  {/* Per-class */}
                  <Card>
                    <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 16 }}>Per-Genre Performance</h3>
                    {Object.entries(metrics.per_class || {}).map(([genre, m]) => (
                      <div key={genre} style={{ marginBottom: 14 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                          <span style={{ color: GENRE_COLORS[genre], fontSize: 13, fontWeight: 600 }}>{genre}</span>
                          <div style={{ display: "flex", gap: 12, fontSize: 10, fontFamily: "monospace", color: "#555" }}>
                            <span>P: {m.precision}</span>
                            <span>R: {m.recall}</span>
                            <span style={{ color: GENRE_COLORS[genre] }}>F1: {m.f1}</span>
                          </div>
                        </div>
                        <ProgressBar value={m.f1} color={GENRE_COLORS[genre]} animated />
                      </div>
                    ))}
                  </Card>

                  {/* Confusion matrix */}
                  <Card>
                    <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 16 }}>Confusion Matrix (normalised)</h3>
                    <ConfusionMatrix
                      matrix={metrics.confusion_matrix || []}
                      classes={metrics.classes || GENRES}
                    />
                  </Card>

                  {/* Training info */}
                  <Card>
                    <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Training Details</h3>
                    {[
                      ["Model", metrics.model_key],
                      ["Version", metrics.production_version],
                      ["Train samples", metrics.training_info?.n_train],
                      ["Test samples", metrics.training_info?.n_test],
                      ["CV Accuracy", metrics.training_info?.best_cv_accuracy?.toFixed(4)],
                      ["Train time", `${metrics.training_info?.train_time_seconds}s`],
                    ].map(([k, v]) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid #111", fontSize: 12 }}>
                        <span style={{ color: "#666" }}>{k}</span>
                        <span style={{ fontFamily: "monospace", color: "#aaa" }}>{v || "—"}</span>
                      </div>
                    ))}
                    {Object.entries(metrics.training_info?.best_params || {}).length > 0 && (
                      <div style={{ marginTop: 12 }}>
                        <p style={{ fontSize: 10, color: "#444", fontFamily: "monospace", marginBottom: 8 }}>BEST HYPERPARAMETERS</p>
                        {Object.entries(metrics.training_info.best_params).map(([k, v]) => (
                          <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, padding: "4px 0" }}>
                            <span style={{ color: "#555", fontFamily: "monospace" }}>{k}</span>
                            <Pill label={String(v)} color="#4ECDC4" />
                          </div>
                        ))}
                      </div>
                    )}
                  </Card>

                  {/* Educational explanation */}
                  <Card>
                    <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>📚 Reading These Metrics</h3>
                    {[
                      { name: "Accuracy", desc: "% of all predictions that are correct. Misleading if classes are imbalanced. We have 6 balanced classes, so it's meaningful here." },
                      { name: "Macro F1", desc: "Average F1 score across all genres, treating each genre equally regardless of sample count. Best for balanced evaluation." },
                      { name: "Precision", desc: "Of all tracks predicted as Genre X, what fraction actually is X? High precision = low false alarm rate." },
                      { name: "Recall", desc: "Of all actual Genre X tracks, what fraction did we correctly identify? High recall = few missed instances." },
                      { name: "Confusion Matrix", desc: "Each cell [i,j] shows what fraction of true genre i was predicted as j. Diagonal = correct. Off-diagonal = errors." },
                    ].map(m => (
                      <div key={m.name} style={{ marginBottom: 12 }}>
                        <span style={{ fontSize: 12, color: "#4ECDC4", fontWeight: 600 }}>{m.name}: </span>
                        <span style={{ fontSize: 12, color: "#666" }}>{m.desc}</span>
                      </div>
                    ))}
                  </Card>
                </div>
              </>
            )}
          </div>
        )}

        {/* ════════════════════════════════════════════════════════
            TAB: REGISTRY
        ════════════════════════════════════════════════════════ */}
        {tab === "registry" && (
          <div className="fade-in">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
              <div>
                <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>🗄️ Model Registry</h2>
                <p style={{ color: "#555", fontSize: 13 }}>Version control for trained models. Promote, demote, and manage deployments.</p>
              </div>
              <div style={{ display: "flex", gap: 10, flexDirection: "column", alignItems: "flex-end" }}>
                <button
                  className="action-btn"
                  onClick={handleTrain}
                  disabled={trainLoading}
                  style={{ background: trainLoading ? "#1a1a2e" : "#A29BFE", color: "#04040f" }}>
                  {trainLoading ? "⟳ Training..." : "🔄 Train New Model"}
                </button>
                {trainMsg && (
                  <div style={{ fontSize: 11, color: "#6BCB77", fontFamily: "monospace", maxWidth: 300, textAlign: "right" }}>
                    {trainMsg}
                  </div>
                )}
              </div>
            </div>

            {!modelList && <div style={{ color: "#555" }}>Loading registry...</div>}
            {modelList?.error && <Card><p style={{ color: "#FF6348" }}>⚠️ Registry unavailable — API offline?</p></Card>}
            {modelList && !modelList.error && (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 20 }}>
                  <MetricCard label="Production" value={modelList.production_version || "—"} color="#6BCB77" />
                  <MetricCard label="Total Versions" value={modelList.total_versions} color="#4ECDC4" />
                  <MetricCard label="Staging" value={modelList.versions?.filter(v => v.status === "staging").length || 0} color="#FFD93D" />
                </div>

                <Card>
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                      <thead>
                        <tr style={{ borderBottom: "2px solid #1a1a2e" }}>
                          {["Version", "Model", "Status", "Accuracy", "F1", "Created", "Actions"].map(h => (
                            <th key={h} style={{ padding: "10px 14px", textAlign: "left", color: "#444", fontFamily: "monospace", fontSize: 10 }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {modelList.versions?.map(v => (
                          <tr key={v.version} style={{ borderBottom: "1px solid #0d0d1a" }}>
                            <td style={{ padding: "10px 14px", fontFamily: "monospace", color: "#ddd" }}>{v.version}</td>
                            <td style={{ padding: "10px 14px", color: "#888" }}>{v.model_key?.replace("_", " ")}</td>
                            <td style={{ padding: "10px 14px" }}>
                              <Pill
                                label={v.status}
                                color={v.status === "production" ? "#6BCB77" : v.status === "staging" ? "#FFD93D" : "#555"}
                              />
                            </td>
                            <td style={{ padding: "10px 14px", fontFamily: "monospace", color: "#4ECDC4" }}>
                              {v.accuracy ? `${(v.accuracy * 100).toFixed(1)}%` : "—"}
                            </td>
                            <td style={{ padding: "10px 14px", fontFamily: "monospace", color: "#A29BFE" }}>
                              {v.macro_f1?.toFixed(3) || "—"}
                            </td>
                            <td style={{ padding: "10px 14px", color: "#555", fontSize: 11 }}>
                              {new Date(v.created_at).toLocaleDateString()}
                            </td>
                            <td style={{ padding: "10px 14px" }}>
                              {v.status !== "production" && (
                                <button
                                  onClick={() => handlePromote(v.version)}
                                  style={{
                                    padding: "4px 12px", borderRadius: 6, border: "1px solid #6BCB77",
                                    background: "transparent", color: "#6BCB77", cursor: "pointer",
                                    fontSize: 11, fontFamily: "monospace",
                                  }}>
                                  Promote ↑
                                </button>
                              )}
                              {v.status === "production" && (
                                <span style={{ fontSize: 11, color: "#6BCB77" }}>✓ Live</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* MLOps explanation */}
                <Card style={{ marginTop: 20 }}>
                  <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>📚 Model Versioning Concepts</h3>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                    {[
                      { icon: "🟡", label: "Staging", desc: "Newly trained model. Passes tests but not yet serving live traffic. Equivalent to a 'candidate' release." },
                      { icon: "🟢", label: "Production", desc: "The single model serving all /predict requests. Only one version can be production at a time. Promotion is atomic." },
                      { icon: "⚫", label: "Archived", desc: "Superseded production model. Kept for rollback. Automatically pruned when MAX_REGISTERED_VERSIONS is exceeded." },
                    ].map(s => (
                      <div key={s.label}>
                        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>{s.icon} {s.label}</div>
                        <p style={{ fontSize: 12, color: "#666", lineHeight: 1.7 }}>{s.desc}</p>
                      </div>
                    ))}
                  </div>
                </Card>
              </>
            )}
          </div>
        )}

        {/* ════════════════════════════════════════════════════════
            TAB: DATASET
        ════════════════════════════════════════════════════════ */}
        {tab === "dataset" && (
          <div className="fade-in">
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>📦 Training Dataset</h2>
            <p style={{ color: "#555", fontSize: 13, marginBottom: 20 }}>
              Spotify-style audio features dataset — real anchors + synthetic samples.
            </p>

            {datasetStats && !datasetStats.error && (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 20 }}>
                  {GENRES.map(g => (
                    <div
                      key={g}
                      onClick={() => setFilterGenre(filterGenre === g ? null : g)}
                      style={{
                        background: "#0a0a1a", borderRadius: 12, padding: "12px 10px",
                        textAlign: "center", cursor: "pointer",
                        border: `1px solid ${filterGenre === g ? GENRE_COLORS[g] : GENRE_COLORS[g] + "33"}`,
                        transition: "all 0.15s",
                      }}>
                      <div style={{ fontSize: 18, fontWeight: 800, color: GENRE_COLORS[g], fontFamily: "monospace" }}>
                        {datasetStats.genre_counts?.[g] || 0}
                      </div>
                      <div style={{ fontSize: 10, color: "#666", marginTop: 2 }}>{g}</div>
                    </div>
                  ))}
                </div>
                <p style={{ fontSize: 11, color: "#444", fontFamily: "monospace", marginBottom: 16 }}>
                  Total: {datasetStats.total_tracks} tracks · {datasetStats.features?.length} features · Click a genre to filter
                </p>
              </>
            )}

            {datasetTracks && (
              <Card>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: "2px solid #1a1a2e" }}>
                        {["Title", "Artist", "Genre", "Dance", "Energy", "Valence", "Tempo", "Acoustic", "Speech"].map(h => (
                          <th key={h} style={{ padding: "10px 12px", textAlign: "left", color: "#444", fontFamily: "monospace", fontSize: 10 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {datasetTracks.tracks?.map((t, i) => (
                        <tr key={i} style={{ borderBottom: "1px solid #0a0a16" }}>
                          <td style={{ padding: "9px 12px", color: "#ddd", fontWeight: 600, maxWidth: 140, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{t.title}</td>
                          <td style={{ padding: "9px 12px", color: "#555", maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{t.artist}</td>
                          <td style={{ padding: "9px 12px" }}><Pill label={t.genre} color={GENRE_COLORS[t.genre]} /></td>
                          {["danceability", "energy", "valence"].map(f => (
                            <td key={f} style={{ padding: "9px 12px", fontFamily: "monospace", color: t[f] > 0.65 ? "#4ECDC4" : "#555" }}>
                              {t[f]?.toFixed(2)}
                            </td>
                          ))}
                          <td style={{ padding: "9px 12px", fontFamily: "monospace", color: "#A29BFE" }}>{Math.round(t.tempo)}</td>
                          <td style={{ padding: "9px 12px", fontFamily: "monospace", color: t.acousticness > 0.4 ? "#FFD93D" : "#555" }}>{t.acousticness?.toFixed(2)}</td>
                          <td style={{ padding: "9px 12px", fontFamily: "monospace", color: t.speechiness > 0.15 ? "#FF6B9D" : "#555" }}>{t.speechiness?.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p style={{ fontSize: 10, color: "#333", marginTop: 10, fontFamily: "monospace" }}>
                  Showing {datasetTracks.tracks?.length} of {datasetTracks.total} tracks
                  {filterGenre ? ` · Filtered: ${filterGenre}` : ""}
                </p>
              </Card>
            )}
          </div>
        )}

        {/* ════════════════════════════════════════════════════════
            TAB: API
        ════════════════════════════════════════════════════════ */}
        {tab === "api" && (
          <div className="fade-in">
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>📡 API Integration</h2>
            <p style={{ color: "#555", fontSize: 13, marginBottom: 20 }}>
              Live HTTP request/response viewer + full endpoint reference.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
              <Card>
                <h3 style={{ fontSize: 13, fontWeight: 700, marginBottom: 12, fontFamily: "monospace", color: "#4ECDC4" }}>
                  LIVE REQUEST/RESPONSE
                </h3>
                <ApiPanel features={features} result={result} loading={predLoading} />
                <button
                  className="action-btn"
                  onClick={handlePredict}
                  disabled={predLoading}
                  style={{ marginTop: 12, width: "100%", background: "#4ECDC4", color: "#04040f" }}>
                  {predLoading ? "⟳ Sending..." : "▶ Send Request"}
                </button>
              </Card>

              <Card>
                <h3 style={{ fontSize: 13, fontWeight: 700, marginBottom: 12, fontFamily: "monospace", color: "#A29BFE" }}>
                  ENDPOINT REFERENCE
                </h3>
                <div style={{ fontSize: 12, fontFamily: "monospace" }}>
                  {[
                    { method: "POST", path: "/predict", desc: "Classify a track", color: "#4ECDC4" },
                    { method: "POST", path: "/predict/batch", desc: "Classify multiple tracks", color: "#4ECDC4" },
                    { method: "GET",  path: "/models", desc: "List all model versions", color: "#FFD93D" },
                    { method: "GET",  path: "/models/active", desc: "Get production model", color: "#FFD93D" },
                    { method: "POST", path: "/models/promote/{v}", desc: "Promote to production", color: "#4ECDC4" },
                    { method: "POST", path: "/models/train", desc: "Trigger retraining", color: "#4ECDC4" },
                    { method: "GET",  path: "/dataset/stats", desc: "Dataset statistics", color: "#FFD93D" },
                    { method: "GET",  path: "/dataset/tracks", desc: "Paginated track list", color: "#FFD93D" },
                    { method: "GET",  path: "/metrics", desc: "Model performance", color: "#FFD93D" },
                    { method: "GET",  path: "/health", desc: "Liveness probe", color: "#6BCB77" },
                    { method: "GET",  path: "/health/ready", desc: "Readiness probe", color: "#6BCB77" },
                  ].map(e => (
                    <div key={e.path} style={{
                      display: "flex", gap: 10, alignItems: "center",
                      padding: "7px 0", borderBottom: "1px solid #0d0d1a",
                    }}>
                      <span style={{ width: 40, color: e.color, fontWeight: 700, fontSize: 10 }}>{e.method}</span>
                      <span style={{ flex: 1, color: "#888" }}>/api/v1{e.path}</span>
                      <span style={{ color: "#444", fontSize: 10 }}>{e.desc}</span>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 14 }}>
                  <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" style={{
                    display: "inline-block", padding: "8px 16px", borderRadius: 8,
                    background: "#1a1a2e", color: "#4ECDC4", fontSize: 12, fontFamily: "monospace",
                    textDecoration: "none", border: "1px solid #4ECDC444",
                  }}>
                    📖 Open Swagger UI →
                  </a>
                </div>
              </Card>
            </div>

            {/* SDK examples */}
            <Card>
              <h3 style={{ fontSize: 13, fontWeight: 700, marginBottom: 16, fontFamily: "monospace", color: "#FFD93D" }}>
                INTEGRATION EXAMPLES
              </h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                {[
                  { lang: "Python (httpx)", code: `import httpx\n\nresponse = httpx.post(\n  "http://localhost:8000/api/v1/predict",\n  json={\n    "danceability": 0.72,\n    "energy": 0.85,\n    "valence": 0.64,\n    "tempo": 128,\n    "acousticness": 0.02,\n    "speechiness": 0.04\n  }\n)\nprint(response.json()["predicted_genre"])` },
                  { lang: "JavaScript (fetch)", code: `const res = await fetch(\n  "http://localhost:8000/api/v1/predict",\n  {\n    method: "POST",\n    headers: {"Content-Type": "application/json"},\n    body: JSON.stringify({\n      danceability: 0.72,\n      energy: 0.85,\n      valence: 0.64,\n      tempo: 128,\n      acousticness: 0.02,\n      speechiness: 0.04\n    })\n  }\n);\nconst {predicted_genre} = await res.json();` },
                  { lang: "cURL", code: `curl -X POST \\\n  http://localhost:8000/api/v1/predict \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "danceability": 0.72,\n    "energy": 0.85,\n    "valence": 0.64,\n    "tempo": 128,\n    "acousticness": 0.02,\n    "speechiness": 0.04\n  }'` },
                ].map(ex => (
                  <div key={ex.lang}>
                    <p style={{ fontSize: 10, color: "#555", fontFamily: "monospace", marginBottom: 8 }}>{ex.lang}</p>
                    <pre style={{
                      background: "#060612", borderRadius: 8, padding: 12,
                      fontSize: 10, color: "#888", fontFamily: "monospace",
                      overflowX: "auto", border: "1px solid #1a1a2e",
                      lineHeight: 1.6, whiteSpace: "pre",
                    }}>{ex.code}</pre>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{
        borderTop: "1px solid #0d0d1a", padding: "16px 24px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginTop: 40, fontSize: 10, color: "#333", fontFamily: "monospace",
      }}>
        <span>SoundML · FastAPI + scikit-learn + React · Educational ML System</span>
        <span>GET /api/v1/health · GET /docs</span>
      </div>
    </div>
  );
}
