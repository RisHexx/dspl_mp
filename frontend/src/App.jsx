import { useState, useRef, useCallback, useEffect } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";
const CONFIDENCE_THRESHOLD = 0.82;
const CAPTURE_INTERVAL_MS = 300;
// A sign must be out of view this long before it's announced again
const SIGN_GONE_MS = 5000;

/* ── Box colours by sign family ──────────────────────────────────────── */
const DANGER = new Set([11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]);
const PROHIBITORY = new Set([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 14, 15, 16, 17]);
const MANDATORY = new Set([33, 34, 35, 36, 37, 38, 39, 40]);
const LOW_CONF_COLOR = "#94a3b8";

function boxColor(classId) {
  if (PROHIBITORY.has(classId)) return "#ef4444"; // red
  if (DANGER.has(classId) || classId === 13) return "#f59e0b"; // amber
  if (MANDATORY.has(classId)) return "#3b82f6"; // blue
  if (classId === 12) return "#eab308"; // priority road – yellow
  return "#a3a3a3"; // "end of …" signs – grey
}

function signCategory(classId) {
  if (PROHIBITORY.has(classId)) return "Prohibitory";
  if (DANGER.has(classId)) return "Warning";
  if (MANDATORY.has(classId)) return "Mandatory";
  if (classId === 12 || classId === 13) return "Priority";
  return "End of restriction";
}

/** Cut a square thumbnail of a detected sign out of the captured frame. */
function cropThumb(source, box, size = 128) {
  const [x1, y1, x2, y2] = box;
  const side = Math.max(x2 - x1, y2 - y1) * 1.15;
  const sx = Math.max(0, Math.min(source.width - side, (x1 + x2) / 2 - side / 2));
  const sy = Math.max(0, Math.min(source.height - side, (y1 + y2) / 2 - side / 2));
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  c.getContext("2d").drawImage(source, sx, sy, side, side, 0, 0, size, size);
  return c.toDataURL("image/jpeg", 0.85);
}

/* ── Context-aware speech messages for each GTSRB class ──────────────── */
const SPEECH_MESSAGES = {
  0: "Speed limit 20, go very slow.",
  1: "Speed limit 30, maintain slow speed.",
  2: "Speed limit 50, drive at moderate speed.",
  3: "Speed limit 60, please slow down.",
  4: "Speed limit 70, watch your speed.",
  5: "Speed limit 80, maintain steady speed.",
  6: "End of 80 speed limit, you may accelerate.",
  7: "Speed limit 100, drive carefully at high speed.",
  8: "Speed limit 120, high speed zone, stay alert.",
  9: "No passing zone, do not overtake.",
  10: "No passing for heavy vehicles.",
  11: "Right of way at next intersection, proceed with caution.",
  12: "Priority road, you have the right of way.",
  13: "Yield sign, give way to other vehicles.",
  14: "Stop sign ahead, stop your vehicle completely.",
  15: "No vehicles allowed ahead.",
  16: "Heavy vehicles prohibited on this road.",
  17: "No entry, do not enter this road.",
  18: "General caution, be alert for hazards.",
  19: "Dangerous curve to the left, slow down.",
  20: "Dangerous curve to the right, slow down.",
  21: "Double curve ahead, reduce speed.",
  22: "Bumpy road ahead, slow down.",
  23: "Slippery road, drive carefully.",
  24: "Road narrows on the right, proceed with caution.",
  25: "Road work ahead, slow down and watch for workers.",
  26: "Traffic signals ahead, prepare to stop.",
  27: "Pedestrians ahead, watch for people crossing.",
  28: "Children crossing, slow down and be very careful.",
  29: "Bicycles crossing, watch for cyclists.",
  30: "Beware of ice or snow, drive carefully.",
  31: "Wild animals crossing, stay alert.",
  32: "End of all speed and passing limits.",
  33: "Turn right ahead.",
  34: "Turn left ahead.",
  35: "Ahead only, continue straight.",
  36: "Go straight or turn right.",
  37: "Go straight or turn left.",
  38: "Keep right.",
  39: "Keep left.",
  40: "Roundabout ahead, yield to traffic in the circle.",
  41: "End of no passing zone.",
  42: "End of no passing for heavy vehicles.",
};

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null); // hidden – used to grab frames
  const overlayRef = useRef(null); // visible – bounding boxes drawn here
  const streamRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [loading] = useState(false);
  const [results, setResults] = useState([]); // confirmed detections
  const [mode, setMode] = useState(null); // "detector" | "fallback"
  const [error, setError] = useState(null);
  const [latency, setLatency] = useState(0);
  const isProcessingRef = useRef(false);

  /* ── Consecutive-match confirmation ref ─────────────────────────────── */
  // class_ids seen (above threshold) in the previous frame. A sign is only
  // "confirmed" once it shows up in two frames in a row – this stops the
  // label from flickering on a single bad frame.
  const previousClassesRef = useRef(new Set());

  /* ── Voice alerts ──────────────────────────────────────────────────── */
  // class_id → last time that sign was confirmed in a frame. A sign is only
  // announced when it *appears*: while it stays in view it is not repeated,
  // and it must be out of view for SIGN_GONE_MS before it's announced again
  // (so a single missed frame doesn't trigger a repeat).
  const lastSeenRef = useRef(new Map());

  const speakAlert = useCallback((classId, signName) => {
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const message = SPEECH_MESSAGES[classId] || signName + " detected.";
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    window.speechSynthesis.speak(utterance);
  }, []);

  /* ── Bounding-box overlay ────────────────────────────────────────────── */
  // The overlay canvas uses the same pixel size as the video frame and the
  // same CSS `object-fit: cover`, so box coordinates from the backend (which
  // are in frame pixels) line up with the video without any manual scaling.
  function clearOverlay() {
    const overlay = overlayRef.current;
    if (overlay) overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  }

  const drawDetections = useCallback((detections, imageSize, confirmed) => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const [w, h] = imageSize;
    if (overlay.width !== w) overlay.width = w;
    if (overlay.height !== h) overlay.height = h;

    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, w, h);

    const lineWidth = Math.max(2, Math.round(w / 220));
    const fontSize = Math.max(12, Math.round(w / 40));
    ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
    ctx.textBaseline = "top";

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.box;
      const sure = det.confidence >= CONFIDENCE_THRESHOLD;
      const color = sure ? boxColor(det.class_id) : LOW_CONF_COLOR;
      const label = sure
        ? `${det.sign_name} · ${(det.confidence * 100).toFixed(0)}%`
        : `? ${(det.confidence * 100).toFixed(0)}%`;

      // Solid = confirmed over 2 frames, dashed = seen once / unsure
      ctx.setLineDash(sure && confirmed.has(det.class_id) ? [] : [8, 6]);
      ctx.lineWidth = lineWidth;
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label tag above the box (or inside if there's no room above)
      ctx.setLineDash([]);
      const pad = 4;
      const tw = ctx.measureText(label).width + pad * 2;
      const th = fontSize + pad * 2;
      const ty = y1 - th >= 0 ? y1 - th : y1;
      const tx = Math.min(x1, w - tw);
      ctx.fillStyle = color;
      ctx.fillRect(tx, ty, tw, th);
      ctx.fillStyle = "#0b1120";
      ctx.fillText(label, tx + pad, ty + pad);
    }
  }, []);

  /* ── Start / stop webcam ─────────────────────────────────────────────── */
  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 640, height: 480 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraOn(true);
    } catch (err) {
      setError("Could not access camera. Please allow camera permissions.");
      console.error(err);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    clearOverlay();
    previousClassesRef.current = new Set();
    setResults([]);
    setCameraOn(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  /* ── Capture & predict ───────────────────────────────────────────────── */
  const captureAndPredict = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;
    if (isProcessingRef.current) return;

    isProcessingRef.current = true;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to Blob and send to backend
    const startTime = performance.now();
    canvas.toBlob(
      async (blob) => {
        if (!blob) {
          setError("Failed to capture image from camera.");
          isProcessingRef.current = false;
          return;
        }

        setError(null);

        try {
          const formData = new FormData();
          formData.append("file", blob, "capture.jpg");

          const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
          }

          const data = await response.json();
          setLatency(Math.round(performance.now() - startTime));

          setMode(data.mode);

          const detections = data.detections || [];
          const confident = detections.filter(
            (d) => d.confidence >= CONFIDENCE_THRESHOLD
          );
          // Confirmed = confident now AND also seen in the previous frame
          const confirmedIds = new Set(
            confident
              .filter((d) => previousClassesRef.current.has(d.class_id))
              .map((d) => d.class_id)
          );
          previousClassesRef.current = new Set(confident.map((d) => d.class_id));

          // Stop drawing if the camera was turned off while we were waiting
          if (!streamRef.current) return;
          drawDetections(detections, data.image_size, confirmedIds);

          // One entry per class (highest confidence wins)
          const seen = new Set();
          const confirmed = confident
            .filter((d) => {
              if (!confirmedIds.has(d.class_id) || seen.has(d.class_id)) return false;
              seen.add(d.class_id);
              return true;
            })
            .map((d) => ({ ...d, thumb: cropThumb(canvas, d.box) }));
          setResults(confirmed);
          // Announce only signs that just came into view (strongest first –
          // detections are sorted by confidence)
          const nowMs = Date.now();
          const lastSeen = lastSeenRef.current;
          const appeared = confirmed.find(
            (d) => nowMs - (lastSeen.get(d.class_id) ?? 0) > SIGN_GONE_MS
          );
          confirmed.forEach((d) => lastSeen.set(d.class_id, nowMs));
          if (appeared) speakAlert(appeared.class_id, appeared.sign_name);
        } catch (err) {
          console.error(err);
          setError(
            err.message.includes("Failed to fetch")
              ? "Cannot connect to backend. Make sure the FastAPI server is running on port 8000."
              : err.message
          );
        } finally {
          isProcessingRef.current = false;
        }
      },
      "image/jpeg",
      0.85
    );
  }, [drawDetections, speakAlert]);

  /* ── Auto-capture interval ───────────────────────────────────────────── */
  // A new frame is only sent once the previous response has arrived
  // (isProcessingRef), so this just sets the maximum frame rate.
  useEffect(() => {
    let intervalId;
    if (cameraOn) {
      intervalId = setInterval(() => {
        captureAndPredict();
      }, CAPTURE_INTERVAL_MS);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [cameraOn, captureAndPredict]);

  /* ── Confidence level helper ─────────────────────────────────────────── */
  const getConfidenceLevel = (conf) => {
    if (conf >= 0.8) return "high";
    if (conf >= 0.5) return "medium";
    return "low";
  };

  /* ── Render ──────────────────────────────────────────────────────────── */
  return (
    <div className="app">
      {/* Animated Background */}
      <div className="background-animations">
        <div className="moving-road"></div>
        <svg className="bg-shape shape-1" viewBox="0 0 100 100">
           <polygon points="50,15 90,85 10,85" fill="none" stroke="currentColor" strokeWidth="6" strokeLinejoin="round"/>
        </svg>
        <svg className="bg-shape shape-2" viewBox="0 0 100 100">
           <circle cx="50" cy="50" r="35" fill="none" stroke="currentColor" strokeWidth="6"/>
        </svg>
        <svg className="bg-shape shape-3" viewBox="0 0 100 100">
           <polygon points="30,15 70,15 85,30 85,70 70,85 30,85 15,70 15,30" fill="none" stroke="currentColor" strokeWidth="6" strokeLinejoin="round"/>
        </svg>
        <svg className="bg-shape shape-4" viewBox="0 0 100 100">
           <polygon points="50,15 85,50 50,85 15,50" fill="none" stroke="currentColor" strokeWidth="6" strokeLinejoin="round"/>
        </svg>
      </div>

      {/* Header */}
      <header className="header">
        <div className="header-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 7V5a2 2 0 0 1 2-2h2" />
            <path d="M17 3h2a2 2 0 0 1 2 2v2" />
            <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
            <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        </div>
        <div className="header-title">
          <h1>Traffic Sign Detector</h1>
          <p>Real-time recognition of 43 traffic sign classes</p>
        </div>
        <div className="header-meta">
          <span className="status-badge">
            <span className={`status-dot ${cameraOn ? "active" : ""}`}></span>
            {cameraOn ? "Active" : "Standby"}
          </span>
        </div>
      </header>

      {/* Main content */}
      <main className="main">
        {/* Camera panel */}
        <section className="camera-panel" id="camera-panel">
          <div className="panel-header">
            {cameraOn && <span className="dot" />}
            <span>{cameraOn ? "Live Camera Feed" : "Camera"}</span>
          </div>

          <div className="camera-feed">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ display: cameraOn ? "block" : "none" }}
            />
            <canvas ref={canvasRef} />
            <canvas
              ref={overlayRef}
              className="overlay-canvas"
              style={{ display: cameraOn ? "block" : "none" }}
            />

            {!cameraOn && (
              <div className="camera-placeholder">
                <div className="icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 7l-7 5 7 5V7z" />
                    <rect x="1" y="5" width="15" height="14" rx="2" />
                  </svg>
                </div>
                <p className="placeholder-title">Camera is off</p>
                <p>Start the camera to begin detecting traffic signs</p>
              </div>
            )}
          </div>

          <div className="camera-controls">
            {!cameraOn ? (
              <button
                className="btn btn-primary"
                onClick={startCamera}
                id="start-camera-btn"
              >
                Start Camera
              </button>
            ) : (
              <button
                className="btn btn-secondary"
                onClick={stopCamera}
                id="stop-camera-btn"
              >
                Stop Camera
              </button>
            )}
          </div>
        </section>

        {/* Result panel */}
        <aside className="result-panel" id="result-panel">
          <div className="panel-header">
            <span>Detection Result</span>
          </div>

          <div className="result-content">
            {/* Loading state */}
            {loading && (
              <div className="loading-spinner">
                <div className="spinner" />
                <p>Analyzing image…</p>
              </div>
            )}

            {/* Error state */}
            {error && !loading && (
              <div className="error-display">
                <p className="error-title">Something went wrong</p>
                <p>{error}</p>
              </div>
            )}

            {/* Results – main sign, then any others in view */}
            {results.length > 0 && !loading && (
              <div className="detections-list">
                {results.map((result, i) => (
                  <div
                    className={`detected-sign ${i === 0 ? "primary" : "secondary"}`}
                    key={result.class_id}
                  >
                    <div className="sign-row">
                      <img className="sign-thumb" src={result.thumb} alt={result.sign_name} />
                      <div className="sign-text">
                        <span
                          className="sign-category"
                          style={{ color: boxColor(result.class_id) }}
                        >
                          {signCategory(result.class_id)}
                        </span>
                        <div className="sign-name">{result.sign_name}</div>
                      </div>
                    </div>

                    {i === 0 && SPEECH_MESSAGES[result.class_id] && (
                      <p className="sign-advice">{SPEECH_MESSAGES[result.class_id]}</p>
                    )}

                    <div className="confidence-section">
                      <div className="confidence-label">
                        <span>Confidence</span>
                        <span className="confidence-value">
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="confidence-bar-bg">
                        <div
                          className={`confidence-bar-fill ${getConfidenceLevel(result.confidence)}`}
                          style={{ width: `${result.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}

                <div className="detections-meta">
                  {results.length} sign{results.length > 1 ? "s" : ""} detected
                  {latency > 0 && <> · {latency} ms</>}
                  {mode === "fallback" && <> · no detector loaded</>}
                </div>
              </div>
            )}

            {/* Empty state */}
            {results.length === 0 && !loading && !error && (
              <div className="result-empty">
                <div className="icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round">
                    <path d="M12 3l9.5 16.5h-19z" />
                    <path d="M12 10v4" strokeLinecap="round" />
                    <circle cx="12" cy="16.8" r="0.5" fill="currentColor" />
                  </svg>
                </div>
                <p className="placeholder-title">
                  {cameraOn ? "Looking for signs…" : "No sign detected yet"}
                </p>
                <p>
                  Point the camera at a traffic sign and it will be detected
                  automatically.
                </p>
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
