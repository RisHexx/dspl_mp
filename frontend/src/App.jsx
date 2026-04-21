import { useState, useRef, useCallback, useEffect } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";
const CONFIDENCE_THRESHOLD = 0.82;

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
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null); // { sign_name, class_id, confidence }
  const [snapshot, setSnapshot] = useState(null); // data URL
  const [error, setError] = useState(null);
  const [latency, setLatency] = useState(0);
  const isProcessingRef = useRef(false);

  /* ── Consecutive-match confirmation ref ─────────────────────────────── */
  const previousPredictionRef = useRef(null); // class_id from last frame

  /* ── TTS cooldown refs ─────────────────────────────────────────────── */
  const lastSpokenSignRef = useRef(null);
  const lastSpokenTimeRef = useRef(0);

  const speakAlert = useCallback((classId, signName) => {
    const now = Date.now();
    // Don't repeat the same sign within 8 seconds
    if (
      lastSpokenSignRef.current === classId &&
      now - lastSpokenTimeRef.current < 8000
    ) {
      return;
    }

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const message = SPEECH_MESSAGES[classId] || signName + " detected.";
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    window.speechSynthesis.speak(utterance);

    lastSpokenSignRef.current = classId;
    lastSpokenTimeRef.current = now;
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

    // Save snapshot for display
    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
    setSnapshot(dataUrl);

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

          // Only show result if confidence meets threshold AND
          // the same sign was detected in the previous frame too
          if (
            data.confidence >= CONFIDENCE_THRESHOLD &&
            previousPredictionRef.current === data.class_id
          ) {
            setResult(data);
            speakAlert(data.class_id, data.sign_name);
          } else if (data.confidence < CONFIDENCE_THRESHOLD) {
            setResult(null);
          }
          // Store current prediction for next-frame comparison
          previousPredictionRef.current =
            data.confidence >= CONFIDENCE_THRESHOLD ? data.class_id : null;
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
      0.9
    );
  }, []);

  /* ── Auto-capture interval ───────────────────────────────────────────── */
  useEffect(() => {
    let intervalId;
    if (cameraOn) {
      intervalId = setInterval(() => {
        captureAndPredict();
      }, 1000);
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
        <h1>Traffic Sign Detector</h1>
        {/* <div className="header-meta">
          <span className="status-badge">
            <span className={`status-dot ${cameraOn ? "active" : ""}`}></span>
            {cameraOn ? "System Active" : "System Standby"}
          </span>
          <span className="meta-divider">•</span>
          <span>GTSRB Model</span>
    </div> */}
      </header >

      {/* Main content */}
      < main className="main" >
        {/* Camera panel */}
        < section className="camera-panel" id="camera-panel" >
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

            {!cameraOn && (
              <div className="camera-placeholder">
                <div className="icon">📷</div>
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
                📷 Start Camera
              </button>
            ) : (
              <button
                className="btn btn-secondary"
                onClick={stopCamera}
                id="stop-camera-btn"
              >
                ⏹ Stop Camera
              </button>
            )}
          </div>
        </section >

        {/* Result panel */}
        < aside className="result-panel" id="result-panel" >
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
                <div className="icon">⚠️</div>
                <p>{error}</p>
              </div>
            )}

            {/* Result */}
            {result && !loading && (
              <div className="detected-sign">
                <div className="sign-name">{result.sign_name}</div>

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
            )}

            {/* Empty state */}
            {!result && !loading && !error && (
              <div className="result-empty">
                <div className="icon">🎯</div>
                <p>
                  Start the camera and point it at a traffic sign. It will
                  <strong> automatically detect</strong> the sign.
                </p>
              </div>
            )}
          </div>
        </aside >
      </main >
    </div >
  );
}

export default App;
