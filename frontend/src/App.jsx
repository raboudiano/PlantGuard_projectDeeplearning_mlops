import { useEffect, useRef, useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const FALLBACK_THRESHOLD = 0.6;
const MAX_CAPTURE = 1024;

const prettyLabel = (name) =>
  name.replace(/___/g, ': ').replace(/_/g, ' ').replace(/\s+/g, ' ').trim();

function splitLabel(name) {
  const [plant, condition = ''] = prettyLabel(name).split(': ');
  return { plant, condition };
}

export default function App() {
  const [mode, setMode] = useState('upload');
  const [result, setResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [webcamOn, setWebcamOn] = useState(false);
  const [dragging, setDragging] = useState(false);

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const abortRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => () => {
    stopWebcam();
    abortRef.current?.abort();
  }, []);

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  async function predictFile(file) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    setResult(null);
    setError('');
    setStatus('Analyzing leaf…');
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Prediction request failed');
      setResult(payload);
      setStatus('');
    } catch (err) {
      if (err.name === 'AbortError') return;
      setError(err.message || 'Prediction error');
      setStatus('');
    } finally {
      if (abortRef.current === controller) setLoading(false);
    }
  }

  function showAndPredict(file) {
    setPreviewUrl(URL.createObjectURL(file));
    predictFile(file);
  }

  function handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please choose an image file.');
      return;
    }
    showAndPredict(file);
  }

  async function startWebcam() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Webcam is not supported in this browser.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setWebcamOn(true);
      setError('');
    } catch (err) {
      setError(
        err.name === 'NotAllowedError'
          ? 'Camera permission denied. Allow camera access and try again.'
          : 'Could not start the camera.'
      );
    }
  }

  function stopWebcam() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setWebcamOn(false);
  }

  function switchMode(next) {
    setMode(next);
    setError('');
    if (next === 'camera') startWebcam();
    else stopWebcam();
  }

  function capture() {
    const video = videoRef.current;
    if (!webcamOn || !video?.videoWidth) return;
    const side = Math.min(video.videoWidth, video.videoHeight);
    const out = Math.min(side, MAX_CAPTURE);
    const sx = (video.videoWidth - side) / 2;
    const sy = (video.videoHeight - side) / 2;
    const canvas = document.createElement('canvas');
    canvas.width = out;
    canvas.height = out;
    canvas.getContext('2d').drawImage(video, sx, sy, side, side, 0, 0, out, out);
    canvas.toBlob(
      (blob) => blob && showAndPredict(new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' })),
      'image/jpeg',
      0.95
    );
  }

  function reset() {
    abortRef.current?.abort();
    setResult(null);
    setPreviewUrl('');
    setError('');
    setStatus('');
    setLoading(false);
  }

  const uncertain = result && (result.is_uncertain ?? result.confidence < FALLBACK_THRESHOLD);
  const top = result ? splitLabel(result.predicted_class) : null;
  const healthy = top && /healthy/i.test(top.condition);
  const tone = uncertain ? 'warn' : healthy ? 'good' : 'bad';

  return (
    <div className="page">
      <div className="glow glow-a" />
      <div className="glow glow-b" />

      <header className="hero">
        <div className="logo">🌿</div>
        <h1>Plant<span>Guard</span></h1>
        <p>AI-powered plant disease detection</p>
      </header>

      <main className="panel">
        <div className="tabs" role="tablist">
          <button className={mode === 'upload' ? 'tab active' : 'tab'} onClick={() => switchMode('upload')}>
            Upload
          </button>
          <button className={mode === 'camera' ? 'tab active' : 'tab'} onClick={() => switchMode('camera')}>
            Camera
          </button>
        </div>

        <div className="stage">
          {/* input area */}
          {!previewUrl && mode === 'upload' && (
            <div
              className={dragging ? 'drop dragging' : 'drop'}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
            >
              <div className="drop-icon">⬆</div>
              <strong>Drop a leaf photo here</strong>
              <span>or click to browse</span>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => { handleFile(e.target.files[0]); e.target.value = ''; }}
          />

          <div className="camera" style={{ display: mode === 'camera' && !previewUrl ? 'block' : 'none' }}>
            <video ref={videoRef} autoPlay muted playsInline />
            <div className="guide" />
            {!webcamOn && <div className="camera-hint">Starting camera…</div>}
          </div>

          {mode === 'camera' && !previewUrl && (
            <button className="primary" onClick={capture} disabled={!webcamOn || loading}>
              Capture leaf
            </button>
          )}

          {/* preview + result */}
          {previewUrl && (
            <div className="review">
              <div className="photo">
                <img src={previewUrl} alt="Leaf preview" />
                {loading && <div className="scan"><span /></div>}
              </div>

              <div className="verdict">
                {loading && <p className="muted">{status}</p>}

                {result && top && (
                  <>
                    <span className={`badge ${tone}`}>
                      {uncertain ? 'Uncertain' : healthy ? 'Healthy' : 'Disease detected'}
                    </span>
                    <h2>{top.plant}</h2>
                    <p className="condition">{top.condition || '—'}</p>

                    <div className="meter">
                      <div className="meter-head">
                        <span>Confidence</span>
                        <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                      </div>
                      <div className="bar"><i className={tone} style={{ width: `${result.confidence * 100}%` }} /></div>
                    </div>

                    {uncertain && (
                      <p className="note">
                        Retake the photo: one leaf, centered, good light, plain background.
                      </p>
                    )}

                    <ul className="alts">
                      {result.top_predictions.slice(1).map((item) => {
                        const l = splitLabel(item.class_name);
                        return (
                          <li key={item.class_name}>
                            <span>{l.plant}{l.condition ? ` · ${l.condition}` : ''}</span>
                            <em>{(item.confidence * 100).toFixed(1)}%</em>
                          </li>
                        );
                      })}
                    </ul>

                    <p className="meta">
                      {result.inference_time_ms.toFixed(0)} ms · model v{result.model_version}
                    </p>
                  </>
                )}

                <button className="ghost" onClick={reset}>Analyze another leaf</button>
              </div>
            </div>
          )}

          {error && <p className="error" role="alert">{error}</p>}
        </div>
      </main>

      <footer>For guidance only. Confirm serious cases with an agronomist.</footer>
    </div>
  );
}