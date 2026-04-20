import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const [mode, setMode] = useState(1);
  const [direction, setDirection] = useState(0);
  const [threshold, setThreshold] = useState(300);
  const fileInputRef = useRef(null);

  // Use a ref to guarantee processImage always reads the absolute latest state
  const stateRef = useRef({ mode: 1, direction: 0, threshold: 300 });

  // Keep ref synced with state
  useEffect(() => {
    stateRef.current = { mode, direction, threshold };
  }, [mode, direction, threshold]);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResultUrl(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('drag-active');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-active');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-active');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResultUrl(null);
    }
  };

  const processImage = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', selectedImage);
    formData.append('threshold', stateRef.current.threshold);
    formData.append('mode', stateRef.current.mode);
    formData.append('direction', stateRef.current.direction);

    try {
      const response = await fetch('http://localhost:8000/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Processing failed');
      }

      const blob = await response.blob();
      setResultUrl(URL.createObjectURL(blob));
    } catch (error) {
      console.error('Error:', error);
      alert('Error: ' + error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Glitch Art Studio</h1>
        <p className="subtitle">CUDA-Accelerated Pixel Sorting</p>
      </header>

      <div className="control-panel">
        <div 
          className="upload-box" 
          onClick={() => fileInputRef.current.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleImageChange} 
            accept="image/*"
          />
          <div className="upload-text">
            {selectedImage ? (
              <p>Selected: <span>{selectedImage.name}</span></p>
            ) : (
              <p>Drag & Drop an image here or <span>Click to Browse</span></p>
            )}
          </div>
        </div>

        <div className="control-group">
          <label>Algorithm Mode</label>
          <div className="mode-selector">
            <button 
              className={`mode-btn ${mode === 1 ? 'active' : ''}`}
              onClick={() => { setMode(1); setTimeout(processImage, 50); }}
            >
              Sort Brightest <span className="mode-desc">T &gt; {threshold}</span>
            </button>
            <button 
              className={`mode-btn ${mode === 2 ? 'active' : ''}`}
              onClick={() => { setMode(2); setTimeout(processImage, 50); }}
            >
              Sort Darkest <span className="mode-desc">T &lt; {threshold}</span>
            </button>
            <button 
              className={`mode-btn ${mode === 3 ? 'active' : ''}`}
              onClick={() => { setMode(3); setTimeout(processImage, 50); }}
            >
              First Band Only <span className="mode-desc">T &gt; {threshold}</span>
            </button>
          </div>
        </div>

        <div className="control-group">
          <label>Sort Direction</label>
          <div className="mode-selector grid-2">
            <button className={`mode-btn center-text ${direction === 0 ? 'active' : ''}`} onClick={() => { setDirection(0); setTimeout(processImage, 50); }}>Left → Right</button>
            <button className={`mode-btn center-text ${direction === 1 ? 'active' : ''}`} onClick={() => { setDirection(1); setTimeout(processImage, 50); }}>Right ← Left</button>
            <button className={`mode-btn center-text ${direction === 2 ? 'active' : ''}`} onClick={() => { setDirection(2); setTimeout(processImage, 50); }}>Top ↓ Bottom</button>
            <button className={`mode-btn center-text ${direction === 3 ? 'active' : ''}`} onClick={() => { setDirection(3); setTimeout(processImage, 50); }}>Bottom ↑ Top</button>
          </div>
        </div>

        <div className="control-group threshold-group">
          <div className="threshold-header">
            <label>Brightness Threshold</label>
            <div className="threshold-val">{threshold}</div>
          </div>
          <input 
            type="range" 
            className="range-slider"
            min="0" max="765" 
            value={threshold} 
            onChange={(e) => setThreshold(e.target.value)}
            onMouseUp={() => setTimeout(processImage, 50)}
            onTouchEnd={() => setTimeout(processImage, 50)}
          />
        </div>

        <button 
          className="action-btn" 
          onClick={processImage}
          disabled={!selectedImage || isProcessing}
        >
          {isProcessing ? 'Processing on GPU...' : 'GENERATE GLITCH ART'}
        </button>

        {resultUrl && (
          <a href={resultUrl} download="glitch_art.jpg" style={{textDecoration:'none'}}>
            <button className="action-btn" style={{width:'100%', background:'transparent', border:'1px solid var(--accent-2)', color:'var(--accent-2)', marginTop:'0'}}>
              Download Image
            </button>
          </a>
        )}
      </div>

      <div className="preview-panel">
        {isProcessing && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p>Running CUDA Kernel...</p>
          </div>
        )}
        
        <div className="image-container">
          {resultUrl ? (
            <img src={resultUrl} alt="Processed Art" />
          ) : previewUrl ? (
            <img src={previewUrl} alt="Original Preview" />
          ) : (
            <div className="empty-state">
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p>Upload an image to see preview</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
