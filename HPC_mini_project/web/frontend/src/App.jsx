import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const [intervalStyle, setIntervalStyle] = useState('threshold');
  const [isIntervalOpen, setIsIntervalOpen] = useState(false);
  
  const [thresholdMode, setThresholdMode] = useState(1);
  const [direction, setDirection] = useState(0);
  const [threshold, setThreshold] = useState(300);
  const [sortingStyle, setSortingStyle] = useState(0); 
  const [masking, setMasking] = useState(0); // 0: None, 1: Inside, 2: Outside
  
  const [maskRect, setMaskRect] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [startPos, setStartPos] = useState({x: 0, y: 0});
  const imageRef = useRef(null);
  const wrapperRef = useRef(null);
  
  const [zoom, setZoom] = useState(1);

  const fileInputRef = useRef(null);

  const getComputedMode = () => {
    if (intervalStyle === 'none') return 4;
    if (intervalStyle === 'random') return 5;
    return thresholdMode; // 1, 2, or 3
  };

  const stateRef = useRef({ 
    mode: 1, direction: 0, threshold: 300, sortingStyle: 0, masking: 0, maskRect: null 
  });

  useEffect(() => {
    stateRef.current = { 
      mode: getComputedMode(), direction, threshold, sortingStyle, masking, maskRect 
    };
  }, [intervalStyle, thresholdMode, direction, threshold, sortingStyle, masking, maskRect]);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResultUrl(null);
      setMaskRect(null);
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
    formData.append('sorting_style', stateRef.current.sortingStyle);
    formData.append('masking', stateRef.current.masking);

    if (stateRef.current.maskRect) {
      const {x,y,w,h} = stateRef.current.maskRect;
      formData.append('mask_rect', `${x},${y},${w},${h}`);
    }

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

  const getRenderedImageRect = () => {
    if (!imageRef.current) return null;
    const img = imageRef.current;
    const rect = img.getBoundingClientRect();
    const naturalW = img.naturalWidth;
    const naturalH = img.naturalHeight;
    if (!naturalW || !naturalH) return rect;

    const scale = Math.min(rect.width / naturalW, rect.height / naturalH);
    const actualW = naturalW * scale;
    const actualH = naturalH * scale;
    
    const letterboxX = (rect.width - actualW) / 2;
    const letterboxY = (rect.height - actualH) / 2;

    return {
      left: rect.left + letterboxX,
      top: rect.top + letterboxY,
      width: actualW,
      height: actualH
    };
  };

  const handleMouseDown = (e) => {
    if (masking === 0 || !imageRef.current) return;
    const r = getRenderedImageRect();
    if (!r) return;
    
    let x = (e.clientX - r.left) / r.width;
    let y = (e.clientY - r.top) / r.height;
    
    // Clamp to exactly image bounds so they can't draw outside the visible photo
    x = Math.max(0, Math.min(1, x));
    y = Math.max(0, Math.min(1, y));
    
    setStartPos({x, y});
    setIsDragging(true);
    setMaskRect({x, y, w: 0, h: 0});
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !imageRef.current) return;
    const r = getRenderedImageRect();
    if (!r) return;
    
    let x = (e.clientX - r.left) / r.width;
    let y = (e.clientY - r.top) / r.height;
    
    x = Math.max(0, Math.min(1, x));
    y = Math.max(0, Math.min(1, y));

    setMaskRect({
      x: Math.min(x, startPos.x),
      y: Math.min(y, startPos.y),
      w: Math.abs(x - startPos.x),
      h: Math.abs(y - startPos.y)
    });
  };

  const handleMouseUp = () => {
    if (isDragging) {
      setIsDragging(false);
      setTimeout(processImage, 50);
    }
  };

  const handleWheel = (e) => {
    // Only zoom if pressing ctrl, otherwise let them scroll if zoomed
    if (e.ctrlKey) {
      e.preventDefault(); // Requires passive: false in native event, but React ignores this. Handled below via CSS.
      const delta = e.deltaY < 0 ? 0.1 : -0.1;
      setZoom((z) => Math.max(0.1, Math.min(z + delta, 10)));
    } else {
      const delta = e.deltaY < 0 ? 0.1 : -0.1;
      setZoom((z) => Math.max(0.1, Math.min(z + delta, 10)));
    }
  };

  return (
    <div className="studio-layout">
      <header className="studio-header">
        <div className="logo">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <line x1="3" y1="9" x2="21" y2="9"></line>
            <line x1="9" y1="21" x2="9" y2="9"></line>
          </svg>
          GLITCH_ART_STUDIO <span>v1.0.0 // CUDA</span>
        </div>
        <div className="header-tools">
          <div className="menu-item" onClick={() => fileInputRef.current.click()}>File</div>
          <div className="menu-item">Edit</div>
          <div className="menu-item">View</div>
          <div className="menu-item">Help</div>
        </div>
      </header>

      <div className="studio-workspace">
        <aside className="studio-sidebar">
          <div className="sidebar-header">
            <h3>PROPERTIES</h3>
          </div>
          
          <div className="panel-content">
            <div className="property-group">
              <label>SOURCE MEDIA</label>
              <div className="file-input-wrapper" onClick={() => fileInputRef.current.click()}>
                <span className="file-name">{selectedImage ? selectedImage.name : "No Media Selected..."}</span>
                <span className="file-btn">BROWSE</span>
              </div>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleImageChange} 
                accept="image/*"
                style={{ display: 'none' }}
              />
            </div>

            <div className="property-divider"></div>

            <div className="property-group">
              <label>INTERVAL STYLE</label>
              <div className="interval-dropdown">
                <div className="interval-header" onClick={() => setIsIntervalOpen(!isIntervalOpen)}>
                  <span className="interval-active-name">{intervalStyle.toUpperCase()}</span>
                  <span className="chevron">{isIntervalOpen ? '⌃' : '⌄'}</span>
                </div>
                {isIntervalOpen && (
                  <div className="interval-list">
                    <div className={`style-card ${intervalStyle === 'none' ? 'active' : ''}`} onClick={() => { setIntervalStyle('none'); setIsIntervalOpen(false); setTimeout(processImage, 50); }}>
                      <div className="card-label">NONE</div>
                      <div className="gradient-bar none"></div>
                    </div>
                    <div className={`style-card ${intervalStyle === 'threshold' ? 'active' : ''}`} onClick={() => { setIntervalStyle('threshold'); setIsIntervalOpen(false); setTimeout(processImage, 50); }}>
                      <div className="card-label">THRESHOLD</div>
                      <div className="gradient-bar threshold"></div>
                    </div>
                    <div className={`style-card ${intervalStyle === 'random' ? 'active' : ''}`} onClick={() => { setIntervalStyle('random'); setIsIntervalOpen(false); setTimeout(processImage, 50); }}>
                      <div className="card-label">RANDOM</div>
                      <div className="gradient-bar random"></div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {intervalStyle === 'threshold' && (
              <div className="property-group" style={{marginTop: '4px'}}>
                <label>THRESHOLD BEHAVIOR</label>
                <div className="custom-select">
                  <select 
                    value={thresholdMode} 
                    onChange={(e) => { 
                      setThresholdMode(parseInt(e.target.value)); 
                      setTimeout(processImage, 50); 
                    }}
                  >
                    <option value={1}>Sort Exceeding Threshold</option>
                    <option value={2}>Sort Below Threshold</option>
                    <option value={3}>First Band Only</option>
                  </select>
                </div>
              </div>
            )}

            <div className="property-group">
              <label>SORT DIRECTION</label>
              <div className="segmented-control">
                <button className={direction === 0 ? 'active' : ''} onClick={() => { setDirection(0); setTimeout(processImage, 50); }}>L → R</button>
                <button className={direction === 1 ? 'active' : ''} onClick={() => { setDirection(1); setTimeout(processImage, 50); }}>R ← L</button>
                <button className={direction === 2 ? 'active' : ''} onClick={() => { setDirection(2); setTimeout(processImage, 50); }}>T ↓ B</button>
                <button className={direction === 3 ? 'active' : ''} onClick={() => { setDirection(3); setTimeout(processImage, 50); }}>B ↑ T</button>
              </div>
            </div>

            <div className="property-group">
              <div className="slider-header">
                <label>{intervalStyle === 'random' ? 'RANDOM BAND RANGE' : 'THRESHOLD VALUE'}</label>
                <span className="val-badge">{threshold}</span>
              </div>
              <input 
                type="range" 
                className="pro-slider"
                min={intervalStyle === 'random' ? "10" : "0"} 
                max={intervalStyle === 'random' ? "800" : "765"} 
                value={threshold} 
                onChange={(e) => setThreshold(e.target.value)}
                onMouseUp={() => setTimeout(processImage, 50)}
                onTouchEnd={() => setTimeout(processImage, 50)}
              />
            </div>

            <div className="property-divider"></div>

            <div className="property-group">
              <label>SORTING VALUE</label>
              <div className="custom-select">
                <select 
                  value={sortingStyle} 
                  onChange={(e) => { 
                    setSortingStyle(parseInt(e.target.value)); 
                    setTimeout(processImage, 50); 
                  }}
                >
                  <option value={0}>Lightness (Value)</option>
                  <option value={1}>Hue</option>
                  <option value={2}>Saturation</option>
                  <option value={3}>Red Channel</option>
                  <option value={4}>Green Channel</option>
                  <option value={5}>Blue Channel</option>
                </select>
              </div>
            </div>

            <div className="property-group">
              <label>SPATIAL MASKING</label>
              <div className="segmented-control">
                <button className={masking === 0 ? 'active' : ''} onClick={() => { setMasking(0); setMaskRect(null); setTimeout(processImage, 50); }}>NONE</button>
                <button className={masking === 1 ? 'active' : ''} onClick={() => { setMasking(1); setTimeout(processImage, 50); }}>SORT INSIDE</button>
                <button className={masking === 2 ? 'active' : ''} onClick={() => { setMasking(2); setTimeout(processImage, 50); }}>SORT OUTSIDE</button>
              </div>
              {masking > 0 && <p className="help-text">Draw a box on the image to mask!</p>}
            </div>

          </div>

          <div className="panel-footer">
            <button 
              className="primary-btn" 
              onClick={processImage}
              disabled={!selectedImage || isProcessing}
            >
              {isProcessing ? 'COMPILING KERNEL...' : 'RENDER TO CANVAS'}
            </button>
            
            {resultUrl && (
              <a href={resultUrl} download="glitch_art.jpg" tabIndex="-1">
                <button className="secondary-btn">EXPORT AS JPEG</button>
              </a>
            )}
          </div>
        </aside>

        <main className="studio-canvas">
          <div className="canvas-toolbar">
            <div className="zoom-controls">
              <span>{isProcessing ? 'STATUS: PROCESSING...' : resultUrl ? 'STATUS: RENDERED' : selectedImage ? 'STATUS: READY' : 'STATUS: IDLE'}</span>
              <span style={{ marginLeft: '16px' }}>ZOOM: {Math.round(zoom * 100)}%</span>
            </div>
            <div className="canvas-info">
              GPU: NVIDIA CUDA // MEM: SHARED
            </div>
          </div>
          
          <div className="canvas-inner" onWheel={handleWheel}>
            {isProcessing && <div className="processing-overlay"><div className="loader"></div></div>}
            
            {!selectedImage && (
              <div className="empty-canvas">
                <div className="empty-icon">⌘</div>
                <p>IMPORT MEDIA TO BEGIN</p>
              </div>
            )}

            {(resultUrl || previewUrl) && (
              <div 
                className={`image-wrapper ${masking > 0 ? 'drawing-mode' : ''}`}
                ref={wrapperRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                style={{ 
                  width: `${zoom * 100}%`, 
                  height: `${zoom * 100}%`,
                  transition: 'width 0.1s ease-out, height 0.1s ease-out'
                }}
              >
                <img 
                  ref={imageRef}
                  src={resultUrl || previewUrl} 
                  alt="Canvas" 
                  className="canvas-img" 
                  draggable="false" 
                />
                
                {masking > 0 && maskRect && maskRect.w > 0 && maskRect.h > 0 && (() => {
                  const r = getRenderedImageRect();
                  const wr = wrapperRef.current?.getBoundingClientRect();
                  if (!r || !wr) return null;
                  
                  // Map the image-relative coordinates back to the wrapper for CSS positioning
                  const leftPx = r.left - wr.left + (maskRect.x * r.width);
                  const topPx = r.top - wr.top + (maskRect.y * r.height);
                  const widthPx = maskRect.w * r.width;
                  const heightPx = maskRect.h * r.height;

                  return (
                    <div className="mask-overlay" style={{
                      left: `${(leftPx / wr.width) * 100}%`,
                      top: `${(topPx / wr.height) * 100}%`,
                      width: `${(widthPx / wr.width) * 100}%`,
                      height: `${(heightPx / wr.height) * 100}%`,
                      borderColor: masking === 1 ? '#00e5ff' : '#ff3366',
                      backgroundColor: masking === 1 ? 'rgba(0, 229, 255, 0.15)' : 'rgba(255, 51, 102, 0.15)'
                    }}></div>
                  );
                })()}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
