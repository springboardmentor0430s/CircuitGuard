import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiImage, FiCheckCircle, FiAlertCircle, FiDownload } from 'react-icons/fi';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [templateFile, setTemplateFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [templatePreview, setTemplatePreview] = useState(null);
  const [testPreview, setTestPreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const onTemplateDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    setTemplateFile(file);
    setError(null);
    // Create preview URL
    const reader = new FileReader();
    reader.onload = () => setTemplatePreview(reader.result);
    reader.readAsDataURL(file);
  }, []);

  const onTestDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    setTestFile(file);
    setError(null);
    // Create preview URL
    const reader = new FileReader();
    reader.onload = () => setTestPreview(reader.result);
    reader.readAsDataURL(file);
  }, []);

  const { getRootProps: getTemplateRootProps, getInputProps: getTemplateInputProps, isDragActive: isTemplateDragActive } = useDropzone({
    onDrop: onTemplateDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  const { getRootProps: getTestRootProps, getInputProps: getTestInputProps, isDragActive: isTestDragActive } = useDropzone({
    onDrop: onTestDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  const processImages = async () => {
    if (!templateFile || !testFile) {
      setError('Please upload both template and test images');
      return;
    }

    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('template', templateFile);
    formData.append('test', testFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to process images');
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadResults = () => {
    if (!results) return;
    
    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `pcb-defect-report-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const resetForm = () => {
    setTemplateFile(null);
    setTestFile(null);
    setTemplatePreview(null);
    setTestPreview(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>CircuitGuard</h1>
        <p>PCB Defect Detection & Classification</p>
      </header>

      <main className="app-main">
        <div className="upload-section">
          <div className="upload-container">
            <div className="upload-box">
              <h3>Template Image (Defect-free)</h3>
              <div 
                {...getTemplateRootProps()} 
                className={`dropzone ${isTemplateDragActive ? 'active' : ''}`}
              >
                <input {...getTemplateInputProps()} />
                {templateFile ? (
                  <div className="file-preview">
                    {templatePreview ? (
                      <img src={templatePreview} alt="Template preview" className="preview-image" />
                    ) : (
                      <FiImage size={48} />
                    )}
                    <p>{templateFile.name}</p>
                    <FiCheckCircle className="check-icon" />
                  </div>
                ) : (
                  <div className="dropzone-content">
                    <FiUpload size={48} />
                    <p>{isTemplateDragActive ? 'Drop the file here' : 'Drag & drop template image or click to select'}</p>
                  </div>
                )}
              </div>
            </div>

            <div className="upload-box">
              <h3>Test Image (To be inspected)</h3>
              <div 
                {...getTestRootProps()} 
                className={`dropzone ${isTestDragActive ? 'active' : ''}`}
              >
                <input {...getTestInputProps()} />
                {testFile ? (
                  <div className="file-preview">
                    {testPreview ? (
                      <img src={testPreview} alt="Test preview" className="preview-image" />
                    ) : (
                      <FiImage size={48} />
                    )}
                    <p>{testFile.name}</p>
                    <FiCheckCircle className="check-icon" />
                  </div>
                ) : (
                  <div className="dropzone-content">
                    <FiUpload size={48} />
                    <p>{isTestDragActive ? 'Drop the file here' : 'Drag & drop test image or click to select'}</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {error && (
            <div className="error-message">
              <FiAlertCircle />
              <span>{error}</span>
            </div>
          )}

          <button 
            className="process-button" 
            onClick={processImages}
            disabled={!templateFile || !testFile || isProcessing}
          >
            {isProcessing ? 'Processing...' : 'Detect Defects'}
          </button>
        </div>

        {results && (
          <div className="results-section">
            <div className="results-header">
              <h2>Detection Results</h2>
              <div className="action-buttons">
                <button className="download-button" onClick={downloadResults}>
                  <FiDownload />
                  Download Report
                </button>
                <button className="reset-button" onClick={resetForm}>
                  Process New Images
                </button>
              </div>
            </div>

            {results.images && (
              <div className="image-visualization-full">
                <h3>Annotated Result</h3>
                <div className="annotated-image-main">
                  <img src={results.images.result} alt="Annotated Result" className="main-annotated-image" />
                  <div className="secondary-images-grid">
                    <div className="secondary-image-item">
                      <h4>Template Image</h4>
                      <img src={results.images.template} alt="Template" className="secondary-image" />
                    </div>
                    <div className="secondary-image-item">
                      <h4>Test Image</h4>
                      <img src={results.images.test} alt="Test" className="secondary-image" />
                    </div>
                    <div className="secondary-image-item">
                      <h4>Defect Mask</h4>
                      <img src={results.images.defect_mask} alt="Defect Mask" className="secondary-image" />
                    </div>
                  </div>
                </div>
                
              </div>
            )}

            <div className="summary-horizontal">
              <div className="stat">
                <span className="stat-value">{results.defect_count}</span>
                <span className="stat-label">Defects Found</span>
              </div>
              <div className="stat">
                <span className="stat-value">{results.processing_time ? 'Success' : 'N/A'}</span>
                <span className="stat-label">Status</span>
              </div>
              {results.confidence_stats && (
                <div className="stat">
                  <span className="stat-value">{(results.confidence_stats.average * 100).toFixed(1)}%</span>
                  <span className="stat-label">Avg Confidence</span>
                </div>
              )}
            </div>

            <div className="defects-table-section">
              <h3>Detected Defects</h3>
              {results.defects && results.defects.length > 0 ? (
                <div className="defects-table-container">
                  <table className="defects-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Defect Type</th>
                        <th>Position</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.defects.map((defect, index) => (
                        <tr key={defect.id}>
                          <td className="defect-id-cell">#{defect.id}</td>
                          <td className="defect-type-cell">{defect.class_name.replace('_', ' ')}</td>
                          <td className="defect-position-cell">({defect.bbox.x}, {defect.bbox.y})</td>
                          <td className="defect-confidence-cell">
                            {defect.confidence ? `${(defect.confidence * 100).toFixed(1)}%` : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="no-defects">
                  <FiCheckCircle size={48} />
                  <p>No defects detected!</p>
                </div>
              )}
            </div>

            {results.defects && results.defects.length > 0 && (
              <div className="frequency-analysis">
                <h3>Defect Frequency Analysis</h3>
                <div className="frequency-chart">
                  {results.frequency_analysis ? 
                    Object.entries(results.frequency_analysis).map(([type, data]) => (
                      <div key={type} className="frequency-item">
                        <span className="frequency-type">{type.replace('_', ' ')}</span>
                        <div className="frequency-bar">
                          <div 
                            className="frequency-fill" 
                            style={{ 
                              width: `${data.percentage}%` 
                            }}
                          />
                        </div>
                        <span className="frequency-count">{data.count} ({data.percentage}%)</span>
                      </div>
                    )) :
                    Object.entries(
                      results.defects.reduce((acc, defect) => {
                        acc[defect.class_name] = (acc[defect.class_name] || 0) + 1;
                        return acc;
                      }, {})
                    ).map(([type, count]) => (
                      <div key={type} className="frequency-item">
                        <span className="frequency-type">{type.replace('_', ' ')}</span>
                        <div className="frequency-bar">
                          <div 
                            className="frequency-fill" 
                            style={{ 
                              width: `${(count / results.defects.length) * 100}%` 
                            }}
                          />
                        </div>
                        <span className="frequency-count">{count}</span>
                      </div>
                    ))
                  }
                </div>
              </div>
            )}

            {results.confidence_stats && (
              <div className="confidence-analysis">
                <h3>Confidence Analysis</h3>
                <div className="confidence-grid">
                  <div className="confidence-item">
                    <span className="confidence-label">Average Confidence</span>
                    <span className="confidence-value">{(results.confidence_stats.average * 100).toFixed(1)}%</span>
                  </div>
                  <div className="confidence-item">
                    <span className="confidence-label">High Confidence (80%)</span>
                    <span className="confidence-value">{results.confidence_stats.high_confidence_count}</span>
                  </div>
                  <div className="confidence-item">
                    <span className="confidence-label">Medium Confidence (50-80%)</span>
                    <span className="confidence-value">{results.confidence_stats.medium_confidence_count}</span>
                  </div>
                  <div className="confidence-item">
                    <span className="confidence-label">Low Confidence (50%)</span>
                    <span className="confidence-value">{results.confidence_stats.low_confidence_count}</span>
                  </div>
                </div>
              </div>
            )}

            
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>CircuitGuard PCB Defect Detection System</p>
      </footer>
    </div>
  );
}

export default App;
