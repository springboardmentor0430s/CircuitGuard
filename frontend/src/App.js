import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiImage, FiCheckCircle, FiAlertCircle, FiFileText } from 'react-icons/fi';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [templateFile, setTemplateFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [templatePreview, setTemplatePreview] = useState(null);
  const [testPreview, setTestPreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDownloadingPDF, setIsDownloadingPDF] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const onTemplateDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    setTemplateFile(file);
    setError(null);
    const reader = new FileReader();
    reader.onload = () => setTemplatePreview(reader.result);
    reader.readAsDataURL(file);
  }, []);

  const onTestDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    setTestFile(file);
    setError(null);
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

  const downloadPDFReport = async () => {
    if (!templateFile || !testFile) {
      setError('Please upload both template and test images');
      return;
    }

    setIsDownloadingPDF(true);
    setError(null);

    const formData = new FormData();
    formData.append('template', templateFile);
    formData.append('test', testFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/process-pdf`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob',
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
      link.download = `CircuitGuard_Report_${timestamp}.pdf`;
      
      document.body.appendChild(link);
      link.click();
      
      link.remove();
      window.URL.revokeObjectURL(url);
      
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate PDF report');
    } finally {
      setIsDownloadingPDF(false);
    }
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

          <div className="button-group">
            <button 
              className="process-button" 
              onClick={processImages}
              disabled={!templateFile || !testFile || isProcessing || isDownloadingPDF}
            >
              {isProcessing ? 'Processing...' : 'Detect Defects'}
            </button>
          </div>
        </div>

        {results && (
          <div className="results-section">
            <div className="results-header">
              <h2>Detection Results</h2>
              <div className='result-buttons'>
                {results && (
                  <button 
                    className="pdf-button" 
                    onClick={downloadPDFReport}
                    disabled={isProcessing || isDownloadingPDF}
                  >
                    <FiFileText />
                    {isDownloadingPDF ? 'Generating PDF...' : 'Download Report'}
                  </button>
                )}
                  <button className="reset-button" onClick={resetForm}>
                    Process New Images
                  </button>
              </div>
            </div>

            {results.images && (
              <div className="image-visualization-full">
                <div className="main-result-container">
                  <h3>Annotated Result</h3>
                  <img src={results.images.result} alt="Annotated Result" className="main-annotated-image" />
                </div>
                
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
            )}

            <div className="stats-overview">
              <div className="stat-card primary">
                <div className="stat-icon">🔍</div>
                <div className="stat-content">
                  <span className="stat-value">{results.defect_count}</span>
                  <span className="stat-label">Total Defects</span>
                </div>
              </div>
              
              <div className="stat-card">
                <div className="stat-icon">✓</div>
                <div className="stat-content">
                  <span className="stat-value">
                    {results.defect_count === 0 ? 'Pass' : 'Fail'}
                  </span>
                  <span className="stat-label">Quality Status</span>
                </div>
              </div>
              
              {results.confidence_stats && (
                <div className="stat-card">
                  <div className="stat-icon">📊</div>
                  <div className="stat-content">
                    <span className="stat-value">{(results.confidence_stats.average * 100).toFixed(1)}%</span>
                    <span className="stat-label">Avg Confidence</span>
                  </div>
                </div>
              )}

              {results.frequency_analysis && (
                <div className="stat-card">
                  <div className="stat-icon">🏷️</div>
                  <div className="stat-content">
                    <span className="stat-value">{Object.keys(results.frequency_analysis).length}</span>
                    <span className="stat-label">Defect Types</span>
                  </div>
                </div>
              )}
            </div>

            {results.defects && results.defects.length > 0 && (
              <>
                <div className="analysis-grid">
                  <div className="analysis-card">
                    <h3>Defect Type Distribution</h3>
                    <div className="frequency-chart">
                      {results.frequency_analysis && 
                        Object.entries(results.frequency_analysis).map(([type, data]) => (
                          <div key={type} className="frequency-item">
                            <span className="frequency-type">{type.replace('_', ' ')}</span>
                            <div className="frequency-bar">
                              <div 
                                className="frequency-fill" 
                                style={{ width: `${data.percentage}%` }}
                              />
                            </div>
                            <span className="frequency-count">{data.count} ({data.percentage}%)</span>
                          </div>
                        ))
                      }
                    </div>
                  </div>

                  {results.confidence_stats && (
                    <div className="analysis-card">
                      <h3>Confidence Distribution</h3>
                      <div className="confidence-bars">
                        <div className="confidence-bar-item">
                          <div className="confidence-bar-header">
                            <span>High (≥80%)</span>
                            <span className="confidence-bar-value">{results.confidence_stats.high_confidence_count}</span>
                          </div>
                          <div className="confidence-bar-track">
                            <div 
                              className="confidence-bar-fill high"
                              style={{ 
                                width: `${(results.confidence_stats.high_confidence_count / results.defect_count) * 100}%` 
                              }}
                            />
                          </div>
                        </div>
                        
                        <div className="confidence-bar-item">
                          <div className="confidence-bar-header">
                            <span>Medium (50-80%)</span>
                            <span className="confidence-bar-value">{results.confidence_stats.medium_confidence_count}</span>
                          </div>
                          <div className="confidence-bar-track">
                            <div 
                              className="confidence-bar-fill medium"
                              style={{ 
                                width: `${(results.confidence_stats.medium_confidence_count / results.defect_count) * 100}%` 
                              }}
                            />
                          </div>
                        </div>
                        
                        <div className="confidence-bar-item">
                          <div className="confidence-bar-header">
                            <span>Low (&lt;50%)</span>
                            <span className="confidence-bar-value">{results.confidence_stats.low_confidence_count}</span>
                          </div>
                          <div className="confidence-bar-track">
                            <div 
                              className="confidence-bar-fill low"
                              style={{ 
                                width: `${(results.confidence_stats.low_confidence_count / results.defect_count) * 100}%` 
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <div className="defects-table-section">
                  <h3>Detailed Defect List</h3>
                  <div className="defects-table-container">
                    <table className="defects-table">
                      <thead>
                        <tr>
                          <th>ID</th>
                          <th>Defect Type</th>
                          <th>Position (X, Y)</th>
                          <th>Size (W × H)</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.defects.map((defect) => (
                          <tr key={defect.id}>
                            <td className="defect-id-cell">#{defect.id}</td>
                            <td className="defect-type-cell">{defect.class_name.replace('_', ' ')}</td>
                            <td className="defect-position-cell">({defect.bbox.x}, {defect.bbox.y})</td>
                            <td className="defect-size-cell">{defect.bbox.width} × {defect.bbox.height}</td>
                            <td className="defect-confidence-cell">
                              {defect.confidence ? `${(defect.confidence * 100).toFixed(1)}%` : 'N/A'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}

            {results.defects && results.defects.length === 0 && (
              <div className="no-defects">
                <FiCheckCircle size={48} />
                <p>No defects detected! PCB is within quality standards.</p>
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