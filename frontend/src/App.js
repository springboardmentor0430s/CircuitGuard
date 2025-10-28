import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { Upload, Image as ImageIcon, CheckCircle, AlertCircle, FileText, BarChart3, PieChart as PieChartIcon, TrendingUp } from 'lucide-react';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
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

  const onTemplateDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0] || e.target.files[0];
    if (file) {
      setTemplateFile(file);
      setError(null);
      const reader = new FileReader();
      reader.onload = () => setTemplatePreview(reader.result);
      reader.readAsDataURL(file);
    }
  }, []);

  const onTestDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0] || e.target.files[0];
    if (file) {
      setTestFile(file);
      setError(null);
      const reader = new FileReader();
      reader.onload = () => setTestPreview(reader.result);
      reader.readAsDataURL(file);
    }
  }, []);

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

  // Prepare chart data
  const getDefectTypeChartData = () => {
    if (!results?.frequency_analysis) return [];
    return Object.entries(results.frequency_analysis).map(([type, data]) => ({
      name: type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      count: data.count,
      percentage: data.percentage
    }));
  };

  const getConfidenceChartData = () => {
    if (!results?.confidence_stats) return [];
    return [
      { name: 'High (≥80%)', count: results.confidence_stats.high_confidence_count, fill: '#10b981' },
      { name: 'Medium (50-80%)', count: results.confidence_stats.medium_confidence_count, fill: '#f59e0b' },
      { name: 'Low (<50%)', count: results.confidence_stats.low_confidence_count, fill: '#ef4444' }
    ];
  };

  const getSizeDistributionData = () => {
    if (!results?.size_distribution) return [];
    return Object.entries(results.size_distribution).map(([key, data]) => ({
      name: data.label,
      count: data.count,
      percentage: data.percentage
    }));
  };

  const COLORS = ['#667eea', '#764ba2', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6'];

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
                className="dropzone"
                onDrop={onTemplateDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => document.getElementById('template-input').click()}
              >
                <input id="template-input" type="file" accept="image/*" onChange={onTemplateDrop} style={{ display: 'none' }} />
                {templateFile ? (
                  <div className="file-preview">
                    {templatePreview && <img src={templatePreview} alt="Template preview" className="preview-image" />}
                    <p>{templateFile.name}</p>
                    <CheckCircle className="check-icon" />
                  </div>
                ) : (
                  <div className="dropzone-content">
                    <Upload size={48} />
                    <p>Drag & drop template image or click to select</p>
                  </div>
                )}
              </div>
            </div>

            <div className="upload-box">
              <h3>Test Image (To be inspected)</h3>
              <div 
                className="dropzone"
                onDrop={onTestDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => document.getElementById('test-input').click()}
              >
                <input id="test-input" type="file" accept="image/*" onChange={onTestDrop} style={{ display: 'none' }} />
                {testFile ? (
                  <div className="file-preview">
                    {testPreview && <img src={testPreview} alt="Test preview" className="preview-image" />}
                    <p>{testFile.name}</p>
                    <CheckCircle className="check-icon" />
                  </div>
                ) : (
                  <div className="dropzone-content">
                    <Upload size={48} />
                    <p>Drag & drop test image or click to select</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {error && (
            <div className="error-message">
              <AlertCircle />
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
                <button 
                  className="pdf-button" 
                  onClick={downloadPDFReport}
                  disabled={isProcessing || isDownloadingPDF}
                >
                  <FileText />
                  {isDownloadingPDF ? 'Generating PDF...' : 'Download Report'}
                </button>
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
                {/* Charts Section */}
                <div className="charts-section">
                  <h3 className="charts-title">
                    <BarChart3 size={24} />
                    Statistical Analysis
                  </h3>
                  
                  <div className="charts-grid">


                    {/* Defect Type Pie Chart */}
                    <div className="chart-card">
                      <h4>Defect Type Percentage</h4>
                      <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                          <Pie 
                            data={getDefectTypeChartData()} 
                            dataKey="count" 
                            nameKey="name" 
                            cx="50%" 
                            cy="50%" 
                            outerRadius={80} 
                            label={(entry) => `${entry.percentage}%`}
                          >
                            {getDefectTypeChartData().map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Confidence Distribution */}
                    <div className="chart-card">
                      <h4>Confidence Level Distribution</h4>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={getConfidenceChartData()}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" style={{ fontSize: '12px' }} />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="count">
                            {getConfidenceChartData().map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Size Distribution */}
                    {results.size_distribution && (
                      <div className="chart-card">
                        <h4>Defect Size Distribution</h4>
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart data={getSizeDistributionData()}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" angle={-15} textAnchor="end" height={80} style={{ fontSize: '12px' }} />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="count" fill="#8b5cf6" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </div>
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
                          <th>Area (px²)</th>
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
                            <td className="defect-area-cell">{defect.area}</td>
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
                <CheckCircle size={48} />
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