import React, { useState, useCallback, useRef, useEffect } from 'react';
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

  // Dropdown state for download options
  const [downloadMenuOpen, setDownloadMenuOpen] = useState(false);
  const downloadMenuRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(e) {
      if (downloadMenuRef.current && !downloadMenuRef.current.contains(e.target)) {
        setDownloadMenuOpen(false);
      }
    }
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const downloadAnnotatedImage = () => {
    if (!results?.images?.result) {
      setError('No annotated image available');
      return;
    }

    try {
      const dataUrl = results.images.result; 
      const link = document.createElement('a');
      link.href = dataUrl;
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      link.download = `annotated_result_${timestamp}.jpg`;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError('Failed to download annotated image');
    } finally {
      setDownloadMenuOpen(false);
    }
  };

const downloadLogsCSV = () => {
  if (!results?.defects) {
    setError('No defect logs available');
    return;
  }

  try {
    // Header with metadata
    let csv = '# CircuitGuard PCB Defect Detection Report\n';
    csv += `# Report Generated: ${new Date().toISOString()}\n`;
    csv += `# Total Defects: ${results.defect_count}\n`;
    csv += `# Quality Status: ${results.defect_count === 0 ? 'PASS' : 'FAIL'}\n`;
    
    if (results.confidence_stats) {
      csv += `# Average Confidence: ${(results.confidence_stats.average * 100).toFixed(1)}%\n`;
      csv += `# Min Confidence: ${(results.confidence_stats.min * 100).toFixed(1)}%\n`;
      csv += `# Max Confidence: ${(results.confidence_stats.max * 100).toFixed(1)}%\n`;
      csv += `# High Confidence Count (≥80%): ${results.confidence_stats.high_confidence_count}\n`;
      csv += `# Medium Confidence Count (50-80%): ${results.confidence_stats.medium_confidence_count}\n`;
      csv += `# Low Confidence Count (<50%): ${results.confidence_stats.low_confidence_count}\n`;
    }
    
    csv += '#\n';
    csv += '# Defect Type Distribution\n';
    if (results.frequency_analysis) {
      Object.entries(results.frequency_analysis).forEach(([type, data]) => {
        csv += `# ${type.replace(/_/g, ' ').toUpperCase()}: ${data.count} (${data.percentage}%)\n`;
      });
    }
    
    csv += '#\n';
    csv += '# Size Distribution\n';
    if (results.size_distribution) {
      Object.entries(results.size_distribution).forEach(([key, data]) => {
        csv += `# ${data.label}: ${data.count} (${data.percentage}%)\n`;
      });
    }
    
    csv += '#\n';
    csv += '# Detailed Defect List\n';
    
    // Main data table header
    csv += 'ID,Defect Type,Confidence (%),Center X,Center Y,BBox X,BBox Y,Width,Height,Area (px²),Severity\n';
    
    // Severity mapping
    const severityMap = {
      'open_circuit': 'Critical',
      'short': 'Critical',
      'missing_hole': 'High',
      'mouse_bite': 'Medium',
      'spur': 'Medium',
      'spurious_copper': 'Low'
    };
    
    results.defects.forEach(d => {
      const conf = (d.confidence !== undefined && d.confidence !== null) 
        ? (d.confidence * 100).toFixed(1) 
        : 'N/A';
      const severity = severityMap[d.class_name] || 'Unknown';
      const centerX = d.center?.x || Math.round(d.bbox.x + d.bbox.width / 2);
      const centerY = d.center?.y || Math.round(d.bbox.y + d.bbox.height / 2);
      
      const row = [
        d.id,
        `"${d.class_name.replace(/_/g, ' ').toUpperCase()}"`,
        conf,
        centerX,
        centerY,
        d.bbox.x,
        d.bbox.y,
        d.bbox.width,
        d.bbox.height,
        d.area,
        severity
      ].join(',');
      csv += row + '\n';
    });
    
    // Add summary statistics at the end
    csv += '\n# Summary Statistics\n';
    csv += `# Total Records: ${results.defects.length}\n`;
    csv += `# Defect Types Found: ${Object.keys(results.frequency_analysis || {}).length}\n`;
    csv += `# Inspection Date: ${new Date().toLocaleDateString()}\n`;
    csv += `# Inspection Time: ${new Date().toLocaleTimeString()}\n`;
    

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    link.download = `CircuitGuard_Detailed_Log_${timestamp}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    setError('Failed to generate/download logs');
    console.error(err);
  } finally {
    setDownloadMenuOpen(false);
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
              <div className='result-buttons' style={{ position: 'relative' }} ref={downloadMenuRef}>
                <button
                  className="pdf-button"
                  onClick={() => setDownloadMenuOpen((s) => !s)}
                  disabled={isProcessing || isDownloadingPDF}
                  aria-haspopup="true"
                  aria-expanded={downloadMenuOpen}
                >
                  <FileText />
                  {isDownloadingPDF ? 'Generating PDF...' : 'Download'}
                  <span style={{ marginLeft: 8, fontSize: 12 }}>{downloadMenuOpen ? '▲' : '▼'}</span>
                </button>

                {downloadMenuOpen && (
  <div
    style={{
      position: 'absolute',
      right: 0,
      top: 'calc(100% + 8px)',
      background: 'white',
      borderRadius: 12,
      boxShadow: '0 10px 30px rgba(0,0,0,0.15)',
      zIndex: 50,
      minWidth: 240,
      overflow: 'hidden',
      border: '1px solid #e5e7eb'
    }}
  >
    <button
      onClick={downloadPDFReport}
      disabled={isProcessing || isDownloadingPDF}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        width: '100%',
        padding: '12px 16px',
        border: 'none',
        background: 'white',
        color: '#374151',
        textAlign: 'left',
        fontWeight: 600,
        fontSize: '0.9rem',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        borderBottom: '1px solid #f3f4f6'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = '#f9fafb';
        e.currentTarget.style.color = '#667eea';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'white';
        e.currentTarget.style.color = '#374151';
      }}
    >
      <FileText size={18} />
      Report (PDF)
    </button>

    <button
      onClick={downloadAnnotatedImage}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        width: '100%',
        padding: '12px 16px',
        border: 'none',
        background: 'white',
        color: '#374151',
        textAlign: 'left',
        fontWeight: 600,
        fontSize: '0.9rem',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        borderBottom: '1px solid #f3f4f6'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = '#f9fafb';
        e.currentTarget.style.color = '#667eea';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'white';
        e.currentTarget.style.color = '#374151';
      }}
    >
      <ImageIcon size={18} />
      Annotated Image
    </button>

    <button
      onClick={downloadLogsCSV}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        width: '100%',
        padding: '12px 16px',
        border: 'none',
        background: 'white',
        color: '#374151',
        textAlign: 'left',
        fontWeight: 600,
        fontSize: '0.9rem',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif',
        cursor: 'pointer',
        transition: 'all 0.2s ease'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = '#f9fafb';
        e.currentTarget.style.color = '#667eea';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'white';
        e.currentTarget.style.color = '#374151';
      }}
    >
      <TrendingUp size={18} />
      Logs (CSV)
    </button>
  </div>
)}

                <button className="reset-button" onClick={resetForm}>
                  ⟳
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