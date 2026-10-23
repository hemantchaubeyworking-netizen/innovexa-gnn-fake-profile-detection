import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import ProfileDetection from './pages/ProfileDetection';
import ModelPerformance from './pages/ModelPerformance';
import NetworkAnalysis from './pages/NetworkAnalysis';
import DetectionHistory from './pages/DetectionHistory';
import ModelInfo from './pages/ModelInfo';
import { fetchHealth, fetchDashboard, fetchMetrics, fetchNetwork, fetchRobustness, fetchModels } from './services/api';
import { Shield, AlertCircle } from 'lucide-react';

export default function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [backendOnline, setBackendOnline] = useState(false);
  const [selectedModel, setSelectedModel] = useState('GraphSAGE');
  const [availableModels, setAvailableModels] = useState([]);
  
  const [dashboardData, setDashboardData] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [networkData, setNetworkData] = useState(null);
  const [robustnessData, setRobustnessData] = useState(null);
  const [loading, setLoading] = useState(true);

  // Poll backend health and preload data
  useEffect(() => {
    async function loadInitialData() {
      try {
        await fetchHealth();
        setBackendOnline(true);
        
        // Fetch core datasets concurrently
        const [dashRes, metricsRes, netRes, robRes, modelsRes] = await Promise.allSettled([
          fetchDashboard(),
          fetchMetrics(),
          fetchNetwork(),
          fetchRobustness(),
          fetchModels()
        ]);

        if (dashRes.status === 'fulfilled') setDashboardData(dashRes.value);
        if (metricsRes.status === 'fulfilled') setMetricsData(metricsRes.value);
        if (netRes.status === 'fulfilled') setNetworkData(netRes.value);
        if (robRes.status === 'fulfilled') setRobustnessData(robRes.value);
        if (modelsRes.status === 'fulfilled') {
          setAvailableModels(modelsRes.value.models || []);
          if (modelsRes.value.default_model) setSelectedModel(modelsRes.value.default_model);
        }
      } catch (err) {
        console.warn('Backend currently unreachable, using offline fallback states:', err);
        setBackendOnline(false);
      } finally {
        setLoading(false);
      }
    }

    loadInitialData();
    const interval = setInterval(loadInitialData, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Header Navbar */}
      <Navbar
        activePage={activePage}
        setActivePage={setActivePage}
        backendOnline={backendOnline}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
        availableModels={availableModels}
      />

      {/* Main Content Area */}
      <main style={{ flex: 1, maxWidth: 1400, width: '100%', margin: '0 auto', padding: '28px 24px' }}>
        {activePage === 'dashboard' && (
          <Dashboard
            dashboardData={dashboardData}
            setActivePage={setActivePage}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
          />
        )}

        {activePage === 'detection' && (
          <ProfileDetection
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            availableModels={availableModels}
          />
        )}

        {activePage === 'performance' && (
          <ModelPerformance
            metricsData={metricsData}
            robustnessData={robustnessData}
          />
        )}

        {activePage === 'network' && (
          <NetworkAnalysis
            networkData={networkData}
          />
        )}

        {activePage === 'history' && (
          <DetectionHistory />
        )}

        {activePage === 'info' && (
          <ModelInfo />
        )}
      </main>

      {/* Footer */}
      <footer style={{
        background: 'rgba(7, 10, 18, 0.95)',
        borderTop: '1px solid var(--border-subtle)',
        padding: '24px',
        marginTop: 40,
        fontSize: '0.8rem',
        color: 'var(--text-muted)'
      }}>
        <div style={{
          maxWidth: 1400,
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 14
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Shield size={16} color="#38bdf8" />
            <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
              INNOVEXA &mdash; Graph Neural Network Research Framework
            </span>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span>Trained on exact 5,000-profile dataset &bull; 7 Architectures Evaluated &bull; SQLite Persistence</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
