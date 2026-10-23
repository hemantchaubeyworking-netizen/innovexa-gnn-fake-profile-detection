import React from 'react';
import { Shield, Activity, BarChart2, Share2, History, Info, CheckCircle, AlertTriangle } from 'lucide-react';

export default function Navbar({ activePage, setActivePage, backendOnline, selectedModel, setSelectedModel, availableModels }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'detection', label: 'Profile Detection', icon: Shield },
    { id: 'performance', label: 'Model Performance', icon: BarChart2 },
    { id: 'network', label: 'Network Analysis', icon: Share2 },
    { id: 'history', label: 'Detection History', icon: History },
    { id: 'info', label: 'Research & Model Info', icon: Info }
  ];

  return (
    <header style={{
      position: 'sticky',
      top: 0,
      zIndex: 100,
      background: 'rgba(7, 10, 18, 0.88)',
      backdropFilter: 'blur(16px)',
      WebkitBackdropFilter: 'blur(16px)',
      borderBottom: '1px solid var(--border-subtle)',
      padding: '0 24px'
    }}>
      <div style={{
        maxWidth: 1400,
        margin: '0 auto',
        height: 72,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        {/* Brand identity */}
        <div 
          onClick={() => setActivePage('dashboard')}
          style={{ display: 'flex', alignItems: 'center', gap: 14, cursor: 'pointer' }}
        >
          <div style={{
            width: 42,
            height: 42,
            borderRadius: 10,
            background: 'linear-gradient(135deg, rgba(56, 189, 248, 0.25), rgba(168, 85, 247, 0.3))',
            border: '1px solid var(--border-focus)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 16px rgba(56, 189, 248, 0.3)'
          }}>
            <Shield size={24} color="#38bdf8" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{
                fontFamily: 'var(--font-heading)',
                fontSize: '1.35rem',
                fontWeight: 900,
                letterSpacing: '-0.03em',
                background: 'linear-gradient(135deg, #ffffff 40%, #38bdf8 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}>
                INNOVEXA
              </span>
              <span style={{
                fontSize: '0.65rem',
                padding: '2px 6px',
                borderRadius: 4,
                background: 'rgba(56, 189, 248, 0.15)',
                color: '#38bdf8',
                border: '1px solid rgba(56, 189, 248, 0.3)',
                fontWeight: 700,
                letterSpacing: '0.05em'
              }}>
                GNN v1.0
              </span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
              Fake Instagram Profile Detection Framework
            </p>
          </div>
        </div>

        {/* Navigation tabs */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {navItems.map(item => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '8px 14px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.85rem',
                  fontWeight: isActive ? 600 : 500,
                  color: isActive ? '#38bdf8' : 'var(--text-secondary)',
                  background: isActive ? 'rgba(56, 189, 248, 0.1)' : 'transparent',
                  border: isActive ? '1px solid rgba(56, 189, 248, 0.25)' : '1px solid transparent',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <Icon size={16} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Header Right: Active Model Selector & Backend Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          {/* Active Model Selector */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '4px 10px',
            borderRadius: 'var(--radius-sm)',
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid var(--border-subtle)'
          }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Model:</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                background: 'transparent',
                color: 'var(--accent-cyan)',
                border: 'none',
                fontFamily: 'var(--font-heading)',
                fontSize: '0.85rem',
                fontWeight: 600,
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              {(availableModels || [
                { id: 'GraphSAGE', name: 'GraphSAGE (GNN)' },
                { id: 'GCN', name: 'GCN (GNN)' },
                { id: 'GAT', name: 'GAT (GNN)' },
                { id: 'Random Forest', name: 'Random Forest' },
                { id: 'MLP', name: 'MLP' },
                { id: 'Support Vector Machine', name: 'SVM' },
                { id: 'Logistic Regression', name: 'Logistic Regression' }
              ]).map(m => (
                <option key={m.id} value={m.id} style={{ background: '#0f172a', color: '#fff' }}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          {/* Backend Status indicator */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '5px 10px',
            borderRadius: 'var(--radius-full)',
            background: backendOnline ? 'rgba(16, 185, 129, 0.12)' : 'rgba(245, 158, 11, 0.12)',
            border: `1px solid ${backendOnline ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
            fontSize: '0.75rem',
            fontWeight: 600,
            color: backendOnline ? '#34d399' : '#fbbf24'
          }}>
            <div style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: backendOnline ? '#10b981' : '#f59e0b',
              boxShadow: backendOnline ? '0 0 8px #10b981' : '0 0 8px #f59e0b'
            }} />
            <span>{backendOnline ? 'API Connected' : 'Checking API...'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
