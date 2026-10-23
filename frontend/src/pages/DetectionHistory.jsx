import React, { useState, useEffect } from 'react';
import { History, Trash2, Search, Filter, RefreshCw, Download, ShieldCheck, UserX } from 'lucide-react';
import { fetchHistory, clearHistoryApi } from '../services/api';

export default function DetectionHistory() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filterClass, setFilterClass] = useState('ALL');

  const loadHistoryData = async () => {
    setLoading(true);
    try {
      const res = await fetchHistory(100);
      setHistory(res.history || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistoryData();
  }, []);

  const handleClearHistory = async () => {
    if (window.confirm("Are you sure you want to clear the prediction history stored in SQLite?")) {
      try {
        await clearHistoryApi();
        setHistory([]);
      } catch (e) {
        alert("Failed to clear history: " + e.message);
      }
    }
  };

  const filtered = history.filter(item => {
    const matchesSearch = item.input_summary.toLowerCase().includes(search.toLowerCase()) ||
                          item.model_name.toLowerCase().includes(search.toLowerCase()) ||
                          String(item.id).includes(search);
    const matchesClass = filterClass === 'ALL' || item.predicted_class === filterClass;
    return matchesSearch && matchesClass;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Title & Controls */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 14 }}>
        <div>
          <h1 style={{ fontSize: '1.6rem', margin: 0 }}>Inference Audit History</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: 4 }}>
            Historical record of profile predictions persisted in backend SQLite database.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            onClick={loadHistoryData}
            className="btn btn-secondary"
            style={{ fontSize: '0.85rem' }}
          >
            <RefreshCw size={15} className={loading ? "animate-spin" : ""} />
            <span>Refresh</span>
          </button>
          <button
            onClick={handleClearHistory}
            className="btn btn-danger"
            style={{ fontSize: '0.85rem' }}
          >
            <Trash2 size={15} />
            <span>Clear History</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          background: 'rgba(15, 23, 42, 0.8)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 8,
          padding: '6px 12px',
          flex: 1,
          minWidth: 260
        }}>
          <Search size={16} color="var(--text-muted)" />
          <input
            type="text"
            placeholder="Search by ID, model, or account features..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: '#fff',
              fontSize: '0.85rem',
              width: '100%'
            }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Filter size={16} color="var(--text-muted)" />
          <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Class:</span>
          {['ALL', 'Fake', 'Genuine'].map(cls => (
            <button
              key={cls}
              onClick={() => setFilterClass(cls)}
              style={{
                background: filterClass === cls ? 'rgba(56, 189, 248, 0.2)' : 'rgba(30, 41, 59, 0.6)',
                color: filterClass === cls ? '#38bdf8' : 'var(--text-secondary)',
                border: filterClass === cls ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid var(--border-subtle)',
                borderRadius: 6,
                padding: '4px 12px',
                fontSize: '0.78rem',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              {cls}
            </button>
          ))}
        </div>
      </div>

      {/* History Table */}
      <div className="glass-panel" style={{ padding: 24 }}>
        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-muted)' }}>
            <RefreshCw size={24} className="animate-spin" style={{ margin: '0 auto 10px' }} />
            <p>Loading historical predictions...</p>
          </div>
        ) : filtered.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '50px 0', color: 'var(--text-muted)' }}>
            <History size={36} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
            <p style={{ fontSize: '0.95rem' }}>No prediction records match your query.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '12px 14px' }}>Record ID</th>
                  <th style={{ padding: '12px 14px' }}>Timestamp (UTC)</th>
                  <th style={{ padding: '12px 14px' }}>Model Used</th>
                  <th style={{ padding: '12px 14px' }}>Predicted Class</th>
                  <th style={{ padding: '12px 14px' }}>Fake Probability</th>
                  <th style={{ padding: '12px 14px' }}>Risk Category</th>
                  <th style={{ padding: '12px 14px' }}>Profile Attributes Summary</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(item => (
                  <tr key={item.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                      #{item.id}
                    </td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-secondary)' }}>
                      {new Date(item.timestamp).toLocaleString()}
                    </td>
                    <td style={{ padding: '12px 14px', fontWeight: 600, color: '#fff' }}>
                      {item.model_name}
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <span className={`badge ${item.predicted_class === 'Fake' ? 'badge-fake' : 'badge-genuine'}`}>
                        {item.predicted_class}
                      </span>
                    </td>
                    <td style={{
                      padding: '12px 14px',
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 700,
                      color: item.fake_probability >= 0.5 ? '#f43f5e' : '#34d399'
                    }}>
                      {(item.fake_probability * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <span className={`badge badge-${item.risk_category.toLowerCase().split(' ')[0]}`}>
                        {item.risk_category}
                      </span>
                    </td>
                    <td style={{ padding: '12px 14px', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                      {item.input_summary}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
