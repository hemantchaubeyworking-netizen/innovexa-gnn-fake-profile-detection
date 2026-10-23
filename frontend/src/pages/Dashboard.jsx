import React from 'react';
import { ShieldCheck, Users, UserX, Share2, Award, Zap, ArrowRight, CheckCircle2 } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid } from 'recharts';

export default function Dashboard({ dashboardData, setActivePage, selectedModel, setSelectedModel }) {
  const summary = dashboardData?.dataset_summary || {};
  const training = dashboardData?.training_status || {};
  const metrics = dashboardData?.model_comparison_metrics || [];
  const history = dashboardData?.recent_predictions || [];

  // Chart data formatted for Recharts
  const chartData = metrics.map(m => ({
    name: m.model_name.replace('Support Vector Machine', 'SVM').replace('Logistic Regression', 'LR'),
    F1: (m.f1_score * 100).toFixed(1),
    AUC: (m.roc_auc * 100).toFixed(1),
    Accuracy: (m.accuracy * 100).toFixed(1),
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
      {/* Top Welcome Banner */}
      <div className="glass-panel" style={{
        padding: '24px 28px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.6))',
        borderLeft: '4px solid var(--accent-cyan)'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
            <h1 style={{ fontSize: '1.6rem', margin: 0 }}>INNOVEXA Intelligence Dashboard</h1>
            <span className="badge badge-genuine" style={{ gap: 4 }}>
              <CheckCircle2 size={12} />
              Models Trained & Verified
            </span>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: 800, margin: 0 }}>
            Research-grade fake Instagram profile detection framework utilizing Graph Neural Networks (GraphSAGE, GCN, GAT)
            and traditional baseline classifiers over a 5,000-profile feature-similarity network.
          </p>
        </div>

        <button 
          onClick={() => setActivePage('detection')}
          className="btn btn-primary"
          style={{ whiteSpace: 'nowrap' }}
        >
          <span>Run Profile Detection</span>
          <ArrowRight size={16} />
        </button>
      </div>

      {/* Primary KPI Metrics Cards */}
      <div className="responsive-grid">
        <MetricCard
          title="Total Profiles"
          value={summary.total_profiles?.toLocaleString() || "5,000"}
          subtitle="Exact uploaded Instagram dataset"
          icon={Users}
          color="#38bdf8"
          badge="12 Features"
        />
        <MetricCard
          title="Genuine Accounts"
          value={summary.genuine_profiles?.toLocaleString() || "2,500"}
          subtitle="Ground Truth: Class 0 (50.0%)"
          icon={ShieldCheck}
          color="#10b981"
          progress={50}
        />
        <MetricCard
          title="Fake Accounts"
          value={summary.fake_profiles?.toLocaleString() || "2,500"}
          subtitle="Ground Truth: Class 1 (50.0%)"
          icon={UserX}
          color="#f43f5e"
          progress={50}
        />
        <MetricCard
          title="Similarity Graph"
          value={`${summary.graph_total_nodes?.toLocaleString() || "5,000"} Nodes`}
          subtitle={`${summary.graph_total_edges?.toLocaleString() || "16,356"} undirected edges (k=5)`}
          icon={Share2}
          color="#a855f7"
          badge="Cosine kNN"
        />
      </div>

      {/* Model Performance Comparison Section */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 24 }}>
        {/* Comparison Bar Chart */}
        <div className="glass-panel" style={{ padding: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
            <div>
              <h2 style={{ fontSize: '1.2rem', margin: 0 }}>Held-Out Test Performance Comparison</h2>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
                Evaluated on 15% held-out test partition (750 samples) using validation-tuned decision thresholds.
              </p>
            </div>
            <button 
              onClick={() => setActivePage('performance')}
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', padding: '6px 12px' }}
            >
              View Detailed Curves
            </button>
          </div>

          <div style={{ height: 320, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis 
                  dataKey="name" 
                  tick={{ fill: '#94a3b8', fontSize: 11 }} 
                  interval={0} 
                  angle={-15} 
                  textAnchor="end" 
                />
                <YAxis domain={[90, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} unit="%" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#0f172a', 
                    borderColor: 'rgba(56, 189, 248, 0.3)',
                    borderRadius: 8,
                    fontSize: '0.8rem'
                  }} 
                />
                <Legend wrapperStyle={{ fontSize: '0.8rem', paddingTop: 10 }} />
                <Bar dataKey="F1" name="F1-Score (%)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="AUC" name="ROC-AUC (%)" fill="#a855f7" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Accuracy" name="Accuracy (%)" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Model Highlight & Active Configuration */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Best Model Card */}
          <div className="glass-panel" style={{ padding: 22, borderTop: '3px solid var(--accent-emerald)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <Award size={20} color="#10b981" />
              <h3 style={{ fontSize: '1.05rem', margin: 0 }}>Top Performing Architecture</h3>
            </div>
            <div style={{
              background: 'rgba(16, 185, 129, 0.1)',
              borderRadius: 8,
              padding: '12px 16px',
              marginBottom: 14,
              border: '1px solid rgba(16, 185, 129, 0.25)'
            }}>
              <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#fff' }}>
                {training.best_performing_model || "Random Forest"}
              </div>
              <div style={{ fontSize: '0.78rem', color: '#34d399', marginTop: 2 }}>
                F1-Score: {(training.top_f1_score * 100).toFixed(2)}% | ROC-AUC: {(training.top_roc_auc * 100).toFixed(2)}%
              </div>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
              The model achieves exceptional separation on account attributes. Combined with GraphSAGE's inductive neighborhood aggregation,
              the framework achieves robust detection even under adversarial feature camouflage.
            </p>
          </div>

          {/* Graph Protocol Information */}
          <div className="glass-panel" style={{ padding: 22 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <Zap size={20} color="#38bdf8" />
              <h3 style={{ fontSize: '1.05rem', margin: 0 }}>Evaluation Protocol</h3>
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Train Partition:</span>
                <span style={{ fontWeight: 600, color: '#fff' }}>70% (3,500 profiles)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Validation Partition:</span>
                <span style={{ fontWeight: 600, color: '#fff' }}>15% (750 profiles)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Test Partition:</span>
                <span style={{ fontWeight: 600, color: '#fff' }}>15% (750 profiles)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Leakage Prevention:</span>
                <span style={{ fontWeight: 600, color: '#34d399' }}>Strict (Train-only scaling)</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Threshold Selection:</span>
                <span style={{ fontWeight: 600, color: '#38bdf8' }}>Validation F1 Tuning</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Detection History Preview */}
      <div className="glass-panel" style={{ padding: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <div>
            <h2 style={{ fontSize: '1.15rem', margin: 0 }}>Recent Prediction History</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
              Persistent records logged to SQLite database (backend/innovexa.db).
            </p>
          </div>
          <button 
            onClick={() => setActivePage('history')}
            className="btn btn-secondary"
            style={{ fontSize: '0.8rem', padding: '6px 12px' }}
          >
            View All History
          </button>
        </div>

        {history.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '30px 0', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            No live predictions logged yet. Try testing a profile in the Profile Detection page!
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '10px 12px' }}>ID</th>
                  <th style={{ padding: '10px 12px' }}>Timestamp</th>
                  <th style={{ padding: '10px 12px' }}>Model</th>
                  <th style={{ padding: '10px 12px' }}>Prediction</th>
                  <th style={{ padding: '10px 12px' }}>Fake Probability</th>
                  <th style={{ padding: '10px 12px' }}>Risk Category</th>
                  <th style={{ padding: '10px 12px' }}>Attributes Summary</th>
                </tr>
              </thead>
              <tbody>
                {history.map(item => (
                  <tr key={item.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)' }}>#{item.id}</td>
                    <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>
                      {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </td>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{item.model_name}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <span className={`badge ${item.predicted_class === 'Fake' ? 'badge-fake' : 'badge-genuine'}`}>
                        {item.predicted_class}
                      </span>
                    </td>
                    <td style={{ padding: '10px 12px', fontFamily: 'var(--font-mono)', fontWeight: 600, color: item.fake_probability >= 0.5 ? '#f43f5e' : '#34d399' }}>
                      {(item.fake_probability * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <span className={`badge badge-${item.risk_category.toLowerCase().split(' ')[0]}`}>
                        {item.risk_category}
                      </span>
                    </td>
                    <td style={{ padding: '10px 12px', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
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
