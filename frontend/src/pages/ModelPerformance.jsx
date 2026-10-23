import React, { useState } from 'react';
import { BarChart2, ShieldAlert, Award, TrendingDown, Layers, FileText } from 'lucide-react';
import ConfusionMatrixChart from '../components/ConfusionMatrixChart';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

export default function ModelPerformance({ metricsData, robustnessData }) {
  const metrics = metricsData?.metrics || [];
  const rocCurves = metricsData?.roc_curves || {};
  const prCurves = metricsData?.pr_curves || {};
  const lossHistory = metricsData?.loss_history || {};
  const featureMasking = robustnessData?.feature_masking || {};
  const edgeInjection = robustnessData?.edge_injection || {};

  const [activeMatrixModel, setActiveMatrixModel] = useState('GraphSAGE');
  const [activeLossModel, setActiveLossModel] = useState('GraphSAGE');

  const selectedMatrix = metrics.find(m => m.model_name === activeMatrixModel)?.confusion_matrix;

  // Prepare multi-model ROC Curve comparison dataset
  // Format: [{fpr: 0.1, GraphSAGE: 0.95, GCN: 0.94, RF: 0.98}, ...]
  const rocPoints = [];
  const sampleRoc = rocCurves['GraphSAGE'] || [];
  sampleRoc.forEach((pt, i) => {
    const row = { fpr: (pt.fpr * 100).toFixed(1) };
    Object.keys(rocCurves).forEach(mName => {
      const curve = rocCurves[mName];
      if (curve && curve[i]) {
        row[mName] = (curve[i].tpr * 100).toFixed(1);
      }
    });
    rocPoints.push(row);
  });

  // Prepare PR Curve comparison dataset
  const prPoints = [];
  const samplePr = prCurves['GraphSAGE'] || [];
  samplePr.forEach((pt, i) => {
    const row = { recall: (pt.recall * 100).toFixed(1) };
    Object.keys(prCurves).forEach(mName => {
      const curve = prCurves[mName];
      if (curve && curve[i]) {
        row[mName] = (curve[i].precision * 100).toFixed(1);
      }
    });
    prPoints.push(row);
  });

  // Prepare Feature Masking Robustness dataset
  const maskRates = [0.0, 0.1, 0.2, 0.3, 0.5];
  const maskChartData = maskRates.map(rate => {
    const row = { rate: `${(rate * 100).toFixed(0)}%` };
    Object.keys(featureMasking).forEach(mName => {
      const entry = featureMasking[mName]?.find(e => Math.abs(e.mask_rate - rate) < 0.01);
      if (entry) {
        row[mName] = (entry.f1_score * 100).toFixed(1);
      }
    });
    return row;
  });

  // Prepare Edge Injection Robustness dataset
  const injRatios = [0.0, 0.05, 0.10, 0.20, 0.30];
  const injChartData = injRatios.map(ratio => {
    const row = { ratio: `${(ratio * 100).toFixed(0)}%` };
    Object.keys(edgeInjection).forEach(mName => {
      const entry = edgeInjection[mName]?.find(e => Math.abs(e.injection_ratio - ratio) < 0.01);
      if (entry) {
        row[mName] = (entry.f1_score * 100).toFixed(1);
      }
    });
    return row;
  });

  // Model Colors map
  const modelColors = {
    'GraphSAGE': '#38bdf8',
    'GCN': '#818cf8',
    'GAT': '#a855f7',
    'Random Forest': '#10b981',
    'MLP': '#f59e0b',
    'Support Vector Machine': '#ec4899',
    'Logistic Regression': '#94a3b8'
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
      {/* Title */}
      <div>
        <h1 style={{ fontSize: '1.6rem', margin: 0 }}>Model Evaluation & Empirical Comparison</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: 4 }}>
          Held-out test results (15% = 750 samples), threshold calibration, ROC/PR curves, and adversarial robustness.
        </p>
      </div>

      {/* Comprehensive Results Table (Table III in Research Paper) */}
      <div className="glass-panel" style={{ padding: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <div>
            <h2 style={{ fontSize: '1.2rem', margin: 0 }}>Model Comparison Matrix (Table III)</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
              All models evaluated under strictly identical 70/15/15 stratified partitions with validation-tuned decision thresholds.
            </p>
          </div>
          <span className="badge badge-genuine" style={{ gap: 4 }}>
            Zero Leakage Enforced
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                <th style={{ padding: '12px 14px' }}>Model Architecture</th>
                <th style={{ padding: '12px 14px' }}>Type</th>
                <th style={{ padding: '12px 14px' }}>Accuracy</th>
                <th style={{ padding: '12px 14px' }}>Precision</th>
                <th style={{ padding: '12px 14px' }}>Recall</th>
                <th style={{ padding: '12px 14px' }}>F1-Score</th>
                <th style={{ padding: '12px 14px' }}>ROC-AUC</th>
                <th style={{ padding: '12px 14px' }}>PR-AUC</th>
                <th style={{ padding: '12px 14px' }}>Threshold (τ)</th>
                <th style={{ padding: '12px 14px' }}>Train Time</th>
                <th style={{ padding: '12px 14px' }}>Latency</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map(m => {
                const isGNN = ['GCN', 'GraphSAGE', 'GAT'].includes(m.model_name);
                return (
                  <tr key={m.model_name} style={{
                    borderBottom: '1px solid rgba(255,255,255,0.04)',
                    background: m.model_name === 'GraphSAGE' ? 'rgba(56, 189, 248, 0.05)' : 'transparent'
                  }}>
                    <td style={{ padding: '12px 14px', fontWeight: 700, color: '#fff', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%', background: modelColors[m.model_name] || '#38bdf8' }} />
                      <span>{m.model_name}</span>
                    </td>
                    <td style={{ padding: '12px 14px' }}>
                      <span style={{
                        fontSize: '0.72rem',
                        padding: '2px 6px',
                        borderRadius: 4,
                        background: isGNN ? 'rgba(56, 189, 248, 0.15)' : 'rgba(255,255,255,0.08)',
                        color: isGNN ? '#38bdf8' : 'var(--text-secondary)'
                      }}>
                        {isGNN ? 'GNN Architecture' : 'Baseline'}
                      </span>
                    </td>
                    <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)' }}>{(m.accuracy * 100).toFixed(2)}%</td>
                    <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)' }}>{(m.precision * 100).toFixed(2)}%</td>
                    <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)' }}>{(m.recall * 100).toFixed(2)}%</td>
                    <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#38bdf8' }}>
                      {(m.f1_score * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#a855f7' }}>
                      {(m.roc_auc * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)' }}>{(m.pr_auc * 100).toFixed(2)}%</td>
                    <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                      {m.threshold.toFixed(2)}
                    </td>
                    <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)' }}>{m.train_time_sec}s</td>
                    <td style={{ padding: '12px 14px', fontFamily: 'var(--font-mono)' }}>{m.inference_latency_ms}ms</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ROC Curves and PR Curves in Two Columns */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 24 }}>
        {/* ROC Curve Chart */}
        <div className="glass-panel" style={{ padding: 22 }}>
          <h2 style={{ fontSize: '1.15rem', marginBottom: 4 }}>Receiver Operating Characteristic (ROC)</h2>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 14 }}>
            True Positive Rate vs. False Positive Rate across all decision thresholds.
          </p>
          <div style={{ height: 300, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rocPoints} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="fpr" tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'False Positive Rate (%)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'True Positive Rate (%)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: 'var(--border-subtle)', borderRadius: 8, fontSize: '0.75rem' }} />
                <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
                <Line type="monotone" dataKey="GraphSAGE" stroke="#38bdf8" strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="GCN" stroke="#818cf8" strokeWidth={1.8} dot={false} />
                <Line type="monotone" dataKey="GAT" stroke="#a855f7" strokeWidth={1.8} dot={false} />
                <Line type="monotone" dataKey="Random Forest" stroke="#10b981" strokeWidth={1.8} dot={false} />
                <Line type="monotone" dataKey="MLP" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* PR Curve Chart */}
        <div className="glass-panel" style={{ padding: 22 }}>
          <h2 style={{ fontSize: '1.15rem', marginBottom: 4 }}>Precision-Recall Curves</h2>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 14 }}>
            Precision vs. Recall trade-off evaluating detection quality at varying thresholds.
          </p>
          <div style={{ height: 300, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={prPoints} margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="recall" tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Recall (%)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                <YAxis domain={[90, 100]} tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Precision (%)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: 'var(--border-subtle)', borderRadius: 8, fontSize: '0.75rem' }} />
                <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
                <Line type="monotone" dataKey="GraphSAGE" stroke="#38bdf8" strokeWidth={2.5} dot={false} />
                <Line type="monotone" dataKey="GCN" stroke="#818cf8" strokeWidth={1.8} dot={false} />
                <Line type="monotone" dataKey="GAT" stroke="#a855f7" strokeWidth={1.8} dot={false} />
                <Line type="monotone" dataKey="Random Forest" stroke="#10b981" strokeWidth={1.8} dot={false} />
                <Line type="monotone" dataKey="MLP" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Confusion Matrix & Training Loss History */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        {/* Interactive Confusion Matrix */}
        <div className="glass-panel" style={{ padding: 22 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <div>
              <h2 style={{ fontSize: '1.15rem', margin: 0 }}>Confusion Matrix Visualizer</h2>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
                750 held-out test predictions breakdown
              </p>
            </div>
            <select
              value={activeMatrixModel}
              onChange={e => setActiveMatrixModel(e.target.value)}
              style={{
                background: 'rgba(15, 23, 42, 0.8)',
                color: 'var(--accent-cyan)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 6,
                padding: '4px 10px',
                fontSize: '0.8rem',
                fontWeight: 600,
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              {metrics.map(m => (
                <option key={m.model_name} value={m.model_name} style={{ background: '#0f172a' }}>
                  {m.model_name}
                </option>
              ))}
            </select>
          </div>

          <ConfusionMatrixChart cmData={selectedMatrix} modelName={activeMatrixModel} />
        </div>

        {/* Training Loss Convergence History */}
        <div className="glass-panel" style={{ padding: 22 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
            <div>
              <h2 style={{ fontSize: '1.15rem', margin: 0 }}>Training Loss History</h2>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
                Train vs. validation loss curve per epoch
              </p>
            </div>
            <select
              value={activeLossModel}
              onChange={e => setActiveLossModel(e.target.value)}
              style={{
                background: 'rgba(15, 23, 42, 0.8)',
                color: 'var(--accent-cyan)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 6,
                padding: '4px 10px',
                fontSize: '0.8rem',
                fontWeight: 600,
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              {['GraphSAGE', 'GCN', 'GAT', 'MLP'].map(m => (
                <option key={m} value={m} style={{ background: '#0f172a' }}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          <div style={{ height: 260, width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lossHistory[activeLossModel] || []} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="epoch" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: 'var(--border-subtle)', borderRadius: 8, fontSize: '0.75rem' }} />
                <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
                <Line type="monotone" dataKey="train_loss" name="Train BCE Loss" stroke="#38bdf8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="val_loss" name="Validation BCE Loss" stroke="#f43f5e" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Adversarial Robustness Section */}
      <div className="glass-panel" style={{ padding: 24, borderTop: '3px solid var(--accent-amber)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
          <ShieldAlert size={22} color="#f59e0b" />
          <div>
            <h2 style={{ fontSize: '1.2rem', margin: 0 }}>Adversarial Robustness Evaluation</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
              Testing model resilience against Feature Masking (attribute camouflage) and Edge Injection (relational camouflage).
            </p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 24 }}>
          {/* Feature Masking Chart */}
          <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: 16, borderRadius: 8 }}>
            <h3 style={{ fontSize: '0.95rem', marginBottom: 4 }}>1. Feature Masking Degradation</h3>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 12 }}>
              F1-score retention when 10% to 50% of profile attributes are randomly obscured.
            </p>
            <div style={{ height: 220, width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={maskChartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="rate" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis domain={[85, 100]} tick={{ fill: '#94a3b8', fontSize: 10 }} unit="%" />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: 'var(--border-subtle)', borderRadius: 8, fontSize: '0.75rem' }} />
                  <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
                  <Line type="monotone" dataKey="GraphSAGE" stroke="#38bdf8" strokeWidth={2} />
                  <Line type="monotone" dataKey="GCN" stroke="#818cf8" strokeWidth={2} />
                  <Line type="monotone" dataKey="Random Forest" stroke="#10b981" strokeWidth={1.5} />
                  <Line type="monotone" dataKey="MLP" stroke="#f59e0b" strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Edge Injection Chart */}
          <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: 16, borderRadius: 8 }}>
            <h3 style={{ fontSize: '0.95rem', marginBottom: 4 }}>2. Edge Injection Perturbation</h3>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 12 }}>
              GNN message passing stability when 5% to 30% random fake edges are injected into the graph.
            </p>
            <div style={{ height: 220, width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={injChartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="ratio" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis domain={[90, 100]} tick={{ fill: '#94a3b8', fontSize: 10 }} unit="%" />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: 'var(--border-subtle)', borderRadius: 8, fontSize: '0.75rem' }} />
                  <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
                  <Line type="monotone" dataKey="GraphSAGE" stroke="#38bdf8" strokeWidth={2} />
                  <Line type="monotone" dataKey="GCN" stroke="#818cf8" strokeWidth={2} />
                  <Line type="monotone" dataKey="GAT" stroke="#a855f7" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
