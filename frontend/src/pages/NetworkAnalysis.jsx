import React, { useState } from 'react';
import { Share2, AlertTriangle, Users, Layers, ShieldCheck, Info } from 'lucide-react';
import GraphVisualization from '../components/GraphVisualization';

export default function NetworkAnalysis({ networkData }) {
  const [selectedNode, setSelectedNode] = useState(null);

  const totalNodes = networkData?.total_nodes_in_full_graph || 5000;
  const totalEdges = networkData?.total_edges_in_full_graph || 16356;
  const highRiskCount = networkData?.high_risk_nodes_count || 2402;
  const clusters = networkData?.clusters || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
      {/* Page Title */}
      <div>
        <h1 style={{ fontSize: '1.6rem', margin: 0 }}>Network Topology & Suspicious Clusters</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: 4 }}>
          Graph-based relational analysis identifying attribute-similarity clusters and coordinated high-risk patterns.
        </p>
      </div>

      {/* Mandatory Scientific Limitation Banner */}
      <div className="glass-panel" style={{
        padding: '16px 20px',
        borderLeft: '4px solid var(--accent-amber)',
        background: 'rgba(245, 158, 11, 0.08)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: 12
      }}>
        <AlertTriangle size={20} color="#fbbf24" style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <span style={{ fontWeight: 700, fontSize: '0.9rem', color: '#fbbf24' }}>
            Critical Research Graph Definition & Limitation
          </span>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '4px 0 0 0', lineHeight: 1.5 }}>
            This network is an <strong>unsupervised k-nearest-neighbor feature-similarity graph (k=5, cosine metric)</strong> constructed
            over normalized profile attributes. It does <strong>NOT</strong> represent confirmed social interactions, follower-following links,
            or communication records. Clusters signify accounts exhibiting similar behavioral metrics, not an organized criminal syndicate.
          </p>
        </div>
      </div>

      {/* Network Stats Cards */}
      <div className="responsive-grid">
        <div className="glass-panel" style={{ padding: 18 }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 4 }}>Total Nodes (Profiles)</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#fff' }}>{totalNodes.toLocaleString()}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>Full dataset coverage</div>
        </div>
        <div className="glass-panel" style={{ padding: 18 }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 4 }}>Similarity Edges</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#38bdf8' }}>{totalEdges.toLocaleString()}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>Cosine attribute proximity</div>
        </div>
        <div className="glass-panel" style={{ padding: 18 }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 4 }}>High-Risk Accounts</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#f43f5e' }}>{highRiskCount.toLocaleString()}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>Estimated fake probability ≥ 70%</div>
        </div>
        <div className="glass-panel" style={{ padding: 18 }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 4 }}>Detected Clusters</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#a855f7' }}>{clusters.length} Groups</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>Greedy modularity communities</div>
        </div>
      </div>

      {/* Interactive Graph Canvas */}
      <div className="glass-panel" style={{ padding: 22 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <div>
            <h2 style={{ fontSize: '1.2rem', margin: 0 }}>Interactive Subgraph Visualization (150 Nodes)</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
              Drag to pan, use wheel or controls to zoom. Click a node to view attributes and neighborhood risk.
            </p>
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Rendered via HTML5 High-Performance Canvas
          </span>
        </div>

        <GraphVisualization
          networkData={networkData}
          onNodeSelect={setSelectedNode}
          selectedNodeId={selectedNode?.id}
        />
      </div>

      {/* Candidate Suspicious Clusters Table */}
      <div className="glass-panel" style={{ padding: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
          <Layers size={20} color="#38bdf8" />
          <div>
            <h2 style={{ fontSize: '1.15rem', margin: 0 }}>Candidate Suspicious Clusters</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
              Community subgraphs with concentrated high-risk accounts identified by modularity optimization.
            </p>
          </div>
        </div>

        {clusters.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No dense clusters detected.</p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '10px 14px' }}>Cluster ID</th>
                  <th style={{ padding: '10px 14px' }}>Accounts Count</th>
                  <th style={{ padding: '10px 14px' }}>Average Fake Prob</th>
                  <th style={{ padding: '10px 14px' }}>Ground Truth Ratio</th>
                  <th style={{ padding: '10px 14px' }}>Risk Assessment</th>
                  <th style={{ padding: '10px 14px' }}>Sample Node Identifiers</th>
                </tr>
              </thead>
              <tbody>
                {clusters.slice(0, 12).map(c => (
                  <tr key={c.cluster_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <td style={{ padding: '10px 14px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                      Cluster #{c.cluster_id}
                    </td>
                    <td style={{ padding: '10px 14px', color: '#fff' }}>
                      {c.size} profiles
                    </td>
                    <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#f43f5e' }}>
                      {(c.average_fake_prob * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)', color: '#94a3b8' }}>
                      {(c.ground_truth_fake_ratio * 100).toFixed(1)}% verified fake
                    </td>
                    <td style={{ padding: '10px 14px' }}>
                      <span className={`badge badge-${c.risk_level.toLowerCase()}`}>
                        {c.risk_level}
                      </span>
                    </td>
                    <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      {c.node_ids.slice(0, 8).map(id => `#${id}`).join(', ')}...
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
