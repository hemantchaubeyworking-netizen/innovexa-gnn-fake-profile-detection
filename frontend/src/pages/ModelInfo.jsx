import React from 'react';
import { BookOpen, Layers, GitBranch, Cpu, Database, AlertTriangle, Terminal, CheckCircle2 } from 'lucide-react';

export default function ModelInfo() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28, maxWidth: 1100 }}>
      {/* Title */}
      <div>
        <h1 style={{ fontSize: '1.6rem', margin: 0 }}>Research Methodology & Architecture</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: 4 }}>
          Comprehensive documentation of the INNOVEXA framework based on the primary research paper.
        </p>
      </div>

      {/* 5-Layer System Architecture */}
      <div className="glass-panel" style={{ padding: 26 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <Layers size={22} color="#38bdf8" />
          <h2 style={{ fontSize: '1.25rem', margin: 0 }}>Five-Layer System Design</h2>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 18 }}>
          As detailed in the research paper, the INNOVEXA framework decouples data ingestion, feature scaling,
          relational modeling, probability inference, and decision support into five modular layers:
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14 }}>
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: 14, borderRadius: 8, borderTop: '3px solid #38bdf8' }}>
            <span style={{ fontSize: '0.72rem', color: '#38bdf8', fontWeight: 700 }}>LAYER 1</span>
            <h3 style={{ fontSize: '0.95rem', margin: '4px 0 6px 0' }}>Data Collection</h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
              Ingests raw profile records, account statistics, and binary status indicators.
            </p>
          </div>

          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: 14, borderRadius: 8, borderTop: '3px solid #818cf8' }}>
            <span style={{ fontSize: '0.72rem', color: '#818cf8', fontWeight: 700 }}>LAYER 2</span>
            <h3 style={{ fontSize: '0.95rem', margin: '4px 0 6px 0' }}>Data Engineering</h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
              Leakage-free log1p transforms, robust scaling, and stratified 70/15/15 partitioning.
            </p>
          </div>

          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: 14, borderRadius: 8, borderTop: '3px solid #a855f7' }}>
            <span style={{ fontSize: '0.72rem', color: '#a855f7', fontWeight: 700 }}>LAYER 3</span>
            <h3 style={{ fontSize: '0.95rem', margin: '4px 0 6px 0' }}>Graph Modeling</h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
              kNN cosine similarity graph construction ($k=5$) and GNN message-passing convolutions.
            </p>
          </div>

          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: 14, borderRadius: 8, borderTop: '3px solid #10b981' }}>
            <span style={{ fontSize: '0.72rem', color: '#10b981', fontWeight: 700 }}>LAYER 4</span>
            <h3 style={{ fontSize: '0.95rem', margin: '4px 0 6px 0' }}>Prediction Layer</h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
              Learned node embeddings, calibrated probability scores, and validation F1 thresholding.
            </p>
          </div>

          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: 14, borderRadius: 8, borderTop: '3px solid #f59e0b' }}>
            <span style={{ fontSize: '0.72rem', color: '#f59e0b', fontWeight: 700 }}>LAYER 5</span>
            <h3 style={{ fontSize: '0.95rem', margin: '4px 0 6px 0' }}>Decision Support</h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
              Interactive graph exploration, suspicious cluster auditing, and explainability context.
            </p>
          </div>
        </div>
      </div>

      {/* Mathematical Formulations */}
      <div className="glass-panel" style={{ padding: 26 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <Cpu size={22} color="#818cf8" />
          <h2 style={{ fontSize: '1.25rem', margin: 0 }}>GNN Message Passing Formulations</h2>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* GraphSAGE */}
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: 18, borderRadius: 8 }}>
            <h3 style={{ fontSize: '1rem', color: '#38bdf8', marginBottom: 6 }}>1. GraphSAGE (Sample and Aggregate)</h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
              Iteratively aggregates localized attribute signals from neighboring profiles and normalizes representations:
            </p>
            <div style={{ background: '#0a0f1d', padding: '10px 14px', borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: '#7dd3fc' }}>
              h_{'{N(u)}'} = Mean({'{h_v : v ∈ N(u)}'}) <br />
              h_u^{'{l+1}'} = σ(W_self · h_u^{'{l}'} + W_neigh · h_{'{N(u)}'}) <br />
              h_u^{'{l+1}'} = h_u^{'{l+1}'} / ||h_u^{'{l+1}'}||_2
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 8, margin: '8px 0 0 0' }}>
              <strong>Inductive Benefit:</strong> Enables classification of completely unseen accounts during live inference by looking up
              nearest neighbors in the training graph without retraining.
            </p>
          </div>

          {/* GCN */}
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: 18, borderRadius: 8 }}>
            <h3 style={{ fontSize: '1rem', color: '#818cf8', marginBottom: 6 }}>2. Graph Convolutional Network (GCN)</h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
              Applies first-order Chebyshev spectral approximations with symmetric renormalized adjacency:
            </p>
            <div style={{ background: '#0a0f1d', padding: '10px 14px', borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: '#c7d2fe' }}>
              Ã = A + I_N <br />
              D̃_{'{ii}'} = Σ_j Ã_{'{ij}'} <br />
              H^{'{l+1}'} = σ(D̃^{'-1/2'} · Ã · D̃^{'-1/2'} · H^{'{l}'} · W^{'{l}'})
            </div>
          </div>

          {/* GAT */}
          <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: 18, borderRadius: 8 }}>
            <h3 style={{ fontSize: '1rem', color: '#a855f7', marginBottom: 6 }}>3. Graph Attention Network (GAT)</h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
              Multi-head self-attention dynamically weights neighbor connections based on attribute compatibility:
            </p>
            <div style={{ background: '#0a0f1d', padding: '10px 14px', borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: '#e9d5ff' }}>
              α_{'{ij}'} = Softmax_j(LeakyReLU(a_src^T · W · h_i + a_dst^T · W · h_j)) <br />
              h_i^{'{l+1}'} = ||_{'{k=1}'}^K σ(Σ_{'{j ∈ N(i)}'} α_{'{ij}'}^k · W^k · h_j)
            </div>
          </div>
        </div>
      </div>

      {/* Dataset Schema & Preprocessing */}
      <div className="glass-panel" style={{ padding: 26 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <Database size={22} color="#10b981" />
          <h2 style={{ fontSize: '1.25rem', margin: 0 }}>Input Features Schema (12 Non-Target Attributes)</h2>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                <th style={{ padding: '8px 10px' }}>Attribute</th>
                <th style={{ padding: '8px 10px' }}>Type</th>
                <th style={{ padding: '8px 10px' }}>Range</th>
                <th style={{ padding: '8px 10px' }}>Engineering / Transform</th>
              </tr>
            </thead>
            <tbody>
              {[
                { name: 'profile pic', type: 'Binary', range: '0 or 1', desc: 'Indicator for profile picture presence' },
                { name: 'nums/length username', type: 'Float', range: '[0.0, 1.0]', desc: 'Ratio of numeric characters in username' },
                { name: 'fullname words', type: 'Integer', range: '0 to 12', desc: 'Standardized word count' },
                { name: 'nums/length fullname', type: 'Float', range: '[0.0, 1.0]', desc: 'Ratio of numeric characters in full name' },
                { name: 'name==username', type: 'Binary', range: '0 or 1', desc: 'Equality between handle and name field' },
                { name: 'description length', type: 'Integer', range: '0 to 150', desc: 'Log1p scaled character count in biography' },
                { name: 'external URL', type: 'Binary', range: '0 or 1', desc: 'Link presence in biography' },
                { name: 'private', type: 'Binary', range: '0 or 1', desc: 'Account privacy flag' },
                { name: '#posts', type: 'Integer', range: '0 to 7,389', desc: 'Log1p scaled total published media' },
                { name: '#followers', type: 'Integer', range: '0 to 15.3M', desc: 'Log1p scaled heavy-tailed audience count' },
                { name: '#follows', type: 'Integer', range: '0 to 7,500', desc: 'Log1p scaled following account count' },
                { name: 'followers_following_ratio', type: 'Float', range: '[0.0, 1.3M]', desc: 'Computed ratio (#followers / max(#follows, 1))' }
              ].map(f => (
                <tr key={f.name} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600, color: '#fff', fontFamily: 'var(--font-mono)' }}>{f.name}</td>
                  <td style={{ padding: '8px 10px', color: 'var(--text-secondary)' }}>{f.type}</td>
                  <td style={{ padding: '8px 10px', fontFamily: 'var(--font-mono)' }}>{f.range}</td>
                  <td style={{ padding: '8px 10px', color: 'var(--text-muted)' }}>{f.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Reproducibility Commands */}
      <div className="glass-panel" style={{ padding: 26 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <Terminal size={22} color="#f59e0b" />
          <h2 style={{ fontSize: '1.25rem', margin: 0 }}>System Startup & Reproduction Guide</h2>
        </div>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 12 }}>
          Execute the following commands in PowerShell from the project root to start services or reproduce experiments:
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>1. Start FastAPI Backend:</span>
            <div style={{ background: '#0a0f1d', padding: '8px 12px', borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: '#38bdf8' }}>
              python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
            </div>
          </div>

          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>2. Start React + Vite Frontend:</span>
            <div style={{ background: '#0a0f1d', padding: '8px 12px', borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: '#38bdf8' }}>
              cd frontend ; npm run dev
            </div>
          </div>

          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>3. Run Complete Model Training & Evaluation Pipeline:</span>
            <div style={{ background: '#0a0f1d', padding: '8px 12px', borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: '#38bdf8' }}>
              python backend/ml/train.py
            </div>
          </div>

          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>4. Run Automated Test Suite:</span>
            <div style={{ background: '#0a0f1d', padding: '8px 12px', borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: '#38bdf8' }}>
              python backend/tests/run_tests.py
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
