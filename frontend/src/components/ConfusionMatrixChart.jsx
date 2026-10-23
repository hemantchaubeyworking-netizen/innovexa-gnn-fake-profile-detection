import React from 'react';

export default function ConfusionMatrixChart({ cmData, modelName = "Model" }) {
  if (!cmData) return null;

  const { true_positives: tp, false_positives: fp, true_negatives: tn, false_negatives: fn } = cmData;
  const total = tp + fp + tn + fn;

  const tpPct = ((tp / total) * 100).toFixed(1);
  const fpPct = ((fp / total) * 100).toFixed(1);
  const tnPct = ((tn / total) * 100).toFixed(1);
  const fnPct = ((fn / total) * 100).toFixed(1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: '80px 1fr 1fr',
        gap: 8,
        textAlign: 'center',
        fontSize: '0.82rem'
      }}>
        {/* Header row */}
        <div></div>
        <div style={{ color: 'var(--text-secondary)', fontWeight: 600, padding: 4 }}>
          Pred: Genuine (0)
        </div>
        <div style={{ color: 'var(--text-secondary)', fontWeight: 600, padding: 4 }}>
          Pred: Fake (1)
        </div>

        {/* Row 1: True Genuine */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-secondary)',
          fontWeight: 600,
          textAlign: 'right'
        }}>
          True: Genuine
        </div>
        
        {/* TN Box */}
        <div style={{
          background: 'rgba(16, 185, 129, 0.15)',
          border: '1px solid rgba(16, 185, 129, 0.4)',
          borderRadius: 8,
          padding: '16px 10px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <span style={{ fontSize: '0.72rem', color: '#34d399', fontWeight: 600 }}>True Negative</span>
          <span style={{ fontFamily: 'var(--font-heading)', fontSize: '1.5rem', fontWeight: 800, color: '#fff' }}>
            {tn}
          </span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{tnPct}%</span>
        </div>

        {/* FP Box */}
        <div style={{
          background: 'rgba(244, 63, 94, 0.1)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          borderRadius: 8,
          padding: '16px 10px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <span style={{ fontSize: '0.72rem', color: '#fb7185', fontWeight: 600 }}>False Positive</span>
          <span style={{ fontFamily: 'var(--font-heading)', fontSize: '1.5rem', fontWeight: 800, color: '#fb7185' }}>
            {fp}
          </span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{fpPct}%</span>
        </div>

        {/* Row 2: True Fake */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-secondary)',
          fontWeight: 600,
          textAlign: 'right'
        }}>
          True: Fake
        </div>

        {/* FN Box */}
        <div style={{
          background: 'rgba(244, 63, 94, 0.1)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          borderRadius: 8,
          padding: '16px 10px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <span style={{ fontSize: '0.72rem', color: '#fb7185', fontWeight: 600 }}>False Negative</span>
          <span style={{ fontFamily: 'var(--font-heading)', fontSize: '1.5rem', fontWeight: 800, color: '#fb7185' }}>
            {fn}
          </span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{fnPct}%</span>
        </div>

        {/* TP Box */}
        <div style={{
          background: 'rgba(56, 189, 248, 0.15)',
          border: '1px solid rgba(56, 189, 248, 0.4)',
          borderRadius: 8,
          padding: '16px 10px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <span style={{ fontSize: '0.72rem', color: '#38bdf8', fontWeight: 600 }}>True Positive</span>
          <span style={{ fontFamily: 'var(--font-heading)', fontSize: '1.5rem', fontWeight: 800, color: '#fff' }}>
            {tp}
          </span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{tpPct}%</span>
        </div>
      </div>

      {/* Summary stats below matrix */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 8,
        background: 'rgba(15, 23, 42, 0.5)',
        padding: '10px 14px',
        borderRadius: 8,
        fontSize: '0.75rem',
        textAlign: 'center'
      }}>
        <div>
          <div style={{ color: 'var(--text-muted)' }}>Precision</div>
          <div style={{ fontWeight: 700, color: '#38bdf8', fontSize: '0.9rem' }}>
            {tp + fp > 0 ? ((tp / (tp + fp)) * 100).toFixed(1) : 0}%
          </div>
        </div>
        <div>
          <div style={{ color: 'var(--text-muted)' }}>Recall</div>
          <div style={{ fontWeight: 700, color: '#10b981', fontSize: '0.9rem' }}>
            {tp + fn > 0 ? ((tp / (tp + fn)) * 100).toFixed(1) : 0}%
          </div>
        </div>
        <div>
          <div style={{ color: 'var(--text-muted)' }}>Specificity</div>
          <div style={{ fontWeight: 700, color: '#a855f7', fontSize: '0.9rem' }}>
            {tn + fp > 0 ? ((tn / (tn + fp)) * 100).toFixed(1) : 0}%
          </div>
        </div>
        <div>
          <div style={{ color: 'var(--text-muted)' }}>Total Tested</div>
          <div style={{ fontWeight: 700, color: '#fff', fontSize: '0.9rem' }}>
            {total}
          </div>
        </div>
      </div>
    </div>
  );
}
