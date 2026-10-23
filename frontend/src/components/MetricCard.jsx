import React from 'react';

export default function MetricCard({ title, value, subtitle, icon: Icon, color = '#38bdf8', badge, progress }) {
  return (
    <div className="glass-panel" style={{
      padding: '20px 22px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Radiant accent line at top */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: 2,
        background: `linear-gradient(90deg, transparent, ${color}, transparent)`
      }} />

      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {title}
          </span>
          {Icon && (
            <div style={{
              width: 36,
              height: 36,
              borderRadius: 8,
              background: `rgba(${color === '#38bdf8' ? '56, 189, 248' : color === '#10b981' ? '16, 185, 129' : '168, 85, 247'}, 0.12)`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: color
            }}>
              <Icon size={18} />
            </div>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
          <span style={{
            fontFamily: 'var(--font-heading)',
            fontSize: '1.9rem',
            fontWeight: 800,
            color: '#fff',
            letterSpacing: '-0.03em'
          }}>
            {value}
          </span>
          {badge && (
            <span style={{
              fontSize: '0.72rem',
              fontWeight: 700,
              padding: '2px 8px',
              borderRadius: 4,
              background: 'rgba(56, 189, 248, 0.15)',
              color: '#38bdf8',
              border: '1px solid rgba(56, 189, 248, 0.3)'
            }}>
              {badge}
            </span>
          )}
        </div>
      </div>

      {progress !== undefined && (
        <div style={{ marginTop: 14 }}>
          <div style={{ width: '100%', height: 4, background: 'rgba(255,255,255,0.08)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{
              width: `${Math.min(100, Math.max(0, progress))}%`,
              height: '100%',
              background: color,
              borderRadius: 2
            }} />
          </div>
        </div>
      )}

      {subtitle && (
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8, margin: '8px 0 0 0' }}>
          {subtitle}
        </p>
      )}
    </div>
  );
}
