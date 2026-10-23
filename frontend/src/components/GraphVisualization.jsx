import React, { useRef, useEffect, useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Shield, AlertTriangle, Eye, Layers } from 'lucide-react';

export default function GraphVisualization({ networkData, onNodeSelect, selectedNodeId }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [hoveredNode, setHoveredNode] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [filterRisk, setFilterRisk] = useState('ALL');

  const nodes = networkData?.visualization_subgraph?.nodes || [];
  const edges = networkData?.visualization_subgraph?.edges || [];

  // Compute node layout positions using a radial / force-inspired layout
  const nodePositions = useRef({});

  useEffect(() => {
    if (!nodes.length) return;
    const positions = {};
    const width = 850;
    const height = 550;
    const centerX = width / 2;
    const centerY = height / 2;

    nodes.forEach((n, idx) => {
      const data = n.data;
      // Cluster by risk tier or community
      let baseRadius;
      let angleOffset = (idx / nodes.length) * 2 * Math.PI;

      if (data.risk_tier === 'Critical') {
        baseRadius = 90 + (idx % 6) * 25;
      } else if (data.risk_tier === 'High') {
        baseRadius = 160 + (idx % 7) * 20;
      } else if (data.risk_tier === 'Moderate') {
        baseRadius = 220 + (idx % 5) * 25;
      } else {
        baseRadius = 270 + (idx % 6) * 30;
      }

      // Add pseudo-random deterministic jitter
      const jitterX = Math.sin(idx * 7.3) * 30;
      const jitterY = Math.cos(idx * 11.1) * 30;

      positions[data.id] = {
        x: centerX + Math.cos(angleOffset) * baseRadius + jitterX,
        y: centerY + Math.sin(angleOffset) * baseRadius + jitterY,
        radius: data.risk_tier === 'Critical' ? 10 : data.risk_tier === 'High' ? 8 : 7,
        data: data
      };
    });

    nodePositions.current = positions;
  }, [nodes]);

  // Canvas render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.save();

    // Apply Pan and Zoom
    ctx.translate(pan.x + width / 2, pan.y + height / 2);
    ctx.scale(zoom, zoom);
    ctx.translate(-width / 2, -height / 2);

    const positions = nodePositions.current;

    // Draw Edges
    edges.forEach(e => {
      const s = positions[e.data.source];
      const t = positions[e.data.target];
      if (!s || !t) return;

      // Filter check
      if (filterRisk !== 'ALL' && s.data.risk_tier !== filterRisk && t.data.risk_tier !== filterRisk) return;

      const isHighlighted = (hoveredNode && (hoveredNode.id === s.data.id || hoveredNode.id === t.data.id)) ||
                            (selectedNode && (selectedNode.id === s.data.id || selectedNode.id === t.data.id));

      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);

      if (isHighlighted) {
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.7)';
        ctx.lineWidth = 1.8;
      } else {
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
        ctx.lineWidth = 0.8;
      }
      ctx.stroke();
    });

    // Draw Nodes
    Object.values(positions).forEach(n => {
      const data = n.data;
      if (filterRisk !== 'ALL' && data.risk_tier !== filterRisk) return;

      const isHovered = hoveredNode && hoveredNode.id === data.id;
      const isSelected = selectedNode && selectedNode.id === data.id;

      // Node Colors based on Risk Tier
      let fillColor;
      let glowColor;
      if (data.risk_tier === 'Critical') {
        fillColor = '#f43f5e';
        glowColor = 'rgba(244, 63, 94, 0.5)';
      } else if (data.risk_tier === 'High') {
        fillColor = '#fb923c';
        glowColor = 'rgba(251, 146, 60, 0.5)';
      } else if (data.risk_tier === 'Moderate') {
        fillColor = '#38bdf8';
        glowColor = 'rgba(56, 189, 248, 0.5)';
      } else {
        fillColor = '#10b981';
        glowColor = 'rgba(16, 185, 129, 0.5)';
      }

      // Outer glow for hovered or critical nodes
      if (isHovered || isSelected || data.risk_tier === 'Critical') {
        ctx.save();
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius + (isHovered || isSelected ? 8 : 4), 0, Math.PI * 2);
        ctx.fillStyle = glowColor;
        ctx.fill();
        ctx.restore();
      }

      // Main Node Circle
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.lineWidth = isSelected ? 2.5 : 1.2;
      ctx.strokeStyle = isSelected ? '#ffffff' : 'rgba(255, 255, 255, 0.4)';
      ctx.stroke();

      // Node ID Label
      if (isHovered || isSelected || zoom >= 1.4) {
        ctx.font = '10px JetBrains Mono';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.fillText(`#${data.id}`, n.x, n.y - n.radius - 5);
      }
    });

    ctx.restore();
  }, [nodes, edges, zoom, pan, hoveredNode, selectedNode, filterRisk]);

  // Mouse handlers for zoom, pan, and click
  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Convert mouse coordinates to world coordinates
    const worldX = (mouseX - (pan.x + canvas.width / 2)) / zoom + canvas.width / 2;
    const worldY = (mouseY - (pan.y + canvas.height / 2)) / zoom + canvas.height / 2;

    let found = null;
    Object.values(nodePositions.current).forEach(n => {
      const dist = Math.hypot(n.x - worldX, n.y - worldY);
      if (dist <= n.radius + 6) {
        found = n.data;
      }
    });

    setHoveredNode(found);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleClick = () => {
    if (hoveredNode) {
      setSelectedNode(hoveredNode);
      if (onNodeSelect) onNodeSelect(hoveredNode);
    }
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div style={{ position: 'relative', width: '100%', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
      {/* Top Controls Toolbar */}
      <div style={{
        position: 'absolute',
        top: 14,
        left: 14,
        right: 14,
        zIndex: 10,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        pointerEvents: 'none'
      }}>
        {/* Risk Filter Buttons */}
        <div style={{ display: 'flex', gap: 6, pointerEvents: 'auto', background: 'rgba(15, 23, 42, 0.85)', backdropFilter: 'blur(8px)', padding: 4, borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
          {['ALL', 'Critical', 'High', 'Moderate', 'Low'].map(tier => (
            <button
              key={tier}
              onClick={() => setFilterRisk(tier)}
              style={{
                background: filterRisk === tier ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                color: filterRisk === tier ? '#38bdf8' : 'var(--text-secondary)',
                border: filterRisk === tier ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid transparent',
                borderRadius: 6,
                padding: '4px 10px',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              {tier}
            </button>
          ))}
        </div>

        {/* Zoom Controls */}
        <div style={{ display: 'flex', gap: 6, pointerEvents: 'auto' }}>
          <button
            onClick={() => setZoom(z => Math.min(2.5, z + 0.2))}
            className="btn btn-secondary"
            style={{ padding: '6px 10px' }}
            title="Zoom In"
          >
            <ZoomIn size={16} />
          </button>
          <button
            onClick={() => setZoom(z => Math.max(0.5, z - 0.2))}
            className="btn btn-secondary"
            style={{ padding: '6px 10px' }}
            title="Zoom Out"
          >
            <ZoomOut size={16} />
          </button>
          <button
            onClick={resetView}
            className="btn btn-secondary"
            style={{ padding: '6px 10px' }}
            title="Reset View"
          >
            <RotateCcw size={16} />
          </button>
        </div>
      </div>

      {/* Main Canvas */}
      <canvas
        ref={canvasRef}
        width={950}
        height={560}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
        style={{
          width: '100%',
          height: 560,
          background: 'radial-gradient(circle at center, #0e1628 0%, #070a12 100%)',
          cursor: isDragging ? 'grabbing' : hoveredNode ? 'pointer' : 'grab',
          display: 'block'
        }}
      />

      {/* Legend at Bottom Left */}
      <div style={{
        position: 'absolute',
        bottom: 14,
        left: 14,
        background: 'rgba(15, 23, 42, 0.88)',
        backdropFilter: 'blur(10px)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 8,
        padding: '8px 14px',
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        fontSize: '0.75rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#f43f5e', boxShadow: '0 0 6px #f43f5e' }} />
          <span>Critical Risk (≥80%)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 9, height: 9, borderRadius: '50%', background: '#fb923c' }} />
          <span>High (≥60%)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#38bdf8' }} />
          <span>Moderate (≥40%)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981' }} />
          <span>Low (Genuine)</span>
        </div>
      </div>

      {/* Selected Node Details Drawer */}
      {selectedNode && (
        <div style={{
          position: 'absolute',
          bottom: 14,
          right: 14,
          width: 280,
          background: 'rgba(15, 23, 42, 0.95)',
          backdropFilter: 'blur(12px)',
          border: '1px solid var(--border-focus)',
          borderRadius: 10,
          padding: 16,
          boxShadow: 'var(--shadow-glow)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
            <span style={{ fontWeight: 700, fontSize: '0.9rem', color: '#fff' }}>
              Node #{selectedNode.id} Detail
            </span>
            <button
              onClick={() => setSelectedNode(null)}
              style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1rem' }}
            >
              ✕
            </button>
          </div>
          <div style={{ fontSize: '0.8rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Risk Tier:</span>
              <span className={`badge badge-${selectedNode.risk_tier.toLowerCase()}`}>
                {selectedNode.risk_tier}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Fake Probability:</span>
              <span style={{ fontWeight: 700, color: '#f43f5e' }}>
                {(selectedNode.fake_prob * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Ground Truth:</span>
              <span style={{ color: selectedNode.ground_truth === 'Fake' ? '#fb7185' : '#34d399', fontWeight: 600 }}>
                {selectedNode.ground_truth}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Model Prediction:</span>
              <span style={{ color: selectedNode.predicted === 'Fake' ? '#fb7185' : '#34d399', fontWeight: 600 }}>
                {selectedNode.predicted}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Graph Degree:</span>
              <span>{selectedNode.degree} similarity edges</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Neighbor Fake Ratio:</span>
              <span>{(selectedNode.neighbor_fake_ratio * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
