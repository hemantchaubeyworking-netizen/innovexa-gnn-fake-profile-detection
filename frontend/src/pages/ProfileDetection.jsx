import React, { useState, useEffect } from 'react';
import { Shield, AlertCircle, CheckCircle2, UserCheck, UserX, Cpu, Sparkles, HelpCircle, Layers, RefreshCw } from 'lucide-react';
import { predictProfile, fetchSamples } from '../services/api';

export default function ProfileDetection({ selectedModel, setSelectedModel, availableModels }) {
  const [formData, setFormData] = useState({
    profile_pic: 1,
    nums_length_username: 0.0,
    fullname_words: 2,
    nums_length_fullname: 0.0,
    name_equals_username: 0,
    description_length: 45,
    external_url: 0,
    is_private: 0,
    posts_count: 50,
    followers_count: 850,
    follows_count: 320,
    followers_following_ratio: 2.656
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState([]);

  useEffect(() => {
    fetchSamples()
      .then(res => setSamples(res.samples || []))
      .catch(() => {});
  }, []);

  // Update ratio automatically when followers or follows change
  const handleChange = (field, value) => {
    setFormData(prev => {
      const updated = { ...prev, [field]: value };
      if (field === 'followers_count' || field === 'follows_count') {
        const followers = field === 'followers_count' ? Number(value) : prev.followers_count;
        const follows = field === 'follows_count' ? Number(value) : prev.follows_count;
        updated.followers_following_ratio = Number((followers / Math.max(follows, 1)).toFixed(3));
      }
      return updated;
    });
  };

  const handleSelectSample = (sample) => {
    const d = sample.data;
    setFormData({
      profile_pic: d['profile pic'] ?? 1,
      nums_length_username: d['nums/length username'] ?? 0.0,
      fullname_words: d['fullname words'] ?? 1,
      nums_length_fullname: d['nums/length fullname'] ?? 0.0,
      name_equals_username: d['name==username'] ?? 0,
      description_length: d['description length'] ?? 0,
      external_url: d['external URL'] ?? 0,
      is_private: d['private'] ?? 0,
      posts_count: d['#posts'] ?? 0,
      followers_count: d['#followers'] ?? 0,
      follows_count: d['#follows'] ?? 0,
      followers_following_ratio: d['followers_following_ratio'] ?? 1.0
    });
    setResult(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = {
        ...formData,
        model_name: selectedModel
      };
      const response = await predictProfile(payload);
      setResult(response);
    } catch (err) {
      setError(err.message || 'Detection failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
      {/* Page Title */}
      <div>
        <h1 style={{ fontSize: '1.6rem', margin: 0 }}>Profile Threat Classification</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: 4 }}>
          Evaluate Instagram accounts using Graph Neural Network message passing and trained ML models.
        </p>
      </div>

      {/* Preset Samples Selector Bar */}
      {samples.length > 0 && (
        <div className="glass-panel" style={{ padding: '16px 20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
            <Sparkles size={16} color="#38bdf8" />
            <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff' }}>
              Load Real Dataset Profile Preset:
            </span>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {samples.map((s, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleSelectSample(s)}
                className="btn btn-secondary"
                style={{
                  fontSize: '0.78rem',
                  padding: '6px 12px',
                  borderRadius: 6,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6
                }}
              >
                <span style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: s.expected_class === 'Genuine' ? '#10b981' : s.expected_class === 'Fake' ? '#f43f5e' : '#f59e0b'
                }} />
                <span>{s.title}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main Two-Column Interface: Form on Left, Live Output on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 28, alignItems: 'start' }}>
        {/* Profile Attributes Input Form */}
        <div className="glass-panel" style={{ padding: 26 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
            <h2 style={{ fontSize: '1.2rem', margin: 0 }}>Input Profile Features (12)</h2>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              <span>Target:</span>
              <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>Model Inference</span>
            </div>
          </div>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            {/* Binary Attribute Toggles */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 12,
              background: 'rgba(15, 23, 42, 0.5)',
              padding: 14,
              borderRadius: 8
            }}>
              <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.85rem', cursor: 'pointer' }}>
                <span>Profile Picture Present</span>
                <input
                  type="checkbox"
                  checked={formData.profile_pic === 1}
                  onChange={e => handleChange('profile_pic', e.target.checked ? 1 : 0)}
                  style={{ width: 18, height: 18, accentColor: '#38bdf8', cursor: 'pointer' }}
                />
              </label>

              <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.85rem', cursor: 'pointer' }}>
                <span>Private Account</span>
                <input
                  type="checkbox"
                  checked={formData.is_private === 1}
                  onChange={e => handleChange('is_private', e.target.checked ? 1 : 0)}
                  style={{ width: 18, height: 18, accentColor: '#38bdf8', cursor: 'pointer' }}
                />
              </label>

              <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.85rem', cursor: 'pointer' }}>
                <span>External URL in Bio</span>
                <input
                  type="checkbox"
                  checked={formData.external_url === 1}
                  onChange={e => handleChange('external_url', e.target.checked ? 1 : 0)}
                  style={{ width: 18, height: 18, accentColor: '#38bdf8', cursor: 'pointer' }}
                />
              </label>

              <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.85rem', cursor: 'pointer' }}>
                <span>Name Equals Username</span>
                <input
                  type="checkbox"
                  checked={formData.name_equals_username === 1}
                  onChange={e => handleChange('name_equals_username', e.target.checked ? 1 : 0)}
                  style={{ width: 18, height: 18, accentColor: '#38bdf8', cursor: 'pointer' }}
                />
              </label>
            </div>

            {/* Continuous & Count Features */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
              {/* Username numeric ratio */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                  <span>Nums / Length Username</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{formData.nums_length_username}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={formData.nums_length_username}
                  onChange={e => handleChange('nums_length_username', parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: '#38bdf8' }}
                />
              </div>

              {/* Fullname numeric ratio */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                  <span>Nums / Length Fullname</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>{formData.nums_length_fullname}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={formData.nums_length_fullname}
                  onChange={e => handleChange('nums_length_fullname', parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: '#38bdf8' }}
                />
              </div>

              {/* Fullname words */}
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                  Fullname Word Count
                </label>
                <input
                  type="number"
                  min="0"
                  max="15"
                  value={formData.fullname_words}
                  onChange={e => handleChange('fullname_words', parseInt(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: 6,
                    background: 'rgba(15, 23, 42, 0.8)',
                    border: '1px solid var(--border-subtle)',
                    color: '#fff',
                    outline: 'none'
                  }}
                />
              </div>

              {/* Description Length */}
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                  Description Character Length
                </label>
                <input
                  type="number"
                  min="0"
                  max="200"
                  value={formData.description_length}
                  onChange={e => handleChange('description_length', parseInt(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: 6,
                    background: 'rgba(15, 23, 42, 0.8)',
                    border: '1px solid var(--border-subtle)',
                    color: '#fff',
                    outline: 'none'
                  }}
                />
              </div>

              {/* Posts Count */}
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                  Number of Posts (#posts)
                </label>
                <input
                  type="number"
                  min="0"
                  value={formData.posts_count}
                  onChange={e => handleChange('posts_count', parseInt(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: 6,
                    background: 'rgba(15, 23, 42, 0.8)',
                    border: '1px solid var(--border-subtle)',
                    color: '#fff',
                    outline: 'none'
                  }}
                />
              </div>

              {/* Followers Count */}
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                  Followers Count (#followers)
                </label>
                <input
                  type="number"
                  min="0"
                  value={formData.followers_count}
                  onChange={e => handleChange('followers_count', parseInt(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: 6,
                    background: 'rgba(15, 23, 42, 0.8)',
                    border: '1px solid var(--border-subtle)',
                    color: '#fff',
                    outline: 'none'
                  }}
                />
              </div>

              {/* Follows Count */}
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                  Following Count (#follows)
                </label>
                <input
                  type="number"
                  min="0"
                  value={formData.follows_count}
                  onChange={e => handleChange('follows_count', parseInt(e.target.value) || 0)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: 6,
                    background: 'rgba(15, 23, 42, 0.8)',
                    border: '1px solid var(--border-subtle)',
                    color: '#fff',
                    outline: 'none'
                  }}
                />
              </div>

              {/* Follower / Following Ratio */}
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                  Followers/Following Ratio (Auto)
                </label>
                <input
                  type="number"
                  step="0.001"
                  value={formData.followers_following_ratio}
                  readOnly
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: 6,
                    background: 'rgba(15, 23, 42, 0.4)',
                    border: '1px solid var(--border-subtle)',
                    color: '#94a3b8',
                    fontFamily: 'var(--font-mono)',
                    outline: 'none'
                  }}
                />
              </div>
            </div>

            {/* Submit Action Button */}
            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary"
              style={{ width: '100%', padding: '12px 20px', marginTop: 10 }}
            >
              {loading ? (
                <>
                  <RefreshCw size={16} className="animate-spin" />
                  <span>Computing GNN Neighborhood Messages...</span>
                </>
              ) : (
                <>
                  <Shield size={18} />
                  <span>Classify Account using {selectedModel}</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Live Output & Neighborhood Explanation Card */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {error && (
            <div className="glass-panel" style={{ padding: 18, borderLeft: '4px solid #f43f5e', background: 'rgba(244,63,94,0.1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#f43f5e', fontWeight: 600 }}>
                <AlertCircle size={18} />
                <span>Detection Error</span>
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: '6px 0 0 0' }}>{error}</p>
            </div>
          )}

          {result ? (
            <div className="glass-panel" style={{
              padding: 26,
              borderTop: `4px solid ${result.predicted_class === 'Fake' ? '#f43f5e' : '#10b981'}`
            }}>
              {/* Classification Outcome */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
                <div>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Predicted Classification
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 4 }}>
                    <span style={{
                      fontFamily: 'var(--font-heading)',
                      fontSize: '1.8rem',
                      fontWeight: 800,
                      color: result.predicted_class === 'Fake' ? '#f43f5e' : '#10b981'
                    }}>
                      {result.predicted_class} Account
                    </span>
                    <span className={`badge badge-${result.risk_category.toLowerCase().split(' ')[0]}`}>
                      {result.risk_category}
                    </span>
                  </div>
                </div>

                <div style={{
                  width: 52,
                  height: 52,
                  borderRadius: 12,
                  background: result.predicted_class === 'Fake' ? 'rgba(244,63,94,0.15)' : 'rgba(16,185,129,0.15)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: result.predicted_class === 'Fake' ? '#f43f5e' : '#10b981'
                }}>
                  {result.predicted_class === 'Fake' ? <UserX size={28} /> : <UserCheck size={28} />}
                </div>
              </div>

              {/* Fake Probability Progress Gauge */}
              <div style={{ marginBottom: 20 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: 6 }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Fake Account Probability:</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: result.fake_probability >= 0.5 ? '#f43f5e' : '#10b981' }}>
                    {(result.fake_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ width: '100%', height: 8, background: 'rgba(255,255,255,0.08)', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{
                    width: `${result.fake_probability * 100}%`,
                    height: '100%',
                    background: result.fake_probability >= 0.5 ? 'linear-gradient(90deg, #f59e0b, #f43f5e)' : '#10b981',
                    borderRadius: 4,
                    transition: 'width 0.4s ease'
                  }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4 }}>
                  <span>0% (Genuine)</span>
                  <span>Decision Threshold: {(result.decision_threshold * 100).toFixed(0)}%</span>
                  <span>100% (Fake)</span>
                </div>
              </div>

              {/* Neighborhood Context (GraphSAGE Inductive Aggregation) */}
              <div style={{
                background: 'rgba(15, 23, 42, 0.6)',
                borderRadius: 8,
                padding: '14px 16px',
                border: '1px solid var(--border-subtle)',
                marginBottom: 16
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <Layers size={16} color="#38bdf8" />
                  <span style={{ fontWeight: 600, fontSize: '0.85rem', color: '#fff' }}>
                    Graph Neighborhood Context ({result.neighborhood_context.nearest_neighbors_count} Nearest Neighbors)
                  </span>
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.4, margin: '0 0 10px 0' }}>
                  {result.neighborhood_context.note}
                </p>
                
                {result.neighborhood_context.neighbors.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {result.neighborhood_context.neighbors.map((n, i) => (
                      <span key={i} style={{
                        background: 'rgba(56, 189, 248, 0.1)',
                        border: '1px solid rgba(56, 189, 248, 0.25)',
                        padding: '3px 8px',
                        borderRadius: 4,
                        fontSize: '0.72rem',
                        fontFamily: 'var(--font-mono)',
                        color: '#38bdf8'
                      }}>
                        Node #{n.neighbor_id} (Sim: {(n.similarity_score * 100).toFixed(1)}%)
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* Research Limitation Disclaimer */}
              <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', fontSize: '0.74rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                <HelpCircle size={15} style={{ flexShrink: 0, marginTop: 2, color: 'var(--accent-amber)' }} />
                <span>{result.disclaimer}</span>
              </div>
            </div>
          ) : (
            <div className="glass-panel" style={{
              padding: 40,
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: 380,
              color: 'var(--text-muted)'
            }}>
              <div style={{
                width: 64,
                height: 64,
                borderRadius: '50%',
                background: 'rgba(56, 189, 248, 0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: 16
              }}>
                <Cpu size={32} color="#38bdf8" />
              </div>
              <h3 style={{ fontSize: '1.1rem', color: '#fff', marginBottom: 6 }}>Awaiting Input Profile</h3>
              <p style={{ fontSize: '0.85rem', maxWidth: 300, margin: 0 }}>
                Adjust attributes or pick a preset sample above, then submit to see live classification and graph neighborhood aggregation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
