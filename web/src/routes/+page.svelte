<script lang="ts">
  import Logo from '$lib/Logo.svelte';
  import Sequence from '$lib/Sequence.svelte';
  import Mutations from '$lib/Mutations.svelte';
  import normalize from '$lib/normalize';

  let sequence = $state(
    'DIQMTQSPSSLSASVGDRVTITCQASQDIGISLSWYQQKPGKAPKLLIYNANNLADGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCLQHNSAPYTFGQGTKLEIKR'
  );
  let { data } = $props();

  let models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'];

  let selectedModel = $state(models[1]);
  let threshold = $state(1);
  let isLoading = $state(false);
  let inference_result = $state<any>(null);

  const alphabet = [...'ACDEFGHIKLMNPQRSTVWY'];

  async function runInference() {
    if (!sequence.trim()) return;

    isLoading = true;
    try {
      let inference_data = { sequence: sequence, model_name: `facebook/${selectedModel}` };

      // FIXME here we assume we are serving frontend and backend together
      const response = await fetch('../naturalness', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(inference_data)
      });
      const data = await response.json();

      const { logp, wt_logp, naturalness } = data;

      const probs = logp.map((v: any) => v.map((p: any) => Math.exp(p)));

      const wt_probs = wt_logp.map((p: any) => Math.exp(p));

      inference_result = { probs, wt_probs, naturalness };
    } catch (error) {
      console.error('Inference failed:', error);
    } finally {
      isLoading = false;
    }
  }
</script>

<div class="app-container">
  <div class="header">
    <div class="header-content">
      <h1>🦞 Lobster Live 🦞</h1>
      <p class="subtitle">Protein Sequence Analysis & Visualization</p>
    </div>
  </div>

  <div class="content-wrapper">
    <div class="sidebar">
      <div class="sidebar-section">
        <h3>Configuration</h3>

        <div class="control-group">
          <label for="model-select" class="control-label">Model Selection</label>
          <select
            id="model-select"
            bind:value={selectedModel}
            class="modern-select"
            title="Select a model from the list"
          >
            {#each models as model}
              <option value={model}>{model}</option>
            {/each}
          </select>
        </div>

        <div class="control-group">
          <label for="threshold" class="control-label">Mutation Threshold: {threshold}x</label>
          <input
            type="range"
            id="threshold"
            name="threshold"
            min="1"
            max="100"
            bind:value={threshold}
            class="modern-slider"
          />
        </div>
      </div>

      {#if inference_result}
        <div class="sidebar-section">
          <h3>Analysis</h3>
          <div class="metric-card">
            <div class="metric-label">Naturalness Score</div>
            <div class="metric-value">{inference_result.naturalness.toFixed(2)}</div>
          </div>
        </div>
      {/if}
    </div>

    <div class="main-content">
      <div class="container">
        <div class="input-section">
          <label for="sequence-input" class="input-label">Protein Sequence</label>
          <div class="textarea-container">
            <textarea
              id="sequence-input"
              bind:value={sequence}
              class="sequence-input"
              placeholder="Enter your protein sequence here..."
              rows="4"
            ></textarea>
            <button
              class="submit-button"
              onclick={runInference}
              disabled={isLoading || !sequence.trim()}
            >
              {isLoading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </div>

        <div class="visualization-section">
          {#if inference_result}
            <div class="viz-card">
              <h3>Sequence Logo</h3>
              <div class="viz-container">
                <Logo probs={inference_result.probs} font={data.font} width={1200} height={300} />
              </div>
            </div>

            <div class="viz-card">
              <h3>Sequence Visualization</h3>
              <div class="viz-container">
                <Sequence
                  {sequence}
                  probs={inference_result.wt_probs}
                  font={data.font}
                  width={1200}
                  height={100}
                />
              </div>
            </div>

            <div class="viz-card">
              <h3>Mutation Analysis</h3>
              <div class="viz-container">
                <Mutations
                  {sequence}
                  probs={inference_result.probs}
                  {threshold}
                  font={data.font}
                  width={1200}
                  height={100}
                />
              </div>
            </div>
          {:else if isLoading}
            <div class="viz-card">
              <div class="loading-message">
                <div class="loading-spinner"></div>
                <p>Analyzing sequence...</p>
              </div>
            </div>
          {:else}
            <div class="viz-card">
              <div class="empty-message">
                <p>Enter a protein sequence and click "Analyze Sequence" to see visualization</p>
              </div>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  :global(body) {
    margin: 0;
    height: 100%;
    width: 100%;
    font-family:
      -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  }

  .app-container {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    width: 100%;
    border: none;
  }

  .header {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
    flex-shrink: 0;
  }

  .header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
    pointer-events: none;
  }

  .header-content {
    position: relative;
    z-index: 1;
    padding: 2rem;
    text-align: center;
  }

  .header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .subtitle {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
    font-weight: 400;
  }

  .content-wrapper {
    display: flex;
    flex-grow: 1;
    width: 100%;
    min-height: 0;
  }

  .sidebar {
    width: 320px;
    background: white;
    padding: 2rem;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 2rem;
    box-shadow: 4px 0 20px rgba(0, 0, 0, 0.08);
    position: relative;
  }

  .sidebar-section {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .sidebar-section h3 {
    margin: 0;
    font-size: 1.3rem;
    font-weight: 600;
    color: #2d3748;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.5rem;
  }

  .control-group {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .control-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: #4a5568;
    margin: 0;
  }

  .modern-select {
    padding: 0.75rem;
    font-size: 1rem;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    background: white;
    transition: all 0.2s ease;
    cursor: pointer;
  }

  .modern-select:hover {
    border-color: #cbd5e0;
  }

  .modern-select:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }

  .modern-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: #e2e8f0;
    outline: none;
    transition: all 0.2s ease;
  }

  .modern-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
  }

  .modern-slider::-webkit-slider-thumb:hover {
    transform: scale(1.1);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }

  .modern-slider::-moz-range-thumb {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    cursor: pointer;
    border: none;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }

  .metric-card {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  }

  .metric-label {
    font-size: 0.9rem;
    font-weight: 500;
    color: #64748b;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .main-content {
    flex-grow: 1;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 2rem;
    overflow-y: auto;
    min-height: 0;
  }

  .container {
    display: flex;
    flex-direction: column;
    gap: 2rem;
    width: 100%;
    max-width: 1200px;
  }

  .input-section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .input-label {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2d3748;
    margin: 0;
  }

  .textarea-container {
    position: relative;
    width: 100%;
    max-width: 1200px;
  }

  .sequence-input {
    padding: 1rem;
    padding-right: 120px; /* Make room for button */
    font-size: 1.1rem;
    width: 100%;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    font-family: 'SF Mono', 'Monaco', 'Cascadia Code', 'Roboto Mono', monospace;
    background: white;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    resize: vertical;
    min-height: 120px;
    box-sizing: border-box;
  }

  .sequence-input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow:
      0 0 0 3px rgba(102, 126, 234, 0.1),
      0 4px 12px rgba(0, 0, 0, 0.1);
  }

  .submit-button {
    position: absolute;
    top: 1rem;
    right: 1rem;
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    font-weight: 600;
    color: white;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    white-space: nowrap;
    z-index: 1;
  }

  .submit-button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
  }

  .submit-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }

  .loading-message,
  .empty-message {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem;
    text-align: center;
  }

  .loading-message p,
  .empty-message p {
    margin: 0;
    color: #64748b;
    font-size: 1.1rem;
  }

  .loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #e2e8f0;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 1rem;
  }

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }

  .character-guide {
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    width: 100%;
    max-width: 1200px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    font-family: 'SF Mono', 'Monaco', 'Cascadia Code', 'Roboto Mono', monospace;
    color: #64748b;
    letter-spacing: 0.05em;
  }

  .visualization-section {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  .viz-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    transition: all 0.2s ease;
  }

  .viz-card:hover {
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    transform: translateY(-2px);
  }

  .viz-card h3 {
    margin: 0 0 1.5rem 0;
    font-size: 1.3rem;
    font-weight: 600;
    color: #2d3748;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.75rem;
  }

  .viz-container {
    width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
    border-radius: 8px;
    background: #fafafa;
    padding: 1rem;
    min-height: 120px;
    /* Allow content to scroll horizontally when needed */
    scrollbar-width: thin;
    scrollbar-color: #cbd5e0 #f7fafc;
  }

  .viz-container::-webkit-scrollbar {
    height: 8px;
  }

  .viz-container::-webkit-scrollbar-track {
    background: #f7fafc;
    border-radius: 4px;
  }

  .viz-container::-webkit-scrollbar-thumb {
    background: #cbd5e0;
    border-radius: 4px;
  }

  .viz-container::-webkit-scrollbar-thumb:hover {
    background: #a0aec0;
  }

  .viz-container :global(svg) {
    height: auto;
    display: block;
    min-width: 800px; /* Ensure minimum readable size */
    /* SVG will maintain its natural width for scrolling */
  }

  /* Responsive adjustments */
  @media (max-width: 1024px) {
    .content-wrapper {
      flex-direction: column;
    }

    .sidebar {
      width: 100%;
      flex-direction: row;
      gap: 2rem;
      padding: 1.5rem;
    }

    .sidebar-section {
      flex: 1;
      min-width: 0;
    }
  }

  @media (max-width: 768px) {
    .header-content {
      padding: 1.5rem;
    }

    .header h1 {
      font-size: 2rem;
    }

    .subtitle {
      font-size: 1rem;
    }

    .sidebar {
      flex-direction: column;
      gap: 1.5rem;
    }

    .main-content {
      padding: 1rem;
    }

    .sequence-input {
      font-size: 1rem;
    }

    .viz-container {
      padding: 0.5rem;
    }

    .viz-card {
      padding: 1rem;
    }
  }
</style>
