<script lang="ts">
  import Logo from '$lib/Logo.svelte';
  import Sequence from '$lib/Sequence.svelte';
  import Mutations from '$lib/Mutations.svelte';
  import { PUBLIC_INFERENCE_SERVER } from '$env/static/public';
  import normalize from '$lib/normalize.ts';

  let sequence = $state(
    'DIQMTQSPSSLSASVGDRVTITCQASQDIGISLSWYQQKPGKAPKLLIYNANNLADGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCLQHNSAPYTFGQGTKLEIKR'
  );
  let { data } = $props();

  let models = ['esm2_t6_8M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'];

  let selectedModel = $state(models[1]);
  let threshold = $state(1);

  const alphabet = [...'ACDEFGHIKLMNPQRSTVWY'];

  let inference_result = $derived.by(async () => {
    let inference_data = { sequence: sequence, model_name: `facebook/${selectedModel}` };

    let url = new URL('./naturalness', PUBLIC_INFERENCE_SERVER);

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(inference_data)
    });
    const data = await response.json();

    const { logp, wt_logp, naturalness } = data;

    //const probs = logp.map((v) => normalize(alphabet_vocab_indices.map((i) => Math.exp(v[i]))));

    const probs = logp.map((v) => v.map((p) => Math.exp(p)));

    const wt_probs = wt_logp.map((p) => Math.exp(p));

    return { probs, wt_probs, naturalness };
  });
</script>

<div class="app-container">
  <div class="header">
    <h1>Lobster Live</h1>
  </div>

  <div class="content-wrapper">
    <div class="sidebar">
      <div class="model-selection-container">
        <label for="model-select" class="model-select-label">Select Model</label>
        <select
          id="model-select"
          bind:value={selectedModel}
          class="model-select"
          title="Select a model from the list"
        >
          {#each models as model}
            <option value={model}>{model}</option>
          {/each}
        </select>
      </div>

      <div class="threshold-selection-container">
        <label for="threshold">Mutation Threshold: {threshold}x</label>
        <input
          type="range"
          id="threshold"
          name="threshold"
          min="1"
          max="100"
          bind:value={threshold}
        />
      </div>

      {#await inference_result then { naturalness }}
        <div class="naturalness-container">
          <div class="naturalness-label">Naturalness</div>
          <div class="naturalness-value">{naturalness.toFixed(2)}</div>
        </div>
      {/await}
    </div>
    <div class="main-content">
      <div class="container">
        <input bind:value={sequence} class="sequence-input" />
        <div class="character-guide">
          {'0123456789'.repeat(sequence.length / 10 + 1).slice(0, sequence.length)}
        </div>
        {#await inference_result then { probs, wt_probs }}
          <Logo {probs} font={data.font} width="1400" height="300" />
          <Sequence {sequence} probs={wt_probs} font={data.font} width="1400" height="100" />
          <Mutations {sequence} {probs} {threshold} font={data.font} width="1400" height="100" />
        {/await}
      </div>
    </div>
  </div>
</div>

<style>
  :global(body) {
    margin: 0;
    height: 100%;
    width: 100%;
  }

  h1 {
    margin: 0;
    margin-left: 1em;
    text-align: left;
  }

  .app-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    width: 100%;
    border: none;
  }

  .header {
    width: 100%;
    background-color: #001f3f;
    color: #fff;
    padding: 20px 0;
    text-align: center;
  }

  .content-wrapper {
    display: flex;
    flex-grow: 1;
    width: 100%;
  }

  .sidebar {
    width: 300px;
    background-color: #f4f4f4;
    padding: 20px;
    margin: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-sizing: border-box;
  }

  .model-select-label {
    margin-bottom: 10px;
    font-weight: bold;
    font-size: 1em;
  }
  .model-selection-container {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 20px;
    width: 90%;
  }

  .model-selection-container label {
    font-size: 1.1em;
    font-weight: 600;
    color: #333;
    margin-bottom: 10px;
    display: block;
  }

  .model-selection-container select {
    width: 100%;
    padding: 10px;
    font-size: 1em;
    border: 1px solid #ccc;
    border-radius: 5px;
    margin-top: 5px;
  }

  .threshold-selection-container {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-bottom: 20px;
    max-width: 220px;
    width: 90%;
  }

  .threshold-selection-container label {
    font-size: 1.1em;
    font-weight: 600;
    color: #333;
    margin-bottom: 10px;
    display: block;
  }

  .threshold-selection-container input {
    padding: 10px;
    font-size: 1em;
    border: 1px solid #ccc;
    border-radius: 5px;
    margin-top: 5px;
  }

  .model-select {
    margin-bottom: 15px;
    padding: 5px;
    font-size: 1em;
    border-radius: 5px;
    border: 1px solid #ccc;
    width: 80%;
  }

  .main-content {
    flex-grow: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    box-sizing: border-box;
  }

  .sequence-input {
    margin-bottom: 0px;
    padding: 0px;
    font-size: 1.2em;
    width: 1000px;
    max-width: 1000px;
    border: 2px solid #ccc;
    border-radius: 5px;
    font-family: monospace;
  }

  .character-guide {
    margin-bottom: 20px;
    padding: 0px;
    font-size: 1.2em;
    width: 1000px;
    max-width: 1000px;
    border: 2px solid #ccc;
    border-radius: 5px;
    font-family: monospace;
    color: grey;
  }

  .naturalness-container {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    padding: 15px;
    margin-top: 20px;
    width: 90%;
    max-width: 220px;
    text-align: center;
  }

  .naturalness-label {
    font-size: 1.1em;
    font-weight: 600;
    color: #333;
    margin-bottom: 10px;
    display: block;
  }

  .naturalness-value {
    font-size: 1.3em;
    font-weight: bold;
    color: #001f3f;
  }
</style>
