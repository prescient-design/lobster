<script lang="ts">
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  export let font;
  export let probs;
  export let sequence;
  export let threshold = 10.0;
  export let xticks = [20];
  export let width = 500;
  export let height = 500;
  export let marginTop = 20;
  export let marginBottom = 30;
  export let marginLeft = 40;
  export let marginRight = 40;

  let svg: SVGElement;
  let gx;
  let gy;

  const alphabet = [...'ACDEFGHIKLMNPQRSTVWY'];
  const argmax = (v) => v.reduce((iMax, x, i, arr) => (x > arr[iMax] ? i : iMax), 0);

  $: x = d3.scaleLinear([0, sequence.length], [marginLeft, width - marginRight]);
  $: y = d3.scaleLinear([0, 1.0], [height - marginBottom, marginTop]);

  $: d3.select(gy).call(d3.axisLeft(y).ticks([]));
  $: d3.select(gx).call(d3.axisBottom(x).ticks(xticks, ',f')); // FIXME how to make sensible ticks for different sequence lengths

  $: glyphContainer = [...sequence].map(() => null);

  $: argmax_sequence = probs.map((v) => alphabet[argmax(v)]);

  $: wt_probs = Array.from(probs.entries().map(([i, v]) => v[alphabet.indexOf(sequence[i])]));
  $: argmax_probs = Array.from(
    probs.entries().map(([i, v]) => v[alphabet.indexOf(argmax_sequence[i])])
  );

  $: mutated_sequence = Array(sequence.length)
    .keys()
    .map((i) => (argmax_probs[i] / wt_probs[i] > threshold ? argmax_sequence[i] : sequence[i]));
  //  $: mutated_sequence = Array(sequence.length).keys().map((i) => argmax_sequence[i]);

  onMount(() => {
    console.log([...argmax_probs]);
    for (const [i, c] of [...sequence].entries()) {
      let container = glyphContainer[i];
      let inner = d3.select(container).select('.glyphContainerInner').node();
      let path = d3.select(inner).select('path').node();

      let bbox = inner.getBBox();

      container.setAttribute('transform', `translate(${x(i)}, 0)`);

      let xscale = x(1) - x(0);
      inner.setAttribute(
        'transform',
        `scale(${(0.9 * xscale) / bbox.width}, 1.0) translate(${-bbox.x}, 0)`
      );
      container.setAttribute('transform', `translate(${x(i)}, ${y(0)})`);
    }
  });
</script>

<svg bind:this={svg} {width} {height}>
  <g bind:this={gx} transform="translate(0,{height - marginBottom})" />
  <g bind:this={gy} transform="translate({marginLeft},0)" />
  {#each mutated_sequence as c, i}
    <g class="glyphContainer" bind:this={glyphContainer[i]}>
      <g class="glyphContainerInner">
        <path
          class={c}
          d={font.getPath(c, 0, 0).toPathData()}
          fill={c == sequence[i] ? 'black' : 'red'}
        />
      </g>
    </g>
  {/each}
</svg>

<style>
</style>
