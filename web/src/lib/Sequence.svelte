<script lang="ts">
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  export let font;
  export let probs;
  export let sequence;
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

  $: x = d3.scaleLinear([0, sequence.length], [marginLeft, width - marginRight]);
  $: y = d3.scaleLinear([0, 1.0], [height - marginBottom, marginTop]);

  $: d3.select(gy).call(d3.axisLeft(y).ticks([]));
  $: d3.select(gx).call(d3.axisBottom(x).ticks(xticks, ',f')); // FIXME how to make sensible ticks for different sequence lengths

  $: glyphContainer = [...sequence].map(() => null);

  const colorscale = d3.scaleSequential(d3.interpolateTurbo);

  onMount(() => {
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
  <defs>
    <linearGradient id="color-gradient" x1="0%" y1="100%" x2="0%" y2="0%">
      <!-- Define gradient stops -->
      {#each Array.from({ length: 101 }) as _, i}
        <stop offset="{i}%" stop-color={colorscale(i / 100)} />
      {/each}
    </linearGradient>
  </defs>

  <text text-anchor="middle" transform="translate({x(-2.1)}, {y(0.5)}) rotate(-90)">prob</text>

  <g bind:this={gx} transform="translate(0,{height - marginBottom})" />
  <g bind:this={gy} transform="translate({marginLeft},0)" />
  <g class="colorbarContainer">
    <rect
      class="colorbar"
      x={x(-1.3)}
      y={y(1)}
      width={x(1) - x(0)}
      height={y(0) - y(1)}
      fill="url(#color-gradient)"
    />
  </g>
  {#each sequence as c, i}
    <g class="glyphContainer" bind:this={glyphContainer[i]}>
      <g class="glyphContainerInner">
        <path class={c} d={font.getPath(c, 0, 0).toPathData()} fill={colorscale(probs[i])} />
      </g>
    </g>
  {/each}
</svg>

<style>
</style>
