<script lang="ts">
  import { onMount } from 'svelte';
  import flower from '$lib/flower.json';
  import * as d3 from 'd3';

  export let font;
  export let probs;
  export let xticks = [20];
  export let width = 500;
  export let height = 500;
  export let marginTop = 20;
  export let marginBottom = 30;
  export let marginLeft = 40;
  export let marginRight = 40;
  export let params = { padding: 0, colorScheme: flower, border: false };
  export let alphabet = [...'ACDEFGHIKLMNPQRSTVWY'];

  let svg: SVGElement;
  let gx;
  let gy;
  let visibility = 'hidden';

  $: x = d3.scaleLinear([0, probs.length], [marginLeft, width - marginRight]);
  $: y = d3.scaleLinear([0, maxEntropy], [height - marginBottom, marginTop]);

  $: d3.select(gy).call(d3.axisLeft(y));
  $: d3.select(gx).call(d3.axisBottom(x).ticks(xticks, ',f')); // FIXME how to make sensible ticks for different sequence lengths
  $: maxEntropy = Math.log2(alphabet.length);
  $: glyphContainer = probs.map(() => alphabet.map(() => null));

  let entropy = (p) => p.map((a) => -a * Math.log2(a)).reduce((a, b) => a + b, 0);

  onMount(() => {
    for (const [i, col] of probs.entries()) {
      console.assert(col.length == alphabet.length);
      let S = entropy(col);
      let height = 0;
      let width = 0;
      for (const [j, pj] of col.entries()) {
        let inner = d3.select(glyphContainer[i][j]).select('.glyphContainerInner').node();
        let bbox = inner.getBBox();
        height += pj * bbox.height;
        width += bbox.width;
      }
      let avgWidth = width / alphabet.length;
      let xscale = (x(1) - x(0)) / avgWidth;

      let R = maxEntropy - S;
      let ypos = y(R);
      let yscale = (y(0) - ypos) / height;

      const sorted = Array.from(col.entries())
        .map(([j, p]) => ({ j, p }))
        .sort(({ p: pa }, { p: pb }) => pb - pa);

      for (const { j, p } of sorted) {
        let inner = d3.select(glyphContainer[i][j]).select('.glyphContainerInner').node();
        let path = d3.select(inner).select('path').node();
        let rect = d3.select(inner).select('.glyphBBoxRect').node();

        path.setAttribute('transform', `scale(${xscale}, ${p * yscale})`);

        let bbox = inner.getBBox();
        inner.setAttribute(
          'transform',
          `scale(${(xscale * avgWidth) / bbox.width}, 1.0) translate(${-bbox.x}, ${-bbox.y})`
        );

        rect.setAttribute('x', `${bbox.x}`);
        rect.setAttribute('y', `${bbox.y}`);
        rect.setAttribute('width', `${bbox.width}`);
        rect.setAttribute('height', `${bbox.height}`);

        glyphContainer[i][j].setAttribute('transform', `translate(${x(i)}, ${ypos})`);
        ypos += bbox.height;
      }
      visibility = 'visible';
    }
  });
</script>

<svg bind:this={svg} {width} {height} style="visibility: {visibility}">
  <g bind:this={gx} transform="translate(0,{height - marginBottom})" />
  <g bind:this={gy} transform="translate({marginLeft},0)" />

  <text transform="translate({x(-2.1)}, {y(2)}) rotate(-90)">bits</text>

  {#each probs.keys() as i}
    <g class="stack">
      {#each alphabet as c, j}
        <g class="glyphContainer" bind:this={glyphContainer[i][j]}>
          <g class="glyphContainerInner">
            <path
              class={c}
              d={font.getPath(c, 0, 0).toPathData()}
              fill={params.colorScheme.colors[c]}
            />
            <rect class="glyphBBoxRect" />
          </g>
        </g>
      {/each}
    </g>
  {/each}
</svg>

<style>
  .glyphBBoxRect {
    stroke: none;
    fill: none;
  }
</style>
