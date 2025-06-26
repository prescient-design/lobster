/** @type {import('./$types').PageLoad} */
import opentype from 'opentype.js';

// curl -L 'https://fonts.googleapis.com/css?family=Roboto%20Mono:400'
async function load_font(fetch) {
  //  const url = 'https://fonts.gstatic.com/s/roboto/v32/KFOmCnqEu92Fr1Mu4mxP.ttf'; // roboto 400
  //  const url = 'https://fonts.gstatic.com/s/robotomono/v23/L0xuDF4xlVMF-BfR8bXMIhJHg45mwgGEFl0_3vq_ROW9.ttf'; // roboto mono 400
  const url = 'https://fonts.gstatic.com/s/spacemono/v13/i7dPIFZifjKcF5UAWdDRYEF8QA.ttf'; // space mono 400
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  const font = opentype.parse(buffer);
  return font;
}

export async function load({ fetch }) {
  const font = await load_font(fetch);
  return { font };
}
