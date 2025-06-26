function normalize(v: number[]) {
  let norm = v.reduce((a, b) => a + b, 0);
  return v.map((a) => a / norm);
}

export default normalize;
