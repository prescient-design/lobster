function argsort(x, compareFn = (a, b) => a - b) {
  return Array.from(x.entries())
    .sort(([, a], [, b]) => compareFn(a, b))
    .map(([i]) => i);
}

export default argsort;
