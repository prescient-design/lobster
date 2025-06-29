import defaultVocabData from '$lib/vocab.json';

const defaultVocab: Map<string, number> = new Map(Object.entries(defaultVocabData));

function tokenize(input: string, vocab: Map<string, number> = defaultVocab) {
  const tokens = [];
  let i = 0;

  // Sort vocab keys by length in descending order once, outside the loop
  const sortedKeys = Array.from(vocab.keys()).sort((a, b) => b.length - a.length);

  while (i < input.length) {
    let matched = false;

    // Iterate over the sorted vocab keys
    for (const key of sortedKeys) {
      if (input.startsWith(key, i)) {
        tokens.push(vocab.get(key));
        i += key.length;
        matched = true;
        break;
      }
    }

    // If no match is found, use <unk>
    if (!matched) {
      tokens.push(vocab.get('<unk>'));
      i++;
    }
  }

  return tokens;
}

export default tokenize;
