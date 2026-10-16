// Run with: node docs/figures/generate-expert-allocation-comparison.mjs
// The checked-in snapshot contains real layer-0 allocation metadata; no weights
// or network access are needed to regenerate the English and Chinese SVGs.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const source = new URL('./data/qwen38-flash-next-layer0-allocation.json', import.meta.url);
const data = JSON.parse(fs.readFileSync(source, 'utf8'));
const recipeText = fs.readFileSync(new URL('../../mfq/calibration/recipes/IQ3_XXS.json', import.meta.url), 'utf8');
const recipe = JSON.parse(recipeText);
const expertIds = Array.from({length: 12}, (_, id) => id);
const projections = ['gate', 'up', 'down'];
const nExperts = data.mfq.experts_per_layer;
const nLayers = data.mfq.layer_count;
assert.equal(nExperts, 512);
assert.equal(nLayers, 48);
assert.equal(recipe.revision, data.gguf.revision);
assert.equal(createHash('sha256').update(recipeText).digest('hex'), data.gguf.recipe_sha256);

const colors = {
  IQ2_S: ['#dce9f6', '#b0c7dd', '#214461'],
  IQ4_NL: ['#f2dfbd', '#d5ba89', '#654717'],
  'NVQ1-S': ['#faedc7', '#e3cd8e', '#614b14'],
  'NVQ1-L': ['#f6dca3', '#d9b867', '#614b14'],
  NVQ2J: ['#d3eee5', '#9ecdbd', '#204f44'],
  'NVQ2J-L': ['#a4dcd4', '#6cb9ae', '#164e47'],
  'NVQ2J-XL': ['#77c2cb', '#4399a5', '#123e49'],
  NVQ3J: ['#c4dcf7', '#91b6e2', '#22466c'],
  'NVQ3J-512': ['#b2bce8', '#8898cf', '#303d70'],
  'NVQ3J-L': ['#d4c2ee', '#ae92d2', '#503576'],
  NINT4: ['#efc590', '#d39b52', '#654014'],
};
for (const projection of data.mfq.projections) {
  assert.equal(projection.profiles.length, nExperts);
  const covered = new Set();
  for (const pool of projection.pools) {
    for (const id of pool.expert_ids) {
      assert(!covered.has(id));
      covered.add(id);
      assert.equal(projection.profiles[id], pool.profile);
    }
  }
  assert.equal(covered.size, nExperts);
  for (const profile of projection.profiles) assert(colors[profile], `Unknown profile: ${profile}`);
  const tensor = `blk.${data.mfq.layer}.ffn_${projection.projection}_exps.weight`;
  assert.equal(recipe.tensor_types[tensor], data.gguf.tensor_types[projection.projection]);
}

const language = {
  en: {
    title: 'Expert × projection precision',
    description: 'Qwen3.8-Flash-Next, layer 0, experts 0–11. This packed GGUF recipe uses one format per projection tensor; MFQ selects profiles per expert projection. Colors distinguish profiles. GGUF block-level parameters can still vary.',
    gguf: 'GGUF · Unsloth UD IQ3_XXS',
    slots: 'slots / layer',
    expert: 'Expert →',
    mfq: 'MFQ · V3-XXS',
  },
  'zh-CN': {
    title: '专家 × 投影精度分配',
    description: 'Qwen3.8-Flash-Next 第 0 层、专家 0–11 的真实档位。该 GGUF 配方的每个打包投影张量共用一种格式，MFQ 则按专家投影分配档位。颜色区分档位，GGUF 块内参数仍可变化。',
    gguf: 'GGUF · Unsloth UD IQ3_XXS',
    slots: '个分配位置／层',
    expert: '专家 →',
    mfq: 'MFQ · V3-XXS',
  },
};

const xml = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&apos;'}[c]));
for (const [locale, t] of Object.entries(language)) {
  const parts = [];
  const text = (x, y, value, size = 18, weight = 400, fill = '#25354a', anchor = 'start') =>
    parts.push(`<text x="${x}" y="${y}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}">${xml(value)}</text>`);
  const rect = (x, y, width, height, fill, stroke = 'none', radius = 8) =>
    parts.push(`<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" fill="${fill}" stroke="${stroke}"/>`);

  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="660" viewBox="0 0 1600 660" role="img" aria-labelledby="title desc" xml:lang="${locale}">`);
  parts.push(`<title id="title">${xml(t.title)}</title>`);
  parts.push(`<desc id="desc">${xml(t.description)}</desc>`);
  parts.push(`<metadata>${xml(JSON.stringify({model:data.mfq.model, layer:data.mfq.layer, expert_ids:expertIds, mfq_revision:data.mfq.revision, gguf_revision:data.gguf.revision, source:'data/qwen38-flash-next-layer0-allocation.json'}))}</metadata>`);
  rect(0, 0, 1600, 660, '#ffffff', 'none', 0);
  parts.push('<g font-family="Noto Sans CJK SC, Noto Sans, Arial, sans-serif">');
  text(48, 49, t.title, 32, 700, '#14243a');

  const gridX = 190, cellWidth = 108, gridWidth = 12*cellWidth;
  rect(48, 78, 1504, 244, '#f6f8fb', '#dce3ec', 16);
  text(76, 120, t.gguf, 26, 700);
  text(1496, 118, '3', 32, 700, '#52647b', 'end');
  text(1496, 144, t.slots, 14, 400, '#62758d', 'end');
  text(76, 164, t.expert, 15, 500, '#62758d');
  for (const id of expertIds) text(gridX + id*cellWidth + cellWidth/2, 164, id, 15, 500, '#62758d', 'middle');
  for (const [row, projection] of projections.entries()) {
    const y = 179 + row*44;
    const label = data.gguf.tensor_types[projection];
    const [fill, stroke, ink] = colors[label];
    text(76, y+26, projection[0].toUpperCase()+projection.slice(1), 19, 600);
    rect(gridX, y, gridWidth, 36, fill, stroke, 7);
    for (let column=1; column<12; column++) {
      const x = gridX + column*cellWidth;
      parts.push(`<path d="M${x} ${y+4}v28" fill="none" stroke="${stroke}" stroke-dasharray="3 4" opacity="0.65"/>`);
    }
    rect(gridX + gridWidth/2-64, y+2, 128, 32, fill, 'none', 0);
    text(gridX + gridWidth/2, y+26, label, 18, 600, ink, 'middle');
  }
  text(800, 363, '512×', 28, 700, '#976229', 'middle');

  rect(48, 386, 1504, 254, '#fbfaf7', '#dfd4c3', 16);
  rect(48, 404, 4, 40, '#a77338', 'none', 2);
  text(76, 428, t.mfq, 26, 700);
  text(1496, 426, '1,536', 32, 700, '#976229', 'end');
  text(1496, 452, t.slots, 14, 400, '#62758d', 'end');
  text(76, 473, t.expert, 15, 500, '#62758d');
  for (const id of expertIds) text(gridX + id*cellWidth + cellWidth/2, 473, id, 15, 500, '#62758d', 'middle');
  for (const [row, projection] of projections.entries()) {
    const y = 488 + row*46;
    const tensor = data.mfq.projections.find(p => p.projection === projection);
    text(76, y+28, projection[0].toUpperCase()+projection.slice(1), 19, 600);
    for (const id of expertIds) {
      const x = gridX + id*cellWidth;
      const profile = tensor.profiles[id];
      const [fill, stroke, ink] = colors[profile];
      parts.push(`<g data-projection="${projection}" data-expert="${id}" data-profile="${profile}">`);
      rect(x+2, y, cellWidth-5, 40, fill, stroke, 7);
      text(x+cellWidth/2, y+27, profile, 15, 700, ink, 'middle');
      parts.push('</g>');
    }
  }
  parts.push('</g></svg>');
  const suffix = locale === 'en' ? '' : '.zh-CN';
  const output = new URL(`./qwen38-flash-next-allocation-comparison${suffix}.svg`, import.meta.url);
  fs.writeFileSync(output, parts.join('\n')+'\n');
  console.log(`Wrote ${output.pathname}`);
}
