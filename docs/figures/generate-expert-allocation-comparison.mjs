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
    title: 'Choose precision for every expert projection',
    subtitle: 'Qwen3.8-Flash-Next · Actual layer 0 allocation · First 12 of 512 routed experts',
    gguf: 'GGUF · Unsloth UD IQ3_XXS',
    ggufNote: 'One format per packed Gate, Up or Down tensor',
    slots: 'allocation slots / layer',
    expert: 'Expert',
    shared: 'all 512 experts share this format',
    ggufScope: 'Three packed tensors, with one format selection for each projection.',
    bridge: 'Allocation slots per layer: 3 → 1,536 (512×)',
    mfq: 'MFQ · V3-XXS',
    mfqNote: 'An independent format and precision profile for each expert projection',
    vector: 'VQ',
    scalar: 'SQ',
    mfqScope: 'Profiles can differ across experts and across Gate, Up and Down within one expert.',
    legend: 'VQ: NVQ vector formats · SQ: NINT scalar formats · Cell labels give the selected profile.',
    scope: 'Actual profiles for layer 0, experts 0–11. Colors distinguish profiles, not sensitivity or quality.',
    caveat: 'GGUF comparison: this packed-tensor recipe; block-level quantization parameters can still vary.',
    total: 'Across 48 layers: 144 packed tensor slots → 73,728 expert-projection slots',
  },
  'zh-CN': {
    title: '精度分配，细到每个专家的每个投影',
    subtitle: 'Qwen3.8-Flash-Next · 第 0 层真实分配 · 展示 512 个路由专家中的前 12 个',
    gguf: 'GGUF · Unsloth UD IQ3_XXS',
    ggufNote: 'Gate、Up、Down 各用一个打包张量，每个张量选择一种格式',
    slots: '个分配位置／层',
    expert: '专家',
    shared: '全部 512 个专家共用该格式',
    ggufScope: '三个打包张量，分别为 Gate、Up、Down 选择格式。',
    bridge: '每层精度分配位置：3 → 1,536（512 倍）',
    mfq: 'MFQ · V3-XXS',
    mfqNote: '为每个专家的每个投影，独立选择格式与精度档位',
    vector: 'VQ',
    scalar: 'SQ',
    mfqScope: '不同专家可以选不同档位，同一专家的 Gate、Up、Down 也可以独立选择。',
    legend: 'VQ：NVQ 向量格式 · SQ：NINT 标量格式 · 单元格标注实际使用的档位。',
    scope: '第 0 层、专家 0–11 的真实分配。颜色区分格式档位，不代表敏感度或质量。',
    caveat: 'GGUF 对照限定于该配方的打包张量布局；块内量化参数仍可变化。',
    total: '覆盖 48 层：144 个打包张量分配位置 → 73,728 个专家投影分配位置',
  },
};

const xml = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&apos;'}[c]));
for (const [locale, t] of Object.entries(language)) {
  const parts = [];
  const text = (x, y, value, size = 18, weight = 400, fill = '#25354a', anchor = 'start') =>
    parts.push(`<text x="${x}" y="${y}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}">${xml(value)}</text>`);
  const rect = (x, y, width, height, fill, stroke = 'none', radius = 8) =>
    parts.push(`<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" fill="${fill}" stroke="${stroke}"/>`);

  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1060" viewBox="0 0 1600 1060" role="img" aria-labelledby="title desc" xml:lang="${locale}">`);
  parts.push(`<title id="title">${xml(t.title)}</title>`);
  parts.push(`<desc id="desc">${xml(t.subtitle + '. ' + t.ggufNote + '. ' + t.mfqNote + '. ' + t.caveat)}</desc>`);
  parts.push(`<metadata>${xml(JSON.stringify({model:data.mfq.model, layer:data.mfq.layer, expert_ids:expertIds, mfq_revision:data.mfq.revision, gguf_revision:data.gguf.revision, source:'data/qwen38-flash-next-layer0-allocation.json'}))}</metadata>`);
  rect(0, 0, 1600, 1060, '#ffffff', 'none', 0);
  parts.push('<g font-family="Noto Sans CJK SC, Noto Sans, Arial, sans-serif">');
  text(48, 57, t.title, 37, 700, '#14243a');
  text(48, 95, t.subtitle, 19, 400, '#62758d');

  const gridX = 190, cellWidth = 108, gridWidth = 12*cellWidth;
  rect(48, 124, 1504, 316, '#f6f8fb', '#dce3ec', 16);
  text(76, 165, t.gguf, 27, 700);
  text(76, 198, t.ggufNote, 18, 400, '#62758d');
  text(1496, 164, '3', 33, 700, '#52647b', 'end');
  text(1496, 197, t.slots, 16, 400, '#62758d', 'end');
  for (const id of expertIds) text(gridX + id*cellWidth + cellWidth/2, 231, `${t.expert} ${id}`, 14, 500, '#62758d', 'middle');
  for (const [row, projection] of projections.entries()) {
    const y = 247 + row*56;
    const label = data.gguf.tensor_types[projection];
    const [fill, stroke, ink] = colors[label];
    text(76, y+28, projection[0].toUpperCase()+projection.slice(1), 19, 600);
    rect(gridX, y, gridWidth, 42, fill, stroke, 8);
    for (let column=1; column<12; column++) {
      const x = gridX + column*cellWidth;
      parts.push(`<path d="M${x} ${y+4}v34" fill="none" stroke="${stroke}" stroke-dasharray="3 4" opacity="0.65"/>`);
    }
    rect(gridX + gridWidth/2-252, y+2, 504, 38, fill, 'none', 0);
    text(gridX + gridWidth/2, y+28, `${label} · ${t.shared}`, 18, 600, ink, 'middle');
  }
  text(76, 425, t.ggufScope, 15, 400, '#62758d');

  text(800, 484, t.bridge, 25, 700, '#976229', 'middle');

  rect(48, 510, 1504, 358, '#fbfaf7', '#dfd4c3', 16);
  rect(48, 528, 4, 48, '#a77338', 'none', 2);
  text(76, 551, t.mfq, 27, 700);
  text(76, 584, t.mfqNote, 18, 400, '#62758d');
  text(1496, 550, '1,536', 33, 700, '#976229', 'end');
  text(1496, 583, t.slots, 16, 400, '#62758d', 'end');
  for (const id of expertIds) text(gridX + id*cellWidth + cellWidth/2, 619, `${t.expert} ${id}`, 14, 500, '#62758d', 'middle');
  for (const [row, projection] of projections.entries()) {
    const y = 636 + row*64;
    const tensor = data.mfq.projections.find(p => p.projection === projection);
    text(76, y+33, projection[0].toUpperCase()+projection.slice(1), 19, 600);
    for (const id of expertIds) {
      const x = gridX + id*cellWidth;
      const profile = tensor.profiles[id];
      const [fill, stroke, ink] = colors[profile];
      parts.push(`<g data-projection="${projection}" data-expert="${id}" data-profile="${profile}">`);
      rect(x+2, y, cellWidth-5, 52, fill, stroke, 7);
      text(x+cellWidth/2, y+23, profile, 14, 700, ink, 'middle');
      text(x+cellWidth/2, y+42, profile.startsWith('NINT') ? t.scalar : t.vector, 11, 500, ink, 'middle');
      parts.push('</g>');
    }
  }
  text(76, 851, t.mfqScope, 16, 400, '#62758d');

  text(48, 909, t.legend, 17, 500);
  text(48, 944, t.scope, 16, 400, '#62758d');
  text(48, 975, t.caveat, 16, 400, '#62758d');
  rect(48, 997, 1504, 43, '#f4efe7', 'none', 9);
  text(800, 1026, t.total, 19, 600, '#78552d', 'middle');
  parts.push('</g></svg>');
  const suffix = locale === 'en' ? '' : '.zh-CN';
  const output = new URL(`./qwen38-flash-next-allocation-comparison${suffix}.svg`, import.meta.url);
  fs.writeFileSync(output, parts.join('\n')+'\n');
  console.log(`Wrote ${output.pathname}`);
}
