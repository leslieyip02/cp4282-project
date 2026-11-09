const ANALYSIS_KERNELS = ['linear', 'polynomial'];
const ANALYSIS_COLORS = {linear: '#377bb5', polynomial: '#c87139'};
const byId = id => document.getElementById(id);

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function scenePsnrDelta(scene, kernel) {
  return average(scene.seeds.map(seed =>
    scene.runs[kernel][String(seed)].summary.mean_psnr_db -
    scene.runs.gaussian[String(seed)].summary.mean_psnr_db));
}

function signedValue(value, digits = 3, suffix = '') {
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}${suffix}`;
}

function drawDeltaPanel(id, scenes, kernel) {
  const rows = Object.entries(scenes).map(([name, scene]) => ({name, delta: scenePsnrDelta(scene, kernel)}));
  const max = Math.max(...rows.map(row => Math.abs(row.delta)), 0.01);
  const panel = byId(id);
  panel.insertAdjacentHTML('beforebegin',
    `<p class="delta-scale">Panel scale: ±${max.toFixed(2)} dB · center line = Gaussian</p>`);
  panel.innerHTML = rows.map(({name, delta}) => {
    const width = 45 * Math.abs(delta) / max;
    const left = delta >= 0 ? 50 : 50 - width;
    return `<div class="delta-row"><span class="delta-name">${name}</span><div class="delta-track" role="img" aria-label="${name}: ${signedValue(delta)} decibels versus Gaussian"><i style="left:${left}%;width:${width}%;background:${ANALYSIS_COLORS[kernel]}"></i></div><strong class="${delta >= 0 ? 'delta-good' : 'delta-bad'}">${signedValue(delta)}</strong></div>`;
  }).join('');
}

function costForScene(scene, kernel) {
  const all = [], matched = [], memory = [];
  for (const seed of scene.seeds) {
    const baseline = scene.runs.gaussian[String(seed)].summary;
    const variant = scene.runs[kernel][String(seed)].summary;
    all.push(100 * (variant.mean_tile_pairs / baseline.mean_tile_pairs - 1));
    if (baseline.gpu_name && baseline.gpu_name === variant.gpu_name) {
      matched.push({render: 100 * (variant.mean_render_seconds / baseline.mean_render_seconds - 1),
                    train: 100 * (variant.training_elapsed_seconds / baseline.training_elapsed_seconds - 1)});
      if (baseline.peak_train_gpu_memory_mib != null && variant.peak_train_gpu_memory_mib != null)
        memory.push(variant.peak_train_gpu_memory_mib - baseline.peak_train_gpu_memory_mib);
    }
  }
  return {pairs: matched.length,
          render: matched.length ? average(matched.map(row => row.render)) : null,
          train: matched.length ? average(matched.map(row => row.train)) : null,
          tile: average(all), memory};
}

function range(values) {
  return `${signedValue(Math.min(...values), 1, '%')} to ${signedValue(Math.max(...values), 1, '%')}`;
}

function drawCosts(scenes) {
  const rows = ANALYSIS_KERNELS.map(kernel => {
    const sceneCosts = Object.values(scenes).map(scene => costForScene(scene, kernel));
    const render = sceneCosts.map(row => row.render).filter(value => value != null);
    const train = sceneCosts.map(row => row.train).filter(value => value != null);
    const tile = sceneCosts.map(row => row.tile);
    const memory = sceneCosts.flatMap(row => row.memory);
    const pairs = sceneCosts.reduce((sum, row) => sum + row.pairs, 0);
    const memoryLabel = memory.length && memory.every(value => value === 0) ? '0 MiB observed' :
      memory.length ? `${Math.min(...memory).toFixed(0)} to ${Math.max(...memory).toFixed(0)} MiB` : '—';
    return `<tr><th scope="row" class="cost-kernel ${kernel}">${kernel}</th><td>${pairs}/27</td><td>${range(render)}</td><td>${range(train)}</td><td>${range(tile)}</td><td>${memoryLabel}</td></tr>`;
  });
  byId('cost-rows').innerHTML = rows.join('');
}

async function startAnalysis() {
  try {
    const [realResponse, syntheticResponse] = await Promise.all([
      fetch('data.json'), fetch('synthetic-results.json')]);
    if (!realResponse.ok || !syntheticResponse.ok) throw new Error('The data files could not be loaded.');
    const [real, synthetic] = await Promise.all([realResponse.json(), syntheticResponse.json()]);
    if (!Object.keys(real.scenes || {}).length || !Object.keys(synthetic.scenes || {}).length)
      throw new Error('Both real and synthetic summaries are needed for this analysis.');
    for (const kernel of ANALYSIS_KERNELS) {
      drawDeltaPanel(`${kernel}-synthetic`, synthetic.scenes, kernel);
      drawDeltaPanel(`${kernel}-real`, real.scenes, kernel);
    }
    drawCosts(real.scenes);
  } catch (error) {
    document.querySelector('.analysis-note').insertAdjacentHTML('beforebegin',
      `<div class="analysis-error" role="alert">${error.message} Rebuild the site with <code>--synthetic-root</code> and serve the generated directory over HTTP.</div>`);
  }
}

startAnalysis();
