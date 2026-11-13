const COLORS = {gaussian:'#545f61', linear:'#3575bd', polynomial:'#c67039'};
const KERNELS = ['gaussian','linear','polynomial'];
const $ = id => document.getElementById(id);
let data, syntheticData, scene, seed, step, view, zoom = 1;

function escapeHtml(value) { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function num(value, digits=2) { return value == null ? '—' : Number(value).toFixed(digits); }
function signed(value, digits=2) { return value == null ? '—' : `${value >= 0 ? '+' : ''}${num(value,digits)}`; }
function currentScene() { return data.scenes[scene]; }
function run(kernel) { return currentScene().runs[kernel][String(seed)]; }
function availableSteps() { return Object.keys(run('gaussian').steps).map(Number).filter(s => KERNELS.every(k => run(k).steps[String(s)])).sort((a,b)=>a-b); }
function selectedStats(kernel) { return run(kernel).steps[String(step)]?.stats; }

function selectScene(value) {
  scene = value;
  const seeds = currentScene().seeds;
  seed = seeds.includes(seed) ? seed : seeds[0];
  $('seed-select').innerHTML = seeds.map(s => `<option value="${s}">${s}</option>`).join('');
  $('seed-select').value = seed;
  const views = Object.entries(currentScene().views).filter(([key]) => KERNELS.some(k =>
    Object.values(run(k).steps).some(snapshot => snapshot.images[key])));
  $('view-select').innerHTML = views.map(([key,v]) => `<option value="${escapeHtml(key)}">${v.split.toUpperCase()} · ${escapeHtml(v.name)}</option>`).join('');
  view = views.some(([key]) => key === view) ? view : (views[0]?.[0] || '');
  $('view-select').value = view;
  const steps = availableSteps(); step = steps.includes(step) ? step : steps.at(-1);
  draw();
}

function drawTimeline() {
  const steps = availableSteps(), index = steps.indexOf(step);
  $('step-range').max = Math.max(steps.length-1,0); $('step-range').value = index;
  $('step-label').textContent = `${(step/1000).toFixed(step%1000?1:0)}k steps`;
  $('prev-step').disabled = index === 0; $('next-step').disabled = index === steps.length-1;
  $('step-ticks').innerHTML = steps.map(s => `<span>${s/1000}k</span>`).join('');
}

function imageCard(label, kernel, url, foot) {
  const img = url ? `<img alt="${escapeHtml(label)} at ${step} steps" src="${escapeHtml(url)}" loading="lazy" style="transform:scale(${zoom})">` :
    '<span class="empty">Image not exported for this camera and step</span>';
  return `<article class="image-card" data-kernel="${kernel}"><header><span>${escapeHtml(label)}</span><span class="badge">${kernel === 'reference' ? 'SOURCE' : 'RENDER'}</span></header><div class="image-viewport">${img}</div><div class="foot">${escapeHtml(foot)}</div></article>`;
}

function drawImages() {
  const meta = currentScene().views[view];
  $('view-heading').textContent = meta ? `${meta.split === 'val' ? 'Validation' : 'Training'} · ${meta.name}` : 'No views available';
  const coverage = KERNELS.filter(k => run(k).steps[String(step)]?.images[view]).length;
  $('view-note').textContent = coverage === 3 ? 'All three kernels at the same camera and step.' :
    'Some views need an export from the cluster. Snapshot statistics remain available.';
  let cards = imageCard('Reference', 'reference', meta?.reference,
    meta?.reference ? 'Resized source photograph' : 'Source photograph has not been exported');
  for (const kernel of KERNELS) {
    const source = run(kernel).steps[String(step)]?.images[view];
    const stats = selectedStats(kernel);
    cards += imageCard(kernel[0].toUpperCase()+kernel.slice(1),kernel,source,
      `${stats?.count.toLocaleString() || '—'} splats · ${step.toLocaleString()} steps`);
  }
  $('image-grid').innerHTML = cards;
}

function drawStats() {
  $('stat-grid').innerHTML = KERNELS.map(kernel => {
    const s=selectedStats(kernel);
    return `<article class="stat-card" data-kernel="${kernel}"><div class="title">${kernel}</div><div class="stat-items"><div><span>Splats</span><strong>${s?.count.toLocaleString() || '—'}</strong></div><div><span>Median scale</span><strong>${s ? num(s.medianScale,3) : '—'}</strong></div><div><span>Opacity ≥ 0.5</span><strong>${s ? Math.round(s.opaqueCount/s.count*100)+'%' : '—'}</strong></div></div></article>`;
  }).join('');
}

function svgChart(series, {minimum=null, maximum=null, xLabels=[], yDigits=0, histogram=false}={}) {
  const width=620,height=210,left=54,right=15,top=13,bottom=32;
  const all=series.flatMap(s=>s.values.filter(v=>Number.isFinite(v)));
  if (!all.length) return '<p class="empty">No chart data</p>';
  let min=minimum ?? Math.min(...all), max=maximum ?? Math.max(...all);
  if(min===max){min-=1;max+=1} const pad=(max-min)*.08; if(minimum==null)min-=pad;if(maximum==null)max+=pad;
  const x=i=>left+(width-left-right)*i/Math.max((series[0]?.values.length||1)-1,1);
  const y=v=>top+(height-top-bottom)*(max-v)/(max-min);
  let out=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Kernel comparison chart">`;
  for(let i=0;i<=4;i++){const value=min+(max-min)*(4-i)/4, yy=y(value);out+=`<line class="gridline" x1="${left}" x2="${width-right}" y1="${yy}" y2="${yy}"/><text class="axis" x="${left-7}" y="${yy+3}" text-anchor="end">${num(value,yDigits)}</text>`}
  xLabels.forEach(([index,label])=>{out+=`<text class="axis" x="${x(index)}" y="${height-9}" text-anchor="middle">${escapeHtml(label)}</text>`});
  series.forEach(s=>{if(!s.values.length)return;const points=s.values.map((v,i)=>`${x(i)},${y(v)}`).join(' ');out+=`<polyline points="${points}" fill="none" stroke="${COLORS[s.name]}" stroke-width="2.6" stroke-linejoin="round"/>`;if(!histogram) s.values.forEach((v,i)=>out+=`<circle cx="${x(i)}" cy="${y(v)}" r="2.5" fill="${COLORS[s.name]}"><title>${s.name}: ${num(v,2)}</title></circle>`)});
  out+='</svg><div class="legend">'+series.map(s=>`<span><i style="background:${COLORS[s.name]}"></i>${s.name}</span>`).join('')+'</div>';
  return out;
}

function drawCharts() {
  const steps=availableSteps();
  $('count-chart').innerHTML=svgChart(KERNELS.map(name=>({name,values:steps.map(s=>run(name).steps[String(s)]?.stats.count||0)})),
    {minimum:0,maximum:Math.max(...KERNELS.flatMap(k=>steps.map(s=>run(k).steps[String(s)]?.stats.count||0)))*1.05,xLabels:[[0,`${steps[0]/1000}k`],[steps.length-1,`${steps.at(-1)/1000}k`]]});
  const hist=(field,edges)=>KERNELS.map(name=>{const values=selectedStats(name)?.[field]||[];const total=values.reduce((a,b)=>a+b,0)||1;return {name,values:values.map(v=>100*v/total)}});
  $('scale-chart').innerHTML=svgChart(hist('scaleHistogram',data.scaleEdges),{minimum:0,yDigits:0,histogram:true,xLabels:[[0,'10⁻⁶'],[15,'10⁻²'],[31,'10²']]});
  $('opacity-chart').innerHTML=svgChart(hist('opacityHistogram',data.opacityEdges),{minimum:0,yDigits:0,histogram:true,xLabels:[[0,'0'],[15,'0.5'],[31,'1.0']]});
  const curveSteps=Object.keys(run('gaussian').curve).map(Number).filter(s=>KERNELS.every(k=>run(k).curve[String(s)])).sort((a,b)=>a-b);
  $('psnr-chart').innerHTML=svgChart(KERNELS.map(name=>({name,values:curveSteps.map(s=>run(name).curve[String(s)].psnr)})),
    {yDigits:1,xLabels:[[0,'0'],[curveSteps.length-1,`${curveSteps.at(-1)/1000}k`]]});
}

function draw() { drawTimeline();drawImages();drawStats();drawCharts(); }
function mean(values){const valid=values.filter(Number.isFinite);return valid.length?valid.reduce((a,b)=>a+b,0)/valid.length:null}
function metric(run,key){return run.summary[key] == null ? null : Number(run.summary[key])}
function deltaCell(value,betterPositive,digits){return `<span class="${value==null?'neutral':(value*betterPositive>0?'good':'bad')}">${signed(value,digits)}</span>`}

function matrixSummary(matrix, title, subtitle, id){
  let cards='<div class="summary-grid">';
  const sceneNames=Object.keys(matrix.scenes), total=sceneNames.reduce((n,s)=>n+matrix.scenes[s].seeds.length*3,0);
  cards+=`<article class="summary-card"><span class="eyebrow">COVERAGE</span><b>${sceneNames.length}</b><small>scenes · ${total} kernel/seed runs</small></article>`;
  cards+=`<article class="summary-card"><span class="eyebrow">KERNELS</span><b>3</b><small>Gaussian · linear · polynomial</small></article>`;
  cards+=`<article class="summary-card"><span class="eyebrow">DATA SOURCE</span><b class="source-name">${escapeHtml(matrix.sourceRoot || 'selected runs')}</b><small>generated from its kernels/ directory</small></article></div>`;
  for(const sceneName of sceneNames){
    const s=matrix.scenes[sceneName], seeds=s.seeds, base=seeds.map(seed=>s.runs.gaussian[String(seed)]);
    let rows='';
    for(const kernel of KERNELS){const runs=seeds.map(seed=>s.runs[kernel][String(seed)]);
      const vals={psnr:mean(runs.map(r=>metric(r,'mean_psnr_db'))),ssim:mean(runs.map(r=>metric(r,'mean_ssim'))),lpips:mean(runs.map(r=>metric(r,'mean_lpips')))};
      const paired={psnr:mean(seeds.map((seed,i)=>metric(runs[i],'mean_psnr_db')-metric(base[i],'mean_psnr_db'))),ssim:mean(seeds.map((seed,i)=>metric(runs[i],'mean_ssim')-metric(base[i],'mean_ssim'))),lpips:mean(seeds.map((seed,i)=>{const a=metric(runs[i],'mean_lpips'),b=metric(base[i],'mean_lpips');return a==null||b==null?NaN:a-b}))};
      rows+=`<tr><td><b>${kernel}</b></td><td>${num(vals.psnr)}</td><td>${num(vals.ssim,4)}</td><td>${num(vals.lpips,4)}</td><td>${kernel==='gaussian'?'—':deltaCell(paired.psnr,1,3)}</td><td>${kernel==='gaussian'?'—':deltaCell(paired.ssim,1,4)}</td><td>${kernel==='gaussian'?'—':deltaCell(paired.lpips,-1,4)}</td></tr>`;
    }
    cards+=`<article class="summary-card"><span class="eyebrow">${escapeHtml(sceneName.toUpperCase())} · ${seeds.length} PAIRED SEEDS</span><h2>${escapeHtml(sceneName[0].toUpperCase()+sceneName.slice(1))}</h2><p>Means across final test views and seeds; deltas are paired against Gaussian.</p><div class="table-wrap"><table><thead><tr><th>Kernel</th><th>PSNR dB</th><th>SSIM</th><th>LPIPS</th><th>ΔPSNR</th><th>ΔSSIM</th><th>ΔLPIPS</th></tr></thead><tbody>${rows}</tbody></table></div></article>`;
  }
  return `<section class="result-matrix" id="${id}"><div class="matrix-heading"><span class="eyebrow">${escapeHtml(subtitle)}</span><h2>${escapeHtml(title)}</h2><p>Final held-out test views · three paired seeds per scene</p></div>${cards}</section>`;
}

function drawSummary(){
  const synthetic=syntheticData?.scenes || {};
  let content='<nav class="result-jump" aria-label="Result matrices"><a href="#synthetic-results">Synthetic scenes</a><a href="#real-results">Real scenes</a><a href="analysis.html">Analysis & conclusion</a></nav>';
  content+='<div class="metric-key"><b>How to read the metrics</b><span><strong>PSNR ↑</strong> measures pixel error in decibels.</span><span><strong>SSIM ↑</strong> compares local structure, brightness, and contrast.</span><span><strong>LPIPS ↓</strong> compares learned image features.</span></div>';
  content+=Object.keys(synthetic).length ? matrixSummary(syntheticData,'NeRF-synthetic scenes','01 · 256 × 256 · 8,000 steps · 100k splat cap','synthetic-results') : '<p class="summary-note" id="synthetic-results">Synthetic results were not bundled. Rebuild with <code>--synthetic-root outputs/kernels</code> to include them.</p>';
  content+=matrixSummary(data,'Mip-NeRF 360 real scenes','02 · 390-pixel long edge · 20,000 steps · 200k splat cap','real-results');
  content+='<section class="result-conclusion" id="conclusion"><span class="eyebrow">ANALYSIS & CONCLUSION</span><h2>What do these differences mean?</h2><p>Read the separate interpretation of linear and polynomial results, training and rendering costs, validation behavior, and study limitations.</p><a href="analysis.html">Read the analysis ↗</a></section>';
  content+='<div class="summary-note">Test views are held out from training and validation. View examples and snapshot distributions are diagnostic; use the paired final-test values for the reported comparison. Timing and memory should only be compared on matched GPU hardware.</div>';
  $('summary-content').innerHTML=content;
}

async function start(){
  try{const response=await fetch('data.json');if(!response.ok)throw Error(`${response.status} ${response.statusText}`);data=await response.json()}
  catch(error){document.querySelector('main').innerHTML=`<div class="summary-card"><h2>Could not load site data</h2><p>Serve this folder with <code>python -m http.server</code> and open the local URL. ${escapeHtml(error.message)}</p></div>`;return}
  try{const response=await fetch('synthetic-results.json');if(response.ok)syntheticData=await response.json()}
  catch(error){console.warn('Synthetic results unavailable:',error)}
  const scenes=Object.keys(data.scenes);$('scene-select').innerHTML=scenes.map(s=>`<option value="${escapeHtml(s)}">${escapeHtml(s[0].toUpperCase()+s.slice(1))}</option>`).join('');
  scene=scenes.includes('bonsai')?'bonsai':scenes[0];$('scene-select').value=scene;
  seed=currentScene().seeds.includes(17)?17:currentScene().seeds[0];selectScene(scene);drawSummary();
  $('scene-select').onchange=e=>selectScene(e.target.value);
  $('seed-select').onchange=e=>{seed=Number(e.target.value);selectScene(scene)};
  $('view-select').onchange=e=>{view=e.target.value;drawImages()};
  $('step-range').oninput=e=>{step=availableSteps()[Number(e.target.value)];draw()};
  $('prev-step').onclick=()=>{const steps=availableSteps();step=steps[Math.max(0,steps.indexOf(step)-1)];draw()};
  $('next-step').onclick=()=>{const steps=availableSteps();step=steps[Math.min(steps.length-1,steps.indexOf(step)+1)];draw()};
  $('zoom').oninput=e=>{zoom=Number(e.target.value);$('zoom-value').textContent=`${zoom.toFixed(1)}×`;document.querySelectorAll('.image-viewport img').forEach(img=>img.style.transform=`scale(${zoom})`)};
  const showPage=page=>{document.querySelectorAll('.tab[data-page]').forEach(tab=>tab.classList.toggle('active',tab.dataset.page===page));$('explore-page').hidden=page!=='explore';$('summary-page').hidden=page!=='summary';};
  document.querySelectorAll('.tab[data-page]').forEach(button=>button.onclick=()=>{location.hash=button.dataset.page==='summary'?'summary':'explore';showPage(button.dataset.page)});
  const summaryHashes=['#summary','#synthetic-results','#real-results','#conclusion'];
  window.addEventListener('hashchange',()=>showPage(summaryHashes.includes(location.hash)?'summary':'explore'));
  showPage(summaryHashes.includes(location.hash)?'summary':'explore');
  document.addEventListener('keydown',e=>{if(['SELECT','INPUT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')$('prev-step').click();if(e.key==='ArrowRight')$('next-step').click()});
}
start();
