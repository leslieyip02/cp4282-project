// These constants match shared/kernels.py. d = sqrt(q), so d is the
// Mahalanobis distance measured in the projected ellipse's coordinates.
const HALF = Math.sqrt(2 * Math.log(2));
const LINEAR_RADIUS = Math.sqrt(8 * Math.log(2));
const QUADRATIC_RADIUS = Math.sqrt(2 * Math.log(2) / (1 - 1 / Math.sqrt(2)));
const MAX_D = 3.15;
const kernels = [
  {key:'gaussian', name:'Gaussian', color:'#586a65', radius:Infinity,
    weight:d=>Math.exp(-d*d/2), caption:'Soft tail; practical renderer support is capped.'},
  {key:'linear', name:'Linear radial', color:'#377bb5', radius:LINEAR_RADIUS,
    weight:d=>Math.max(0,1-d/LINEAR_RADIUS), caption:'Straight radial falloff; exact finite edge.'},
  {key:'quadratic', name:'Quadratic', color:'#c87139', radius:QUADRATIC_RADIUS,
    weight:d=>Math.max(0,1-d*d/(QUADRATIC_RADIUS**2))**2, caption:'Rounded center and smooth finite edge.'}
];
const $=id=>document.getElementById(id);
const escapeText=value=>String(value).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const footprintCache=new Map();

function createCards(){
  for(const [grid,type] of [['footprint-grid','footprint'],['surface-grid','surface']]){
    $(grid).innerHTML=kernels.map(k=>`<article class="viz-card"><header><span>${k.name}</span><small>${type==='footprint'?'x–y plane':'x–y–weight surface'}</small></header><canvas id="${type}-${k.key}" role="img" aria-label="${k.name} ${type==='footprint'?'2D footprint heatmap':'3D weight surface'}"></canvas><div class="caption">${k.caption}</div></article>`).join('');
  }
}

function color(hex,amount){
  const rgb=[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));
  const t=Math.pow(amount,.66);
  return rgb.map(v=>Math.round(249+(v-249)*t));
}
function canvasSetup(canvas,width,height){
  const ratio=Math.min(window.devicePixelRatio||1,2);
  canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
  const ctx=canvas.getContext('2d');
  ctx.setTransform(ratio,0,0,ratio,0,0);
  return ctx;
}

function drawFootprint(kernel,d){
  const canvas=$('footprint-'+kernel.key),w=360,h=265,ctx=canvasSetup(canvas,w,h);
  const scale=39,cx=w/2,cy=h/2;
  if(!footprintCache.has(kernel.key)){
    const bitmap=document.createElement('canvas');
    bitmap.width=w;bitmap.height=h;
    const bitmapContext=bitmap.getContext('2d');
    const image=bitmapContext.createImageData(w,h);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){
      const distance=Math.hypot((x-cx)/scale,(y-cy)/scale);
      const [r,g,b]=color(kernel.color,kernel.weight(distance));
      const i=(y*w+x)*4;image.data[i]=r;image.data[i+1]=g;image.data[i+2]=b;image.data[i+3]=255;
    }
    bitmapContext.putImageData(image,0,0);
    footprintCache.set(kernel.key,bitmap);
  }
  // drawImage obeys the HiDPI transform; putImageData does not.
  ctx.drawImage(footprintCache.get(kernel.key),0,0,w,h);
  ctx.strokeStyle='rgba(29,62,45,.15)';ctx.lineWidth=1;
  for(const ring of [1,2,3]){ctx.beginPath();ctx.arc(cx,cy,ring*scale,0,Math.PI*2);ctx.stroke()}
  ctx.strokeStyle='rgba(31,82,58,.75)';ctx.setLineDash([5,4]);ctx.beginPath();ctx.arc(cx,cy,HALF*scale,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#243c30';ctx.beginPath();ctx.arc(cx,cy,3,0,Math.PI*2);ctx.fill();
  const mx=cx+d*scale;
  ctx.fillStyle='#fff';ctx.strokeStyle='#233b2e';ctx.lineWidth=2;ctx.beginPath();ctx.arc(mx,cy,5,0,Math.PI*2);ctx.fill();ctx.stroke();
  ctx.fillStyle='#3d5848';ctx.font='11px system-ui';ctx.fillText('half-height ring',12,22);
  ctx.fillText('d = 0',cx-15,cy+18);
}

function drawSurface(kernel,d){
  const canvas=$('surface-'+kernel.key),w=360,h=265,ctx=canvasSetup(canvas,w,h);
  ctx.fillStyle='#f8faf7';ctx.fillRect(0,0,w,h);
  const project=(x,y,z)=>[w/2+(x-y)*31,194+(x+y)*13-z*128];
  const corners=[[-3,-3],[3,-3],[3,3],[-3,3]].map(([x,y])=>project(x,y,0));
  ctx.fillStyle='#eef3ed';ctx.strokeStyle='#becfc1';ctx.lineWidth=1;
  ctx.beginPath();corners.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fill();ctx.stroke();
  const n=38,min=-3,max=3,step=(max-min)/n;
  const cells=[];
  for(let i=0;i<n;i++)for(let j=0;j<n;j++)cells.push({i,j,depth:i+j});
  cells.sort((a,b)=>a.depth-b.depth);
  for(const {i,j} of cells){
    const x=min+i*step,y=min+j*step;
    const pts=[[x,y],[x+step,y],[x+step,y+step],[x,y+step]].map(([a,b])=>project(a,b,kernel.weight(Math.hypot(a,b))));
    const v=kernel.weight(Math.hypot(x+step/2,y+step/2));
    const [r,g,b]=color(kernel.color,Math.max(.13,v*.88));
    ctx.fillStyle=`rgb(${r},${g},${b})`;
    ctx.strokeStyle='rgba(52,79,64,.09)';ctx.lineWidth=.45;
    ctx.beginPath();pts.forEach(([px,py],m)=>m?ctx.lineTo(px,py):ctx.moveTo(px,py));ctx.closePath();ctx.fill();ctx.stroke();
  }
  const [px,py]=project(d,0,kernel.weight(d));
  ctx.fillStyle='#fff';ctx.strokeStyle='#203b2d';ctx.lineWidth=2;ctx.beginPath();ctx.arc(px,py,4.5,0,Math.PI*2);ctx.fill();ctx.stroke();
  ctx.fillStyle='#526758';ctx.font='11px system-ui';ctx.fillText('height = weight',12,21);
  ctx.fillText('x',312,212);ctx.fillText('y',32,211);
}

function drawProfile(d){
  const svg=$('profile'),W=900,H=294,L=46,R=24,T=18,B=41;
  const x=v=>L+(W-L-R)*v/MAX_D,y=v=>T+(H-T-B)*(1-v);
  let html=`<title>Kernel weight by elliptical distance</title><desc>The three curves start at one and share half height at distance ${HALF.toFixed(3)}. Linear and quadratic reach zero at finite radii.</desc>`;
  for(const value of [0,.25,.5,.75,1]){
    html+=`<line x1="${L}" y1="${y(value)}" x2="${W-R}" y2="${y(value)}" stroke="#e5ebe5"/><text x="${L-9}" y="${y(value)+4}" text-anchor="end" fill="#718175" font-size="12">${value.toFixed(value===0||value===1?0:2)}</text>`;
  }
  for(const value of [0,1,2,3])html+=`<line x1="${x(value)}" y1="${H-B}" x2="${x(value)}" y2="${H-B+5}" stroke="#89998c"/><text x="${x(value)}" y="${H-13}" text-anchor="middle" fill="#718175" font-size="12">${value}</text>`;
  html+=`<line x1="${L}" y1="${y(.5)}" x2="${W-R}" y2="${y(.5)}" stroke="#7f9c87" stroke-dasharray="5 5"/><line x1="${x(HALF)}" y1="${T}" x2="${x(HALF)}" y2="${H-B}" stroke="#7f9c87" stroke-dasharray="5 5"/><text x="${x(HALF)+8}" y="${T+15}" fill="#4f7660" font-size="12" font-weight="700">shared 0.5</text>`;
  for(const [edge,tint] of [[QUADRATIC_RADIUS,'#c87139'],[LINEAR_RADIUS,'#377bb5']])html+=`<line x1="${x(edge)}" y1="${T}" x2="${x(edge)}" y2="${H-B}" stroke="${tint}" opacity=".42" stroke-dasharray="3 5"/>`;
  for(const kernel of kernels){
    const points=Array.from({length:251},(_,i)=>{const v=MAX_D*i/250;return `${x(v).toFixed(1)},${y(kernel.weight(v)).toFixed(1)}`}).join(' ');
    html+=`<polyline points="${points}" fill="none" stroke="${kernel.color}" stroke-width="3.4" stroke-linejoin="round"/>`;
    html+=`<circle cx="${x(d)}" cy="${y(kernel.weight(d))}" r="5.5" fill="${kernel.color}" stroke="#fff" stroke-width="2"/>`;
  }
  html+=`<line x1="${x(d)}" y1="${T}" x2="${x(d)}" y2="${H-B}" stroke="#253d30" stroke-dasharray="3 4" opacity=".65"/><text x="${(L+W-R)/2}" y="${H-1}" text-anchor="middle" fill="#526659" font-size="13">elliptical distance d = √q</text><text x="${L}" y="11" fill="#526659" font-size="12">weight</text>`;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=html;
}

function update(){
  const d=Number($('distance').value);
  $('distance-value').textContent=d.toFixed(3);
  $('sample-weights').innerHTML=kernels.map(k=>`<span>${escapeText(k.name)} <strong>${k.weight(d).toFixed(3)}</strong></span>`).join('');
  drawProfile(d);
  kernels.forEach(k=>{drawFootprint(k,d);drawSurface(k,d)});
}
createCards();
$('distance').addEventListener('input',update);
update();
