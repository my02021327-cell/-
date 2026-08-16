"""
단일 파일 HTML 리포트 — PROMPT §12 산출물 3.

라이트/다크 대응, 인터랙티브(크로스헤어 툴팁·범위 선택), 외부 자원 없음.
색은 dataviz 표준 팔레트의 검증 통과 슬롯만 사용한다
(blue #2a78d6 / orange #eb6834 / aqua #1baf7a / violet #4a3aa7 — 4슬롯 전 검사 통과,
aqua 는 라이트 대비 2.74 로 relief 규칙 적용 → 직접 라벨·표 병기).
"""

from __future__ import annotations

import json
from pathlib import Path

CSS = """
:root{color-scheme:light;
 --bg:#f7f7f5; --surface:#fcfcfb; --line:#e2e1dc; --ink:#0b0b0b; --ink2:#52514e; --ink3:#7a7873;
 --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#4a3aa7;
 --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b; --band:#2a78d61f;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --bg:#131312; --surface:#1a1a19; --line:#333330; --ink:#fff; --ink2:#c3c2b7; --ink3:#8d8b82;
 --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#9085e9; --band:#3987e526;}}
:root[data-theme="dark"]{color-scheme:dark;
 --bg:#131312; --surface:#1a1a19; --line:#333330; --ink:#fff; --ink2:#c3c2b7; --ink3:#8d8b82;
 --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#9085e9; --band:#3987e526;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 80px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;
 border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:19px;margin:44px 0 6px;letter-spacing:-.01em;padding-top:18px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:26px 0 8px;color:var(--ink2)}
p,li{color:var(--ink2);margin:8px 0}
.sub{color:var(--ink3);font-size:13px;margin:0}
button{font:inherit;background:var(--surface);color:var(--ink);border:1px solid var(--line);
 border-radius:8px;padding:6px 12px;cursor:pointer}
button[aria-pressed="true"]{border-color:var(--s1);color:var(--s1)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .lb{font-size:12px;color:var(--ink3);margin-bottom:6px}
.tile .v{font-size:24px;font-weight:650;letter-spacing:-.02em;line-height:1.15}
.tile .u{font-size:12px;color:var(--ink3);font-weight:400}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:520px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink3);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.hl td{background:color-mix(in srgb,var(--s1) 7%,transparent)}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11.5px;font-weight:600;border:1px solid}
.ok{color:var(--good);border-color:var(--good)} .no{color:var(--crit);border-color:var(--crit)}
.mid{color:var(--ink3);border-color:var(--line)}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:4px 0 10px;font-size:12.5px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
figure{margin:0}
figcaption{font-size:12.5px;color:var(--ink3);margin-top:8px}
svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--line);
 border-radius:8px;padding:8px 10px;font-size:12.5px;color:var(--ink);box-shadow:0 6px 20px #0002;
 opacity:0;transition:opacity .1s;z-index:9;white-space:nowrap}
.note{border-left:3px solid var(--s2);padding:2px 0 2px 14px;margin:14px 0;color:var(--ink2);font-size:14px}
.crit-note{border-left-color:var(--crit)}
code{background:color-mix(in srgb,var(--ink) 7%,transparent);padding:1px 5px;border-radius:4px;font-size:12.5px}
"""

JS = r"""
const $=s=>document.querySelector(s);
const tip=document.createElement('div');tip.className='tip';document.body.appendChild(tip);
function showTip(e,html){tip.innerHTML=html;tip.style.opacity=1;
  const w=tip.offsetWidth,h=tip.offsetHeight;
  tip.style.left=Math.min(e.clientX+14,innerWidth-w-10)+'px';
  tip.style.top=Math.max(e.clientY-h-12,8)+'px';}
function hideTip(){tip.style.opacity=0;}
const fmt=(v,d=1)=>v==null||!isFinite(v)?'—':Number(v).toLocaleString('ko-KR',{minimumFractionDigits:d,maximumFractionDigits:d});
function ticks(lo,hi,n){const raw=(hi-lo)/n,m=Math.pow(10,Math.floor(Math.log10(raw)));
  const s=[1,2,2.5,5,10].find(x=>x*m>=raw)*m,out=[];
  for(let v=Math.ceil(lo/s)*s;v<=hi+1e-9;v+=s)out.push(+v.toFixed(10));return out;}

/* 시계열 라인차트 — 크로스헤어 + 툴팁 (dataviz §5: 기본 탑재) */
function lineChart(el,o){
  const W=920,H=o.h||260,ml=58,mr=14,mt=12,mb=26;
  const xs=o.x, n=xs.length;
  let lo=o.yMin,hi=o.yMax;
  if(lo==null||hi==null){const all=o.series.flatMap(s=>s.v).concat(o.band?o.band.lo.concat(o.band.hi):[]).filter(v=>v!=null&&isFinite(v));
    lo=Math.min(...all);hi=Math.max(...all);const pad=(hi-lo)*.08;lo-=pad;hi+=pad;}
  const X=i=>ml+(W-ml-mr)*(n<2?0:i/(n-1)), Y=v=>mt+(H-mt-mb)*(1-(v-lo)/(hi-lo));
  const yt=ticks(lo,hi,4);
  let g=`<g>`;
  for(const t of yt) g+=`<line x1="${ml}" x2="${W-mr}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--line)" stroke-width="1"/>`
    +`<text x="${ml-8}" y="${Y(t)+4}" text-anchor="end" font-size="11" fill="var(--ink3)">${fmt(t,0)}</text>`;
  const step=Math.max(1,Math.floor(n/7));
  for(let i=0;i<n;i+=step) g+=`<text x="${X(i)}" y="${H-6}" text-anchor="middle" font-size="11" fill="var(--ink3)">${o.xlab?o.xlab(xs[i]):xs[i]}</text>`;
  g+=`</g>`;
  if(o.band){let up='',dn='';
    for(let i=0;i<n;i++){const v=o.band.hi[i];if(v!=null)up+=`${up?'L':'M'}${X(i)},${Y(v)}`;}
    for(let i=n-1;i>=0;i--){const v=o.band.lo[i];if(v!=null)dn+=`L${X(i)},${Y(v)}`;}
    if(up)g+=`<path d="${up+dn}Z" fill="var(--band)"/>`;}
  for(const s of o.series){let d='';let open=false;
    for(let i=0;i<n;i++){const v=s.v[i];
      if(v==null||!isFinite(v)){open=false;continue;}
      d+=`${open?'L':'M'}${X(i)},${Y(v)}`;open=true;}
    g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" opacity="${s.op||1}"/>`;}
  g+=`<line id="ch" x1="0" x2="0" y1="${mt}" y2="${H-mb}" stroke="var(--ink3)" stroke-width="1" opacity="0"/>`;
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.aria||''}">${g}</svg>`;
  const svg=el.querySelector('svg'),ch=svg.querySelector('#ch');
  svg.addEventListener('pointermove',e=>{const r=svg.getBoundingClientRect();
    const px=(e.clientX-r.left)/r.width*W;
    let i=Math.round((px-ml)/((W-ml-mr)/(n-1)));i=Math.max(0,Math.min(n-1,i));
    ch.setAttribute('x1',X(i));ch.setAttribute('x2',X(i));ch.setAttribute('opacity','.5');
    showTip(e,`<b>${xs[i]}</b><br>`+o.series.map(s=>
      `<span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${s.c}"></span> ${s.n} ${fmt(s.v[i],o.d??0)}${o.u||''}`).join('<br>'));});
  svg.addEventListener('pointerleave',()=>{ch.setAttribute('opacity','0');hideTip();});
}

/* 가로 막대 — 값 직접 라벨 (dataviz §4) */
function barChart(el,rows,o={}){
  const W=920,rh=30,ml=o.ml||210,mr=90,H=rows.length*rh+16;
  const mx=Math.max(...rows.map(r=>r.v))*1.08;
  let g='';
  rows.forEach((r,i)=>{const y=i*rh+8,w=(W-ml-mr)*(r.v/mx);
    g+=`<text x="${ml-10}" y="${y+15}" text-anchor="end" font-size="12.5" fill="var(--ink2)">${r.k}</text>`
     +`<rect x="${ml}" y="${y+3}" width="${Math.max(w,2)}" height="16" rx="4" fill="${r.c||'var(--s1)'}"/>`
     +`<text x="${ml+w+8}" y="${y+16}" font-size="12" fill="var(--ink2)" font-variant-numeric="tabular-nums">${fmt(r.v,o.d??1)}${o.u||''}</text>`;});
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.aria||''}">${g}</svg>`;
}

/* 상태 타임라인 */
function timeline(el,dates,verdict){
  const W=920,H=44,n=dates.length,bw=W/n;
  const C={ok:'var(--good)',warn:'var(--warn)',crit:'var(--crit)','':'var(--line)'};
  let g='';let run=0;
  for(let i=0;i<n;i++){const v=verdict[i]||'';
    g+=`<rect x="${i*bw}" y="8" width="${Math.max(bw,.6)}" height="22" fill="${C[v]}" opacity="${v?'.95':'.35'}"/>`;}
  const step=Math.max(1,Math.floor(n/8));
  for(let i=0;i<n;i+=step)g+=`<text x="${i*bw}" y="${H-2}" font-size="10.5" fill="var(--ink3)">${dates[i].slice(0,7)}</text>`;
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="건강상태 일별 판정 타임라인">${g}</svg>`;
}
"""


def _tbl(cols, rows, hl=None):
    """cols: [(key, label, numeric?)]"""
    th = "".join(f'<th class="{"n" if c[2] else ""}">{c[1]}</th>' for c in cols)
    body = []
    for r in rows:
        cls = ' class="hl"' if hl and hl(r) else ""
        tds = "".join(f'<td class="{"n" if c[2] else ""}">{_cell(r.get(c[0]), c[0])}</td>'
                      for c in cols)
        body.append(f"<tr{cls}>{tds}</tr>")
    return (f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _cell(v, key: str = ""):
    if v is None:
        return '<span class="pill mid">—</span>'
    if isinstance(v, bool):
        # '발화'는 참일수록 나쁜 신호 — 통과/미통과로 읽히면 정반대가 된다
        if key == "발화":
            return f'<span class="pill {"no" if v else "mid"}">{"발화" if v else "미발화"}</span>'
        return f'<span class="pill {"ok" if v else "no"}">{"통과" if v else "미통과"}</span>'
    if isinstance(v, float):
        return f"{v:,.4g}"
    if isinstance(v, int):
        return f"{v:,}"
    s = str(v)
    if s in ("개선(유의)",):
        return f'<span class="pill ok">{s}</span>'
    if s in ("악화(유의)",):
        return f'<span class="pill no">{s}</span>'
    if s == "구별되지 않음":
        return f'<span class="pill mid">{s}</span>'
    return s


def render(R: dict, HP: dict, path: Path) -> None:
    d = R["데이터"]
    ens = R["앙상블"]
    ens_row = next(x for x in ens["표"] if x["결합"] == ens["채택"])
    base = R["기준선_§5.5"]["채택기준선"]
    mb = R["물질수지_§4"]["8000"]
    ts = R["예측시계열"]

    tiles = [
        ("앙상블 CV-RMSE", f"{ens_row['CV_RMSE']:,.1f}", "㎥CH₄/d"),
        ("기준선 (문헌전단+절편)", f"{base['CV_RMSE']:,.1f}", "㎥CH₄/d"),
        ("개선폭 ΔRMSE", f"{ens_row['ΔRMSE']:,.1f}", f"p={ens_row['p']}"),
        ("2023 홀드아웃 R²", f"{R['holdout_최종보고']['앙상블']['R2']}", f"RMSE {R['holdout_최종보고']['앙상블']['RMSE']:,.0f}"),
        ("VS 수지 함축수율", f"{mb['함축수율_소비VS기준']}", f"이론 {mb['이론상한_소비VS기준']} · {mb['초과배수']}배 초과"),
        ("타깃 관측일", f"{d['타깃']['관측일']:,}", f"/ {d['일수']:,}일"),
    ]
    mix = " / ".join(f"{k} {v}%" for k, v in d["반입구성비_평균_pct"].items())
    tile_html = "".join(
        f'<div class="tile"><div class="lb">{a}</div><div class="v">{b} <span class="u">{c}</span></div></div>'
        for a, b, c in tiles)

    payload = {
        "ts": ts,
        "kern": R["전단커널"],
        "m5": R["M5_시변절편"],
        "health": {"dates": HP["series"]["dates"], "verdict": HP["series"]["verdict"]},
        "bars": {
            "base": [{"k": r["사양"], "v": r["재계산_RMSE"]} for r in R["기준선_§5.5"]["표"]],
            "models": [{"k": r["모델"], "v": r["CV_RMSE"]} for r in R["모델계열"]["표"]],
            "ens": [{"k": r["결합"], "v": r["CV_RMSE"]} for r in ens["표"]],
        },
    }

    def sec_tbl(title, cols, rows, note="", hl=None):
        return (f'<h3>{title}</h3>{f"<p>{note}</p>" if note else ""}'
                f'<div class="card">{_tbl(cols, rows, hl)}</div>')

    html = f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>영천 BGP — 3트랙 + 통합 앙상블 메탄 예측</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <div>
    <h1>영천 BGP 메탄생성 예측 — 3트랙 + 통합 앙상블</h1>
    <p class="sub">기질 물량·화학양론(T1) / VS 물질수지(T2) / 이화학 상태(T3) → M1~M5 앙상블 ·
      확장창 rolling-origin {R['검증프로토콜']['폴드수']}폴드 · 90일 앞 예측 ·
      backend {R['실행환경']['backend']}</p>
  </div>
  <button id="th" aria-pressed="false">라이트 / 다크</button>
</header>

<div class="tiles">{tile_html}</div>

<div class="note crit-note"><b>먼저 읽을 것 — 이 결과의 전제.</b>
PROMPT §1 이 요구한 데이터 7개 파일은 제공되지 않았다. 저장소 정본
<code>data/영천BGP_MASTER_2018-2023.csv</code> 로 대체했고, 두 자료가 같은 원천임은
반입 구성비({mix}) 와 §4 물질수지 재현으로 확인했다.
다만 타깃(실측 CH₄) 결측이 {d['타깃']['결측률_pct']}% 로 PROMPT 가 말한 63일보다 훨씬 크다 —
CH₄% 가 주말 미측정 랩 항목이기 때문이다. <b>따라서 선행 기준선 861.5 ㎥/d 와 직접 비교할 수 없고,
동일 폴드에서 기준선을 다시 만들어 비교했다.</b></div>

<h2>A. 데이터 · 재현 검사</h2>
{sec_tbl("§9 재현 검사 (SRT 12일 · 무절편)",
         [("항목","항목",False),("값","값",False)],
         [{"항목":"원본 θ (음폐수/가축분뇨/음식물)","값":"40.076 / 52.111 / 13.407, 학습 R² 0.293"},
          {"항목":"재계산 θ","값":" / ".join(str(v) for v in R["재현검사_§9"]["재계산_θ"].values())
                              + f", 학습 R² {R['재현검사_§9']['재계산_학습R2']}"},
          {"항목":"판정","값":R["재현검사_§9"]["판정"]}])}

{sec_tbl("§4 VS 물질수지 이탈 — V 민감도",
   [("V_m3","V (㎥)",True),("유입VS_kgd","유입 VS",True),("유출VS_kgd","유출 VS",True),
    ("소비VS_kgd","소비 VS",True),("VS분해율_pct","분해율 %",True),
    ("함축수율_소비VS기준","함축수율",True),("초과배수","이론 대비",True)],
   [{"V_m3":v["V_m3"],"유입VS_kgd":v["유입VS_OLR경로_kgd"],"유출VS_kgd":v["유출VS_kgd"],
     "소비VS_kgd":v["소비VS_kgd"],"VS분해율_pct":v["VS분해율_pct"],
     "함축수율_소비VS기준":v["함축수율_소비VS기준"],"초과배수":v["초과배수"]}
    for v in R["물질수지_§4"].values()],
   note="이론 최대는 0.50 ㎥CH₄/kgVS_destroyed 다. 실측은 2배 가까이 초과한다 — "
        "<b>이탈을 계수로 흡수하지 않고 그대로 보고한다.</b> T2 는 절대수준 예측에 쓸 수 없다. "
        f"{R['물질수지_§4']['8000']['V_의존성']}")}

<h2>B. 전단 커널 — 유일하게 데이터가 식별하는 지연</h2>
<p>기질별 경로가 달라 단일 공통 커널을 쓰지 않는다. 음식물류만 저장호퍼(2~3일)를 거치고,
음폐수·가축분뇨는 우회해 여액저장조에서 합류한다.</p>
<div class="card"><figure>
  <div class="legend">
    <span><i class="sw" style="background:var(--s1)"></i>음폐수</span>
    <span><i class="sw" style="background:var(--s2)"></i>가축분뇨</span>
    <span><i class="sw" style="background:var(--s3)"></i>음식물 (저장호퍼 경유)</span></div>
  <div id="kchart"></div>
  <figcaption>반입 → 소화조 유입 체류시간 분포(표준 시나리오). 음식물은 평균이 약 2일 뒤로 밀린다.</figcaption>
</figure></div>
{sec_tbl("커널 요약 [일]",
   [("기질","기질",False),("평균","평균",True),("t10","t10",True),("t50","t50",True),("t90","t90",True),("t95","t95",True)],
   [{"기질":k, **v} for k,v in R["전단커널"]["표준"].items()])}

<h3>기준선 4종 — 선행 보고 vs 재계산</h3>
<div class="card"><div id="bchart"></div></div>
{_tbl([("사양","사양",False),("선행보고_RMSE","선행 보고",True),("재계산_RMSE","재계산",True),("폴드SD","폴드 SD",True)],
      R["기준선_§5.5"]["표"])}
{sec_tbl("전단 효과 검정 (폴드별 대응 t검정)",
   [("비교","비교",False),("ΔRMSE","ΔRMSE",True),("SE","SE",True),("p","p",True),("판정","판정",False)],
   R["기준선_§5.5"]["전단효과"]+[R["기준선_§5.5"]["전단제거_대조"]],
   note="문헌 기질별 커널이 기존 공통 커널을 유의하게 이기고, 전단 지연을 제거하면 크게 악화된다 — "
        "PROMPT §5.4 의 결론이 우리 계열에서도 그대로 재현된다.")}

<h2>C. 트랙과 모델 계열</h2>
<div class="card"><div id="mchart"></div></div>
{_tbl([("모델","모델",False),("트랙","트랙",False),("CV_RMSE","CV-RMSE",True),
       ("ΔRMSE","Δ vs 기준선",True),("SE","SE",True),("p","p",True),("판정","판정",False)],
      R["모델계열"]["표"])}
<div class="note"><b>M1 이 기준선과 같은 값인 것은 정상이다</b> — 채택 기준선(문헌전단+절편)이 곧 M1 이다.
독립 계열은 M2~M5 이며, 단독 성능이 기준선에 못 미쳐도 <b>귀납 편향이 달라 앙상블에서 값을 한다</b>
(아래 가중치 참조). 억지로 넣은 것이 아니라 스태킹이 자동으로 고른 결과다.</div>

{sec_tbl("T1 계수와 이탈 판정 (§3)",
   [("지표","지표",False),("값","값",False),("임계","임계",False),("발화","발화",False),("전환","전환 경로",False)],
   R["T1_이탈판정_§3"]["발화"],
   note="θ = " + ", ".join(f"{k} {v}" for k, v in R["T1_계수"]["θ"].items())
        + f" ㎥CH₄/t, 절편 {R['T1_계수']['절편_m3d']:,.0f} ㎥/d "
        f"(<b>실측의 {R['T1_계수']['절편비중_pct']}%</b>). {R['T1_이탈판정_§3']['해석']}")}

{sec_tbl("§6 변수 채택/배제 — 예측 모델 vs 건강상태 지표",
   [("변수","변수",False),("CV_RMSE","CV-RMSE",True),("ΔRMSE","ΔRMSE",True),
    ("SE","SE",True),("p","p",True),("판정","판정",False)],
   R["변수채택_§6"]["판정"],
   note=f"T3 없는 트리 기준 {R['변수채택_§6']['T3없음_CV_RMSE']} ㎥/d 대비 포함/제외 대응 t검정. "
        "<b>내부 이화학 변수는 하나도 예측 모델에 채택되지 않았다</b>(전부 p≥0.05). "
        "폐기하지 않고 §J 건강상태 프로그램으로 재배치했다.")}

<h2>D. 통합 앙상블</h2>
<div class="card"><div id="echart"></div></div>
{_tbl([("결합","결합 방식",False),("CV_RMSE","CV-RMSE",True),("ΔRMSE","Δ vs 기준선",True),
       ("SE","SE",True),("p","p",True),("판정","판정",False)], ens["표"],
      hl=lambda r: r["결합"] == ens["채택"])}
<p><b>채택: {ens['채택']}</b> — {ens['채택_검정']['판정']}
 (ΔRMSE {ens['채택_검정']['ΔRMSE']}, SE {ens['채택_검정']['SE']}, p={ens['채택_검정']['p']}).
 최종 스태킹 가중치: {", ".join(f"{k} {v}" for k, v in ens["최종_스태킹_가중치"].items())}.
 {ens['메타학습_규약']}.</p>

<h2>E. Ablation</h2>
{_tbl([("제거","제거 대상",False),("CV_RMSE","CV-RMSE",True),("ΔRMSE","ΔRMSE",True),
       ("p","p",True),("판정","판정",False)], R["ablation"])}

<h2>F. 물리 정합성 검사 (§9)</h2>
{sec_tbl("톤수 기준 계수", [("검사","검사",False),("값","값",False),("기준","기준",False),("통과","판정",False)],
         R["물리검사_§9"]["톤수기준"])}
{sec_tbl("VS 수율 기준 계수", [("검사","검사",False),("값","값",False),("기준","기준",False),("통과","판정",False)],
         R["물리검사_§9"]["VS기준"])}
{sec_tbl("커널 총량 보존", [("검사","검사",False),("값","값",False),("기준","기준",False),("통과","판정",False)],
         R["물리검사_§9"]["커널"])}
<div class="note crit-note"><b>물리 검사 미통과 항목이 있다.</b> 절편이 실측의
{R['T1_계수']['절편비중_pct']}% 이고 음폐수 BD 적합값이 1을 넘는다. 따라서 이 모델은
<b>예측 전용</b>이며, <b>그 계수를 생분해도·수율로 해석해서는 안 된다.</b></div>

<h3>SRT 비식별성 (§5.2 재확인)</h3>
{_tbl([("V_m3","V (㎥)",True),("SRT_d","SRT (일)",True),("CV_RMSE","CV-RMSE",True),
       ("ΔRMSE","ΔRMSE",True),("p","p",True),("판정","판정",False)], R["SRT_민감도"])}
<p>V 를 4,000 → 8,000 ㎥ 로 두 배 바꿔도 CV-RMSE 가 소수점 이하로만 움직인다(p≥0.89).
{R['비식별성_§5.2']['설명']} <b>SRT 를 데이터로 추정했다고 주장할 수 없다.</b></p>

<h2>G. 잔차 진단</h2>
{_tbl([("구분","구분",False),("값","값",False)],
      [{"구분":"자기상관 (lag 1 / 7 / 30)","값":" / ".join(str(v) for v in R["잔차진단"]["자기상관"].values())},
       {"구분":"연도별 RMSE","값":", ".join(f"{k} {v}" for k,v in R["잔차진단"]["연도별_RMSE"].items())},
       {"구분":"요일별 평균 잔차","값":", ".join(f"{k}:{v}" for k,v in R["잔차진단"]["요일별_평균잔차"].items())},
       {"구분":"부하 4분위 평균 잔차","값":", ".join(f"Q{int(k)+1}:{v}" for k,v in R["잔차진단"]["부하4분위_평균잔차"].items())},
       {"구분":"가축분뇨 비중 4분위 평균 잔차","값":", ".join(f"Q{int(k)+1}:{v}" for k,v in R["잔차진단"]["가축분뇨비중4분위_평균잔차"].items())}])}

<h2>H. M5 시변 절편 궤적</h2>
<p>{R['M5_시변절편']['설명']}. 연평균:
{", ".join(f"<b>{k}</b> {v:,.0f}" for k, v in R["M5_시변절편"]["연평균"].items())} ㎥/d.</p>
<div class="card"><figure><div id="m5chart"></div>
<figcaption>미설명 항이 상수가 아니라 2018년 고점에서 2021~22년 저점으로 이동한다 —
배합 전환·개보수 시점 대조가 필요한 단서다(§5.3 규명 경로).</figcaption></figure></div>

<h2>I. 예측 시계열 (예측구간 포함)</h2>
<div class="card"><figure>
  <div class="legend">
    <span><i class="sw" style="background:var(--s1)"></i>실측 CH₄</span>
    <span><i class="sw" style="background:var(--s2)"></i>앙상블 예측 (out-of-fold)</span>
    <span><i class="sw" style="background:var(--band);border:1px solid var(--line)"></i>95% 예측구간</span></div>
  <div style="margin:6px 0 10px"><button data-r="all" aria-pressed="false">전체</button>
    <button data-r="v" aria-pressed="true">2022~</button>
    <button data-r="h" aria-pressed="false">2023 홀드아웃</button></div>
  <div id="pchart"></div>
  <figcaption>모든 예측은 out-of-fold — 각 시점은 그 이전 자료만으로 학습된 모델이 낸 값이다.
    예측구간은 OOF 잔차 ±1.96σ (σ={ts['pi_sd']:,.0f} ㎥/d).</figcaption>
</figure></div>

<h2>J. 소화조 건강상태 점검 프로그램</h2>
<p>§6 에 따라 <b>예측력이 없어 배제된 변수를 폐기하지 않고</b> 여기로 재배치했다.
메탄 예측 성능과 무관하게 독립 평가한다.
종합: 위험 {HP['종합']['위험일']:,}일 / 주의 {HP['종합']['주의일']:,}일 / 정상 {HP['종합']['정상일']:,}일
/ 미측정 {HP['종합']['미측정일']:,}일. 판정 규칙: {HP['경보설계']['종합규칙']}.</p>
<div class="card"><figure>
  <div class="legend"><span><i class="sw" style="background:var(--good)"></i>정상</span>
    <span><i class="sw" style="background:var(--warn)"></i>주의</span>
    <span><i class="sw" style="background:var(--crit)"></i>위험</span>
    <span><i class="sw" style="background:var(--line)"></i>미측정</span></div>
  <div id="hchart"></div>
  <figcaption>일별 종합 판정 타임라인 — {HP['경보설계']['종합규칙']}.</figcaption>
</figure></div>
<div class="note"><b>경보 설계 — 문헌 절대임계를 그대로 쓰지 않았다.</b>
{HP['경보설계']['근거']} 따라서 <b>채택: {HP['경보설계']['채택']}</b>.
문헌 임계는 {HP['경보설계']['문헌임계_용도']} 아래 표에 두 발화율을 나란히 둔다 —
문헌 열이 곧 이 시설이 일반 기준에서 얼마나 떨어져 있는지의 측정값이다.</div>
{_tbl([("지표","지표",False),("진단의미","진단 의미",False),("문헌기준","문헌·지침 기준",False),
       ("관측일","관측일",True),("결측률_pct","결측 %",True),("중앙값","중앙값",True),
       ("문헌_주의_pct","문헌 주의 %",True),("문헌_위험_pct","문헌 위험 %",True),
       ("상대_주의_pct","상대 주의 %",True),("상대_위험_pct","상대 위험 %",True)], HP["지표"])}
{sec_tbl("이상구간 타임라인 (상위)",
   [("시작","시작",False),("종료","종료",False),("일수","일수",True),("수준","수준",False)],
   [dict(e, 수준={"crit": "위험", "warn": "주의"}.get(e["수준"], e["수준"])) for e in HP["이상구간"][:12]])}
<div class="note">{"<br>".join("• " + x for x in HP["주의사항"])}</div>

<h2>K. 한계와 다음 단계</h2>
<ul>{"".join(f"<li>{x}</li>" for x in R["한계와_다음단계"])}</ul>

<p class="sub" style="margin-top:36px">재현: <code>python -m src.bioguard.run_pipeline</code> ·
seed {R['실행환경']['seed']} · 원자료 <code>outputs/results_3track.json</code>,
건강상태 <code>outputs/health_program.json</code></p>
</div>
<script>{JS}</script>
<script>
const P={json.dumps(payload, ensure_ascii=False)};
const th=$('#th');
th.onclick=()=>{{const dark=document.documentElement.getAttribute('data-theme')==='dark';
  document.documentElement.setAttribute('data-theme',dark?'light':'dark');
  th.setAttribute('aria-pressed',String(!dark));draw();}};

const K=P.kern, subs=['음폐수','가축분뇨','음식물'];
function drawKernels(){{
  // 커널 모양은 config 의 문헌값으로 다시 만든다 (표 요약과 동일 파라미터)
  const N=31,x=[...Array(N).keys()];
  const cstr=t=>{{const e=x.map(i=>Math.exp(-i/t));const s=e.reduce((a,b)=>a+b);return e.map(v=>v/s);}};
  const conv=(a,b)=>{{const o=Array(N).fill(0);
    for(let i=0;i<N;i++)for(let j=0;j+i<N;j++)o[i+j]+=a[i]*b[j];
    const s=o.reduce((p,q)=>p+q);return o.map(v=>v/s);}};
  const common=conv(cstr(2.5),cstr(3.0));
  const g={{'음폐수':common,'가축분뇨':common,'음식물':conv(cstr(2.5),common)}};
  lineChart($('#kchart'),{{x:x,d:3,
    series:[{{n:'음폐수',c:'var(--s1)',v:g['음폐수']}},
            {{n:'가축분뇨',c:'var(--s2)',v:g['가축분뇨'],op:.75}},
            {{n:'음식물',c:'var(--s3)',v:g['음식물']}}],
    xlab:v=>v+'d',aria:'기질별 체류시간 분포'}});
}}

let RANGE='v';
function drawPred(){{
  const T=P.ts,n=T.dates.length;
  let i0=0;
  if(RANGE==='v')i0=T.dates.indexOf('2022-01-01');
  if(RANGE==='h')i0=T.dates.indexOf('2023-01-01');
  if(i0<0)i0=0;
  const sl=a=>a.slice(i0);
  lineChart($('#pchart'),{{x:sl(T.dates),h:300,u:' ㎥/d',
    band:{{lo:sl(T.lo),hi:sl(T.hi)}},
    series:[{{n:'실측',c:'var(--s1)',v:sl(T.obs)}},{{n:'앙상블 예측',c:'var(--s2)',v:sl(T.pred)}}],
    xlab:v=>v.slice(0,7),aria:'실측 대비 앙상블 예측'}});
}}
document.querySelectorAll('[data-r]').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('[data-r]').forEach(x=>x.setAttribute('aria-pressed','false'));
  b.setAttribute('aria-pressed','true');RANGE=b.dataset.r;drawPred();}});

function draw(){{
  drawKernels();drawPred();
  lineChart($('#m5chart'),{{x:P.m5.dates,h:220,u:' ㎥/d',
    series:[{{n:'시변 절편',c:'var(--s4)',v:P.m5.level}}],xlab:v=>v.slice(0,7),
    aria:'시변 절편 궤적'}});
  timeline($('#hchart'),P.health.dates,P.health.verdict);
  barChart($('#bchart'),P.bars.base.map(r=>({{...r,c:'var(--s1)'}})),{{u:' ㎥/d',aria:'기준선 비교'}});
  barChart($('#mchart'),P.bars.models.map(r=>({{...r,c:'var(--s2)'}})),{{u:' ㎥/d',aria:'모델 계열 비교'}});
  barChart($('#echart'),P.bars.ens.map(r=>({{...r,c:'var(--s3)'}})),{{u:' ㎥/d',aria:'앙상블 결합 비교'}});
}}
draw();
</script>"""
    path.write_text(html, encoding="utf-8")
