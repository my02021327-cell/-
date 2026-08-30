/* BioGuard V2 live adapter: uploaded backend datasets are the sole authority. */
(function () {
  'use strict';
  window.__BIOGUARD_BACKEND_ONLY__ = true;
  /* Same-origin only: the page is served by the backend that answers these
     paths, so the console runs unchanged on any host, port or deployment. */
  const API = '/api/v1';
  const UPLOAD_PATH = '/datasets/upload';
  const HEALTH_RETRY_MS = 5000;
  const V2 = {connected:false,datasetId:null,descriptor:null,history:[],forecast:null,dq:null,selectedIndex:null,pendingIndex:null,selectedRow:null,currentIndex:null,trendMode:'ALL',trendYear:null,uploadState:'CONNECTING',healthRetryTimer:null,requestVersion:0};
  const $ = id => document.getElementById(id);
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  function sanitizeNumeric(value){return finite(value)?value:null;}
  function dateEpoch(value){
    if(typeof value!=='string'||!/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(value))return null;
    const epoch=Date.parse(value+'T00:00:00Z');return finite(epoch)?epoch:null;
  }
  function sortUniqueRowsByDate(rows){
    const unique=new Map();(Array.isArray(rows)?rows:[]).forEach(row=>{if(row&&dateEpoch(row.date)!==null)unique.set(row.date,row);});
    return Array.from(unique.values()).sort((left,right)=>dateEpoch(left.date)-dateEpoch(right.date));
  }
  const fmt = (value, digits=1) => finite(value) ? value.toFixed(digits) : 'N/A';
  const text = (id,value) => { if ($(id)) $(id).textContent=value; };

  class BioGuardApiError extends Error {
    constructor(code,message,status){super(message||code);this.name='BioGuardApiError';this.code=code;this.status=status||0;}
  }
  async function request(path, options) {
    let response;
    try{response=await fetch(API+path,options||{});}catch(error){throw new BioGuardApiError('UPLOAD_REQUEST_FAILED',error.message,0);}
    let body=null;
    try { body=await response.json(); } catch (_) {}
    if (!response.ok) { const detail=body&&body.detail?body.detail:body,code=detail&&detail.code||('HTTP_'+response.status),message=detail&&detail.message||code; throw new BioGuardApiError(code,message,response.status); }
    return body;
  }
  window.BioGuardAPI=Object.freeze({
    uploadEndpoint:API+UPLOAD_PATH,
    healthCheck:()=>request('/health'),
    uploadDataset:file=>{const form=new FormData();form.append('file',file);return request(UPLOAD_PATH,{method:'POST',body:form});},
    getDataset:id=>request('/datasets/'+encodeURIComponent(id)),
    getLatest:id=>request('/datasets/'+encodeURIComponent(id)+'/latest'),
    getRow:(id,date)=>request('/datasets/'+encodeURIComponent(id)+'/rows/'+encodeURIComponent(date)),
    getHistory:id=>request('/datasets/'+encodeURIComponent(id)+'/history'),
    getForecast:id=>request('/datasets/'+encodeURIComponent(id)+'/forecast'),
    getDataQuality:id=>request('/datasets/'+encodeURIComponent(id)+'/data-quality')
  });

  function setUploadState(state,message){
    V2.uploadState=state;
    const indicator=$('localDataIndicator'),button=$('btnLocalDataSelect');
    if(indicator){indicator.classList.remove('online','processing','error');if(state==='READY')indicator.classList.add('online');else if(state==='PROCESSING')indicator.classList.add('processing');else if(state==='ERROR'||state==='OFFLINE')indicator.classList.add('error');}
    if(button)button.disabled=!V2.connected||state==='PROCESSING'||state==='CONNECTING';
    if(message)text('localDataStatus',message);
  }
  function validateUploadFileClientSide(file){
    if(!file||typeof file.name!=='string'||!file.name.toLowerCase().endsWith('.xlsx'))throw new BioGuardApiError('INVALID_FILE_TYPE','BioGuard 공식 .xlsx 파일만 업로드할 수 있습니다.',0);
  }
  function uploadErrorCode(error){
    const code=error&&error.code||'DATASET_PROCESSING_FAILED',message=String(error&&error.message||'');
    if(code==='WORKBOOK_VALIDATION_FAILED'){
      if(message.startsWith('INVALID_WORKBOOK'))return'WORKBOOK_PARSE_FAILED';
      if(message.startsWith('MISSING_SHEET'))return'BIOGUARD_INPUT_MISSING';
      return'SCHEMA_INVALID';
    }
    return code;
  }
  async function uploadBioGuardDataset(file){validateUploadFileClientSide(file);return window.BioGuardAPI.uploadDataset(file);}

  function setMode(connected) {
    V2.connected=connected;
    const label=connected?'MODE: BACKEND LIVE · SERVER: ONLINE':'MODE: BACKEND OFFLINE';
    text('sbMode',label); const subbar=document.querySelector('.hmi-subbar > div:last-child'); if(subbar)subbar.textContent=label;
    const button=$('btnLocalDataSelect');if(button)button.disabled=!connected||V2.uploadState==='PROCESSING'||V2.uploadState==='CONNECTING';
  }
  /* Global tower = backend process health only. Telemetry lamps are never aggregated here. */
  const TOWER_TEXT_CLASS={GREEN:'status-green',AMBER:'status-amber',RED:'status-red',DATA:'status-gray'};
  function setLamps(level,title,detail) {
    const ids={RED:'lampRed',AMBER:'lampAmber',GREEN:'lampGreen'};
    Object.keys(ids).forEach(key=>{const lamp=$(ids[key]);if(!lamp)return;lamp.classList.toggle('is-on',key===level);lamp.classList.toggle('is-off',key!==level);});
    const tower=$('signalTower');if(tower)tower.setAttribute('data-tower-state',level==='DATA'?'DATA':level||'NONE');
    const titleNode=$('annTitle');
    if(titleNode){Object.keys(TOWER_TEXT_CLASS).forEach(key=>titleNode.classList.remove(TOWER_TEXT_CLASS[key]));titleNode.classList.add(TOWER_TEXT_CLASS[level]||'status-gray');}
    text('annTitle',title);text('annDetail',detail);
  }
  function clearRecorder(){
    ['ch4RecorderTransitionTrace','ch4RecorderForecastTrace'].forEach(id=>{if($(id))$(id).setAttribute('points','');});
    ['ch4RecorderActualTrace','ch4RecorderBridgeTrace','ch4RecorderActualMarkers','ch4RecorderForecastMarkers','ch4RecorderYTicks','ch4RecorderXLabels'].forEach(id=>{if($(id))$(id).innerHTML='';});
    if($('ch4RecorderT0')){$('ch4RecorderT0').setAttribute('x1','-100');$('ch4RecorderT0').setAttribute('x2','-100');}
    if($('ch4RecorderMessage')){$('ch4RecorderMessage').textContent='';$('ch4RecorderMessage').setAttribute('visibility','hidden');}
    if($('ch4RecorderSvg'))$('ch4RecorderSvg').setAttribute('data-chart-state','EMPTY');
    text('ch4CurrentReadout','CURR: N/A');text('ch4ForecastReadout','FORECAST (T+1): N/A');text('ch4ModelReadout','MODEL: A30 V2');
    ['summaryLastActual','summaryD1','summaryD7','summaryDelta7','summaryT0'].forEach(id=>text(id,'N/A'));
  }
  function clearCharts() {
    clearRecorder();if($('trendForecastTrace'))$('trendForecastTrace').setAttribute('points','');
    ['trendActualSegments','trendActualMarkers','trendGrid','trendYTicks','trendXLabels'].forEach(id=>{if($(id))$(id).innerHTML='';});
  }
  function resetDatasetState(message) {
    V2.requestVersion+=1;
    V2.datasetId=null;V2.descriptor=null;V2.history=[];V2.forecast=null;V2.dq=null;V2.selectedIndex=null;V2.pendingIndex=null;V2.selectedRow=null;V2.currentIndex=null;V2.trendMode='ALL';V2.trendYear=null;
    clearCharts();
    ['gaugeForecastValue','gaugePurityForecastValue','gaugeBiogasValue'].forEach(id=>{text(id,'—');if($(id)){$(id).classList.remove('led-forecast');$(id).classList.add('led-unavailable');}});
    ['gaugeTempValue','gaugePhValue'].forEach(id=>{text(id,'—');if($(id))$(id).className='led-number led-dual-value led-unavailable';});
    ['gaugeForecastNote','gaugePurityNote','gaugeBiogasNote','gaugeTempPhSides'].forEach(id=>{if($(id)){$(id).textContent='';$(id).classList.remove('is-currentonly');}});
    ['gaugeTempStatus','gaugePhStatus'].forEach(id=>{if($(id)){$(id).className='led-dual-status';$(id).textContent='';}});
    document.querySelectorAll('[id^="pf_"]').forEach(node=>{node.textContent='N/A';});
    text('ch4CurrentReadout','CURR: N/A');text('ch4ForecastReadout','FORECAST (T+1): N/A');text('ch4ModelReadout','MODEL: A30 V2');
    ['summaryLastActual','summaryD1','summaryD7','summaryDelta7','summaryT0'].forEach(id=>text(id,'N/A'));
    text('telemetrySourceLabel','SOURCE: DATA UNAVAILABLE · SELECTED: —');text('telemetryUpdateLabel','UPDATE: DATASET EVENT');
    if($('telemetryBody'))$('telemetryBody').innerHTML='<tr><td colspan="3">DATASET NOT LOADED</td></tr>';clearOperatorActions('SERVER DATASET NOT LOADED');
    text('processSourceLabel','SOURCE: DATA UNAVAILABLE');text('trendTitleLabel','UPLOADED DATASET — CH4 PRODUCTION · DATA UNAVAILABLE');text('trendSourceLabel','SERVER DATASET NOT LOADED');
    text('trendRangeReadout','—');text('trendPointReadout','POINT: —');text('trendCoverage','—');text('paramTitleLabel','PARAMETER / 실데이터 — DATA UNAVAILABLE');text('paramSourceLabel','SERVER DATASET NOT LOADED');text('paramDateReadout','DATE: —');
    if($('paramTableBody'))$('paramTableBody').innerHTML='<tr><td colspan="7">DATA UNAVAILABLE</td></tr>';if($('dqPanel'))$('dqPanel').hidden=true;
    document.querySelectorAll('.trend-controls .year-btn').forEach(node=>node.remove());const scrubberYears=document.querySelector('.scrubber-years');if(scrubberYears)scrubberYears.innerHTML='';
    const picker=$('sigDatePicker');if(picker){picker.removeAttribute('min');picker.removeAttribute('max');picker.value='';}
    text('sigSelectedReadout','SELECTED: —');text('sigModeReadout','[NO DATASET]');text('sigDateValidation','');text('sbPlcAddr','DATASET: NOT LOADED');text('sbProtocol','MODEL: READY');text('sbScanRate','DQ: N/A');
    text('localDataStatus',message||'소화조 데이터를 업로드해주세요.');text('localDataMeta',V2.connected?'DROP EXCEL DATA HERE · SERVER CONNECTED':'DROP EXCEL DATA HERE · SERVER OFFLINE');
    setLamps(null,'SYSTEM STATUS: NO DATASET','소화조 데이터를 업로드해주세요.');document.documentElement.setAttribute('data-bioguard-v2',V2.connected?'CONNECTED_NO_DATASET':'OFFLINE');
  }
  function recorderScale(values,top,bottom){
    const usable=values.map(sanitizeNumeric).filter(finite);if(!usable.length)throw new Error('NO_FINITE_CH4_VALUES');
    let low=Math.min(...usable),high=Math.max(...usable);const range=high-low;
    const padding=range===0?Math.max(100,Math.abs(high)*0.05):Math.max(range*0.10,100);
    low-=padding;high+=padding;
    return{low,high,y:value=>bottom-(value-low)/(high-low)*(bottom-top)};
  }
  function normalizedForecastPredictions(strict){
    const raw=V2.forecast&&V2.forecast.methane&&Array.isArray(V2.forecast.methane.predictions)?V2.forecast.methane.predictions:[];
    const predictions=raw.slice().sort((left,right)=>Number(left.horizon)-Number(right.horizon)).map(row=>({...row,value:sanitizeNumeric(row.value)}));
    const valid=predictions.length===7&&predictions.every((row,index)=>row.horizon===index+1&&row.value!==null&&dateEpoch(row.target_date)!==null);
    if(strict&&!valid)throw new Error('FORECAST_REQUIRES_EXACTLY_7_FINITE_HORIZONS');
    return valid?predictions:[];
  }
  function forecastAvailable(){return Boolean(V2.forecast&&V2.forecast.status==='AVAILABLE'&&normalizedForecastPredictions(false).length===7);}
  function setPrimaryGauge(id,value,digits,mode){
    const node=$(id),available=finite(value);if(!node)return;
    node.classList.toggle('led-forecast',available&&mode==='FORECAST');node.classList.toggle('led-unavailable',!available);node.textContent=available?value.toFixed(digits):'N/A';
  }

  /* Digits per project variable; also used by the telemetry matrix. */
  const SIGNAL_DIGITS={temperature:2,pH:2,VFA:0,ALK:0,VFA_TA:3,TAN:0,FAN:0,TS:2,VS:2,CODcr:0,
    ch4_purity:1,biogas_total:0,ch4_observed:0};
  /* CURRENT is the dataset's own forecast origin, never the OS date. */
  function selectedDateText(){
    const row=V2.selectedRow;return row&&row.measurement_date?row.measurement_date:'—';
  }
  function isCurrentMode(){
    return V2.selectedIndex!==null&&V2.selectedIndex===V2.currentIndex;
  }
  function sideText(signal,digits){
    const sides=signal&&signal.side_values;
    if(!sides||(!finite(sides.A)&&!finite(sides.B)))return '';
    return (finite(sides.A)?'A '+fmt(sides.A,digits):'A N/A')+' · '+(finite(sides.B)?'B '+fmt(sides.B,digits):'B N/A');
  }
  function setNote(id,value,currentOnly){
    const node=$(id);if(!node)return;
    node.textContent=value||'';node.classList.toggle('is-currentonly',Boolean(currentOnly));
  }

  /* Gauges 1-3 switch mode with the selected date. On the forecast origin they
     carry the D+1 predictions in cyan; on any earlier date they carry that date's
     measured process values in green. A forecast is never shown as if it were an
     actual, and an actual is never shown as if it were a forecast. */
  function setModeGauge(id,titleId,title,value,digits,mode,noteId,note){
    text(titleId,title);
    setPrimaryGauge(id,finite(value)?value:null,digits,mode);
    setNote(noteId,note,mode==='FORECAST'&&!finite(value));
  }
  function renderForecastGauge(){
    if(!isCurrentMode()){
      const value=sanitizeNumeric((V2.selectedRow||{}).CH4_m3d_observed);
      setModeGauge('gaugeForecastValue','gauge1Title','메탄생성량 측정',value,0,'ACTUAL',
        'gaugeForecastNote',finite(value)?'실측 · '+selectedDateText():'해당 날짜 실측값 없음');
      return;
    }
    const h1=forecastAvailable()?V2.forecast.methane.predictions.find(item=>item.horizon===1):null;
    setModeGauge('gaugeForecastValue','gauge1Title','메탄생성량 예측',h1&&h1.value,0,'FORECAST',
      'gaugeForecastNote',h1?'T0 '+V2.forecast.origin_date+' · D+1 '+h1.target_date:'FORECAST N/A');
  }
  function renderPurityForecastGauge(){
    if(!isCurrentMode()){
      const signal=backendSignal('ch4_purity'),value=signal?sanitizeNumeric(signal.value):null;
      setModeGauge('gaugePurityForecastValue','gauge2Title','메탄순도 측정',value,1,'ACTUAL',
        'gaugePurityNote',finite(value)?sideText(signal,1):'해당 날짜 실측값 없음');
      return;
    }
    const purity=forecastAvailable()?V2.forecast.purity_next_day:null,
      value=purity?sanitizeNumeric(purity.value_pct):null;
    setModeGauge('gaugePurityForecastValue','gauge2Title','메탄순도 예측',value,1,'FORECAST',
      'gaugePurityNote',finite(value)?'D+1 '+purity.target_date:'FORECAST N/A');
  }
  function biogasForecastH1(){
    const raw=V2.forecast&&V2.forecast.biogas&&Array.isArray(V2.forecast.biogas.predictions)
      ?V2.forecast.biogas.predictions:[];
    const h1=raw.find(item=>item.horizon===1);
    return h1&&finite(sanitizeNumeric(h1.value))?h1:null;
  }
  function renderBiogasForecastGauge(){
    if(!isCurrentMode()){
      const signal=backendSignal('biogas_total'),value=signal?sanitizeNumeric(signal.value):null;
      setModeGauge('gaugeBiogasValue','gauge3Title','총 바이오가스 측정량',value,0,'ACTUAL',
        'gaugeBiogasNote',finite(value)?sideText(signal,0):'해당 날짜 실측값 없음');
      return;
    }
    const h1=V2.forecast&&V2.forecast.status==='AVAILABLE'?biogasForecastH1():null;
    setModeGauge('gaugeBiogasValue','gauge3Title','총 바이오가스 생산량 예측',h1&&h1.value,0,'FORECAST',
      'gaugeBiogasNote',h1?'D+1 '+h1.target_date:'FORECAST N/A');
  }
  /* Gauge 4 — temperature and pH in one bay, each keeping its own backend status. */
  function renderTempPhGauge(){
    const notes=[];
    [['temperature','gaugeTempValue','gaugeTempStatus','TEMP'],
     ['pH','gaugePhValue','gaugePhStatus','pH']].forEach(spec=>{
      const signal=backendSignal(spec[0]),value=signal?sanitizeNumeric(signal.value):null,
        digits=SIGNAL_DIGITS[spec[0]],node=$(spec[1]),status=$(spec[2]);
      if(node){
        // Colour follows the backend status; no threshold is evaluated here.
        node.className='led-number led-dual-value'+(
          !finite(value)||!signal?' led-unavailable':
          signal.status==='WATCH'?' led-watch':
          signal.status==='ACTION'?' led-action':
          signal.status==='NORMAL'?'':' led-unavailable');
        node.textContent=finite(value)?value.toFixed(digits):'—';
      }
      if(status){
        const presentation=signal?resolveSignalPresentation(signal):null;
        status.className='led-dual-status'+(presentation?' '+presentation.css:'');
        status.textContent=presentation?presentation.text:'';
      }
      const sides=finite(value)?sideText(signal,digits):'';
      if(sides)notes.push(spec[3]+' '+sides);
    });
    setNote('gaugeTempPhSides',notes.join('  ·  '));
  }
  function renderGauges(){
    renderForecastGauge();renderPurityForecastGauge();renderBiogasForecastGauge();renderTempPhGauge();
  }
  function buildCh4RecorderSeries(){
    const hasForecast=forecastAvailable(),origin=hasForecast?V2.forecast.origin_date:null,endDate=origin||V2.descriptor.latest_data_date,endEpoch=dateEpoch(endDate);
    if(endEpoch===null)throw new Error('INVALID_FORECAST_ORIGIN_DATE');
    const windowStart=endEpoch-13*86400000;
    const actualObservedSeries=sortUniqueRowsByDate(V2.history).filter(row=>{const epoch=dateEpoch(row.date);return epoch>=windowStart&&epoch<=endEpoch;}).map(row=>({...row,CH4_m3d_observed:sanitizeNumeric(row.CH4_m3d_observed)}));
    const forecastSeries=hasForecast?normalizedForecastPredictions(true):[];
    return{origin,actualObservedSeries,actualDisplaySeries:actualObservedSeries.slice(),forecastSeries};
  }
  function buildActualSegments(rows,xOfDate,y){
    const segments=[];let current=[],previousDate=null;
    rows.forEach(row=>{const dayGap=previousDate?Math.round((dateEpoch(row.date)-dateEpoch(previousDate))/86400000):1;if(dayGap>1&&current.length){segments.push(current);current=[];}if(finite(row.CH4_m3d_observed))current.push(xOfDate(row.date).toFixed(1)+','+y(row.CH4_m3d_observed).toFixed(1));else if(current.length){segments.push(current);current=[];}previousDate=row.date;});
    if(current.length)segments.push(current);return segments;
  }
  function buildMissingBridges(rows,xOfDate,y){
    const observed=[];rows.forEach((row,index)=>{if(finite(row.CH4_m3d_observed))observed.push(index);});
    const bridges=[];for(let index=1;index<observed.length;index+=1){const left=observed[index-1],right=observed[index],gap=Math.round((dateEpoch(rows[right].date)-dateEpoch(rows[left].date))/86400000)-1;if(gap>=1&&gap<=2)bridges.push([xOfDate(rows[left].date).toFixed(1)+','+y(rows[left].CH4_m3d_observed).toFixed(1),xOfDate(rows[right].date).toFixed(1)+','+y(rows[right].CH4_m3d_observed).toFixed(1)]);}
    return bridges;
  }
  function buildForecastSegments(actual,future,xOfDate,y){
    const forecastPoints=future.map(row=>xOfDate(row.target_date).toFixed(1)+','+y(row.value).toFixed(1));
    let transition=[];for(let index=actual.length-1;index>=0;index-=1){if(finite(actual[index].CH4_m3d_observed)&&future.length){transition=[xOfDate(actual[index].date).toFixed(1)+','+y(actual[index].CH4_m3d_observed).toFixed(1),forecastPoints[0]];break;}}
    return{forecastPoints,transition};
  }
  function buildForecastSummary(series){
    const observed=series.actualObservedSeries.filter(row=>finite(row.CH4_m3d_observed));const last=observed.length?observed[observed.length-1]:null,d1=series.forecastSeries.find(row=>row.horizon===1),d7=series.forecastSeries.find(row=>row.horizon===7),delta=last&&d7?d7.value-last.CH4_m3d_observed:null;
    text('summaryLastActual',last?last.CH4_m3d_observed.toLocaleString(undefined,{maximumFractionDigits:0})+' m³/d':'N/A');text('summaryD1',d1?d1.value.toLocaleString(undefined,{maximumFractionDigits:0})+' m³/d':'N/A');text('summaryD7',d7?d7.value.toLocaleString(undefined,{maximumFractionDigits:0})+' m³/d':'N/A');text('summaryDelta7',finite(delta)?(delta>=0?'+':'')+delta.toLocaleString(undefined,{maximumFractionDigits:0})+' m³/d':'N/A');text('summaryT0',series.origin||'N/A');
    return{last,d1,d7,delta};
  }
  function renderCh4TrendRecorder(){
    clearRecorder();
    try{
      const series=buildCh4RecorderSeries(),actual=series.actualDisplaySeries,future=series.forecastSeries,values=actual.map(row=>row.CH4_m3d_observed).concat(future.map(row=>row.value)),chart=recorderScale(values,15,175),timelineDates=actual.map(row=>row.date).concat(future.map(row=>row.target_date)),epochs=timelineDates.map(dateEpoch).filter(finite);
      if(!epochs.length)throw new Error('EMPTY_CH4_TIMELINE');
      const firstEpoch=Math.min(...epochs),lastEpoch=Math.max(...epochs),span=Math.max(86400000,lastEpoch-firstEpoch),xOfDate=date=>55+665*(dateEpoch(date)-firstEpoch)/span;
      const segments=buildActualSegments(actual,xOfDate,chart.y),bridges=buildMissingBridges(actual,xOfDate,chart.y),forecast=buildForecastSegments(actual,future,xOfDate,chart.y),summary=buildForecastSummary(series);
      $('ch4RecorderActualTrace').innerHTML=segments.map(points=>'<polyline vector-effect="non-scaling-stroke" points="'+points.join(' ')+'"/>').join('');
      $('ch4RecorderBridgeTrace').innerHTML=bridges.map(points=>'<polyline vector-effect="non-scaling-stroke" data-visual-bridge="true" points="'+points.join(' ')+'"/>').join('');
      // Markers carry their backend value and provenance so the pre-T0 trace can
      // be verified against /history, and so a synthesised input is not drawn as
      // if it were a direct measurement.
      $('ch4RecorderActualMarkers').innerHTML=actual.map(row=>{
        if(!finite(row.CH4_m3d_observed))return '';
        const imputed=row.ch4_source==='DERIVED_FROM_IMPUTED';
        const picked=!isCurrentMode()&&row.date===selectedDateText();
        if(picked){
          return '<rect vector-effect="non-scaling-stroke" data-date="'+row.date
            +'" data-value="'+row.CH4_m3d_observed+'" data-source="'+(row.ch4_source||'DERIVED_FROM_OBSERVED')
            +'" data-selected="true" x="'+(xOfDate(row.date)-4).toFixed(1)+'" y="'+(chart.y(row.CH4_m3d_observed)-4).toFixed(1)
            +'" width="8" height="8" fill="'+(imputed?'none':'#00FF00')+'" stroke="#FFFFFF" stroke-width="2"/>';
        }
        return '<rect vector-effect="non-scaling-stroke" data-date="'+row.date
          +'" data-value="'+row.CH4_m3d_observed+'" data-source="'+(row.ch4_source||'DERIVED_FROM_OBSERVED')
          +'" x="'+(xOfDate(row.date)-2.5).toFixed(1)+'" y="'+(chart.y(row.CH4_m3d_observed)-2.5).toFixed(1)
          +'" width="5" height="5"'+(imputed?' fill="none" stroke="#00FF00" stroke-width="1.5"':'')+'/>';
      }).join('');
      $('ch4RecorderForecastTrace').setAttribute('points',forecast.forecastPoints.join(' '));$('ch4RecorderTransitionTrace').setAttribute('points',forecast.transition.join(' '));
      $('ch4RecorderForecastMarkers').innerHTML=future.map(row=>'<rect vector-effect="non-scaling-stroke" x="'+(xOfDate(row.target_date)-2.5).toFixed(1)+'" y="'+(chart.y(row.value)-2.5).toFixed(1)+'" width="5" height="5"/>').join('');
      $('ch4RecorderYTicks').innerHTML=[0,1,2,3,4].map(index=>'<text x="18" y="'+(19+index*40)+'">'+(chart.high-(chart.high-chart.low)*index/4).toFixed(0)+'</text>').join('');
      const labels=[],actualIndexes=new Set();if(actual.length){const stride=Math.max(1,Math.ceil(actual.length/7));for(let index=0;index<actual.length;index+=stride)actualIndexes.add(index);actualIndexes.add(actual.length-1);}
      Array.from(actualIndexes).sort((a,b)=>a-b).forEach(index=>{const row=actual[index],isT0=series.origin&&row.date===series.origin;labels.push('<text x="'+xOfDate(row.date).toFixed(1)+'" y="'+(isT0?'188':'202')+'"'+(isT0?' fill="#FF0000" font-weight="bold"':'')+'>'+row.date.slice(5).replace('-','/')+(isT0?'[T0]':'')+'</text>');});
      future.forEach(row=>labels.push('<text x="'+xOfDate(row.target_date).toFixed(1)+'" y="202">'+row.target_date.slice(5).replace('-','/')+'</text>'));$('ch4RecorderXLabels').innerHTML=labels.join('');
      if(series.origin){const t0=xOfDate(series.origin);$('ch4RecorderT0').setAttribute('x1',t0.toFixed(1));$('ch4RecorderT0').setAttribute('x2',t0.toFixed(1));}
      // On a historical date the readout must describe that date, not T0.
      if(isCurrentMode()){
        text('ch4CurrentReadout','CURR: '+(summary.last?summary.last.CH4_m3d_observed.toLocaleString(undefined,{maximumFractionDigits:0})+' m³/d':'N/A'));
        text('ch4ForecastReadout','FORECAST (T+1): '+(summary.d1?summary.d1.value.toLocaleString(undefined,{maximumFractionDigits:0})+' m³/d':'N/A'));
      }else{
        const picked=sanitizeNumeric((V2.selectedRow||{}).CH4_m3d_observed);
        text('ch4CurrentReadout','SELECTED ACTUAL: '+(finite(picked)?picked.toLocaleString(undefined,{maximumFractionDigits:0})+' m³/d':'N/A')+' · '+selectedDateText());
        text('ch4ForecastReadout','FORECAST ORIGIN: '+(series.origin||'N/A')+' (T0 이후 표시)');
      }
      text('ch4ModelReadout','MODEL: A30 V2'+(series.origin?' · T0: '+series.origin:''));
      if(!future.length){$('ch4RecorderMessage').textContent='FORECAST UNAVAILABLE';$('ch4RecorderMessage').setAttribute('visibility','visible');$('ch4RecorderSvg').setAttribute('data-chart-state','FORECAST_UNAVAILABLE');}else $('ch4RecorderSvg').setAttribute('data-chart-state','READY');
    }catch(error){
      console.error('[BioGuard CH4 Recorder] render failed',error);clearRecorder();$('ch4RecorderMessage').textContent='CHART DATA ERROR';$('ch4RecorderMessage').setAttribute('visibility','visible');$('ch4RecorderSvg').setAttribute('data-chart-state','ERROR');text('ch4CurrentReadout','CHART DATA ERROR');
    }
  }
  function trendRows(){let rows=V2.history.slice();if(V2.trendYear)return rows.filter(row=>row.date.startsWith(V2.trendYear+'-'));const sizes={'7D':7,'30D':30,'90D':90,'1Y':365};return sizes[V2.trendMode]?rows.slice(-sizes[V2.trendMode]):rows;}
  const TREND={left:40,right:890,top:10,bottom:170};
  function renderTrend(){
    ['trendActualSegments','trendActualMarkers','trendGrid','trendYTicks','trendXLabels'].forEach(id=>{if($(id))$(id).innerHTML='';});
    if($('trendForecastTrace'))$('trendForecastTrace').setAttribute('points','');
    text('trendForecastLabel','BACKEND ACTUAL HISTORY');
    try{
      const rows=trendRows().map(row=>({date:row.date,value:sanitizeNumeric(row.CH4_m3d_observed)})).filter(row=>dateEpoch(row.date)!==null);
      text('trendCoverage',rows.filter(row=>finite(row.value)).length+' OBSERVED / '+rows.length+' ROWS');
      text('trendTitleLabel','CH4 PRODUCTION — ACTUAL · '+V2.descriptor.first_date+' .. '+V2.descriptor.latest_data_date);
      text('trendSourceLabel','SOURCE: BACKEND DATASET · FILE: '+V2.descriptor.file_name);
      text('trendRangeReadout',rows.length?rows[0].date+' .. '+rows[rows.length-1].date:'NO ROWS');
      const observed=rows.filter(row=>finite(row.value));
      text('trendPointReadout',observed.length?'POINT: '+observed[observed.length-1].date+' = '+fmt(observed[observed.length-1].value,0)+' m³/d':'POINT: —');
      if(!observed.length){$('trendSvg').setAttribute('data-chart-state','NO_OBSERVED_VALUES');return;}
      const chart=recorderScale(observed.map(row=>row.value),TREND.top,TREND.bottom);
      const epochs=rows.map(row=>dateEpoch(row.date)),firstEpoch=Math.min(...epochs),lastEpoch=Math.max(...epochs),span=Math.max(86400000,lastEpoch-firstEpoch);
      const xOfDate=date=>TREND.left+(TREND.right-TREND.left)*(dateEpoch(date)-firstEpoch)/span;
      const point=row=>xOfDate(row.date).toFixed(1)+','+chart.y(row.value).toFixed(1);
      let segments=[],segment=[];
      rows.forEach(row=>{if(finite(row.value))segment.push(point(row));else if(segment.length){segments.push(segment);segment=[];}});
      if(segment.length)segments.push(segment);
      $('trendActualSegments').innerHTML=segments.filter(points=>points.length>1).map(points=>'<polyline vector-effect="non-scaling-stroke" points="'+points.join(' ')+'"/>').join('')
        +segments.filter(points=>points.length===1).map(points=>'<polyline vector-effect="non-scaling-stroke" points="'+points[0]+' '+points[0]+'"/>').join('');
      if(observed.length<=60)$('trendActualMarkers').innerHTML=observed.map(row=>'<rect vector-effect="non-scaling-stroke" x="'+(xOfDate(row.date)-2).toFixed(1)+'" y="'+(chart.y(row.value)-2).toFixed(1)+'" width="4" height="4"/>').join('');
      const ticks=[0,1,2,3,4];
      $('trendGrid').innerHTML=ticks.slice(1,4).map(index=>{const y=(TREND.top+(TREND.bottom-TREND.top)*index/4).toFixed(1);return'<line vector-effect="non-scaling-stroke" x1="'+TREND.left+'" y1="'+y+'" x2="'+TREND.right+'" y2="'+y+'"/>';}).join('');
      $('trendYTicks').innerHTML=ticks.map(index=>'<text x="'+(TREND.left-4)+'" y="'+(TREND.top+(TREND.bottom-TREND.top)*index/4+3.5).toFixed(1)+'" text-anchor="end">'+(chart.high-(chart.high-chart.low)*index/4).toFixed(0)+'</text>').join('');
      const labelCount=Math.max(2,Math.min(9,rows.length)),indexes=new Set();
      for(let slot=0;slot<labelCount;slot+=1)indexes.add(Math.round(slot*(rows.length-1)/(labelCount-1)));
      const ordered=Array.from(indexes).sort((a,b)=>a-b);
      $('trendXLabels').innerHTML=ordered.map(index=>{const anchor=index===0?'start':index===rows.length-1?'end':'middle';return'<text x="'+xOfDate(rows[index].date).toFixed(1)+'" y="'+(TREND.bottom+18)+'" text-anchor="'+anchor+'">'+rows[index].date+'</text>';}).join('');
      $('trendSvg').setAttribute('data-chart-state','READY');
    }catch(error){
      console.error('[BioGuard CH4 Trend] render failed',error);
      ['trendActualSegments','trendActualMarkers','trendGrid','trendYTicks','trendXLabels'].forEach(id=>{if($(id))$(id).innerHTML='';});
      $('trendSvg').setAttribute('data-chart-state','ERROR');text('trendPointReadout','CHART DATA ERROR');
    }
  }
  function buildYearButtons(){
    const controls=document.querySelector('.trend-controls');if(!controls)return;controls.querySelectorAll('.year-btn').forEach(node=>node.remove());const anchor=$('trendRangeReadout'),years=[...new Set(V2.history.map(row=>row.date.slice(0,4)))];years.forEach(year=>{const button=document.createElement('button');button.type='button';button.className='hmi-btn year-btn';button.dataset.year=year;button.textContent=year;button.style.padding='3px 8px';button.addEventListener('click',()=>{V2.trendYear=year;V2.trendMode='YEAR';renderTrend();});controls.insertBefore(button,anchor);});const scrubber=document.querySelector('.scrubber-years');if(scrubber)scrubber.innerHTML=years.map(year=>'<span>'+year+'</span>').join('');
  }
  function put(id,value,unit,digits=1){text(id,finite(value)?value.toFixed(digits)+(unit?' '+unit:''):'N/A');}
  function renderProcessAndParameters(currentMode){
    const row=V2.selectedRow||{};put('pf_feed_A',row.feed_A_tpd,'t/d');put('pf_feed_B',row.feed_B_tpd,'t/d');put('pf_feed_AB',row.feed_AB_tpd,'t/d');put('pf_biogas_A',row.biogas_A_m3d,'m³/d');put('pf_biogas_B',row.biogas_B_m3d,'m³/d');put('pf_biogas_AB',row.biogas_AB_m3d,'m³/d');put('pf_CH4_purity_A',row.ch4_purity_A_pct,'%',1);put('pf_CH4_purity_B',row.ch4_purity_B_pct,'%',1);put('pf_ch4obs',row.CH4_m3d_observed,'m³/d',0);const h1=currentMode&&forecastAvailable()?V2.forecast.methane.predictions.find(item=>item.horizon===1):null;put('pf_ch4fc',h1?h1.value:null,'m³/d',0);
    const fields=[['pH','pH','',2],['VFA','VFA','mg/L',0],['ALK','ALK','mg/L',0],['Temperature','temperature','℃',2],['TAN','TAN','mg/L',0],['TS','TS','%',2],['VS','VS','%',2],['CODcr','CODcr','mg/L',0]];fields.forEach(field=>['A','B'].forEach(side=>{let key=field[1]+'_'+side;if(field[1]==='temperature')key+='_C';else if(['VFA','ALK','TAN','CODcr'].includes(field[1]))key+='_mgL';else if(['TS','VS'].includes(field[1]))key+='_pct';put('pf_'+field[0]+'_'+side,row[key],field[2],field[3]);}));text('pf_FAN_A','N/A');text('pf_FAN_B','N/A');text('processSourceLabel','SOURCE: BACKEND DATASET · SELECTED: '+row.measurement_date);text('paramTitleLabel','PARAMETER / 실데이터 (BACKEND DATASET)');text('paramSourceLabel','SOURCE: BACKEND DATASET · FILE: '+V2.descriptor.file_name);text('paramDateReadout','DATE: '+row.measurement_date);
    const table=[['Temperature','temperature_A_C','temperature_B_C','℃'],['pH','pH_A','pH_B',''],['VFA','VFA_A_mgL','VFA_B_mgL','mg/L'],['ALK','ALK_A_mgL','ALK_B_mgL','mg/L'],['VFA/TA','VFA_TA_A','VFA_TA_B',''],['TAN','TAN_A_mgL','TAN_B_mgL','mg/L'],['TS','TS_A_pct','TS_B_pct','%'],['VS','VS_A_pct','VS_B_pct','%'],['CODcr','CODcr_A_mgL','CODcr_B_mgL','mg/L']];$('paramTableBody').innerHTML=table.map(field=>'<tr><td>'+field[0]+'</td><td>'+fmt(row[field[1]],2)+'</td><td>'+fmt(row[field[2]],2)+'</td><td>'+field[3]+'</td><td>'+row.measurement_date+'</td><td>0</td><td>'+(finite(row[field[1]])||finite(row[field[2]])?'BACKEND VALID':'MISSING')+'</td></tr>').join('');
  }
  /* The one and only status -> UI mapping. Backend health/DQ already decided the
     status (DQ unusable is resolved to DATA server-side); nothing is re-judged here. */
  /* Lamp + CSS only. The Korean wording is the backend's status_text, which is
     the project's own vocabulary -- notably '주의(통계)' / '이상(통계)', because
     the project specification states a single variable's colour is a statistical
     rarity signal and not a process action. */
  const SIGNAL_PRESENTATION={
    NORMAL:{lamp:'t-green',css:'signal-normal'},
    WATCH:{lamp:'t-amber',css:'signal-watch'},
    ACTION:{lamp:'t-red',css:'signal-action'},
    DATA:{lamp:'t-gray',css:'signal-data'},
    DATA_NORMAL:{lamp:'t-gray',css:'signal-info'},
    DATA_WATCH:{lamp:'t-gray',css:'signal-info'},
    INFO:{lamp:'t-gray',css:'signal-info'}
  };
  function resolveSignalPresentation(signal){
    const base=SIGNAL_PRESENTATION[signal&&signal.status]||SIGNAL_PRESENTATION.DATA;
    return {lamp:base.lamp,css:base.css,text:(signal&&signal.status_text)||'데이터없음'};
  }
  function backendSignal(variable){
    const row=V2.selectedRow,signals=row&&Array.isArray(row.telemetry_signals)?row.telemetry_signals:[];
    return signals.find(signal=>signal&&signal.variable===variable)||null;
  }
  /* The matrix is the process chemistry / stability panel. Temperature, pH,
     purity, total biogas and observed CH4 are shown in the top gauges and the
     recorder instead, and FAN has no formula in the project. They stay in the
     backend payload for the tower, the operator engine and other views -- this
     list only controls what the matrix draws. */
  const TELEMETRY_DISPLAY_ORDER=['VFA_TA','VFA','ALK','TAN','TS','VS','CODcr'];
  function renderTelemetry(currentMode){
    const row=V2.selectedRow||{},
      signals=Array.isArray(row.telemetry_signals)?row.telemetry_signals:[],
      byVariable=new Map(signals.map(signal=>[signal.variable,signal])),
      shown=TELEMETRY_DISPLAY_ORDER.map(variable=>byVariable.get(variable)).filter(Boolean);
    $('telemetryBody').innerHTML=shown.map(signal=>{
      const presentation=resolveSignalPresentation(signal),digits=SIGNAL_DIGITS[signal.variable],
        side=sideText(signal,digits),
        value=finite(signal.value)?fmt(signal.value,digits)+(signal.unit?' '+signal.unit:''):'N/A';
      return'<tr data-variable="'+signal.variable+'" data-signal-status="'+signal.status+'" data-dq-status="'+signal.dq_status+'"><td>'+signal.label+'</td><td class="telemetry-value '+presentation.css+'">'+value+(side?'<span class="telemetry-secondary">'+side+'</span>':'')+'</td><td><div class="signal-cell '+presentation.css+'"><span class="table-lamp '+presentation.lamp+'"></span><b>'+presentation.text+'</b></div></td></tr>';
    }).join('')||'<tr><td colspan="3">BACKEND TELEMETRY SIGNALS UNAVAILABLE</td></tr>';
    text('telemetrySourceLabel','SOURCE: BACKEND DATASET · SELECTED: '+row.measurement_date+' · '+(currentMode?'CURRENT':'HISTORICAL'));
    text('telemetryUpdateLabel','UPDATE: DATASET EVENT');
  }
  const escapeHtml = value => String(value == null ? '' : value)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  const numText = (value,digits) => finite(value) ? value.toFixed(digits) : 'N/A';
  function clearOperatorActions(message){
    const body=$('operatorActionBody');
    if(body)body.innerHTML='<div class="op-provenance">'+escapeHtml(message||'BACKEND OPERATOR ACTION UNAVAILABLE')+'</div>';
  }
  const LEVEL_LAMP={PROCESS_ACTION:'t-red',PROCESS_WATCH:'t-amber',DATA_SUSPECT:'t-amber'};
  const LEVEL_TEXT={PROCESS_ACTION:'조치필요',PROCESS_WATCH:'주의',DATA_SUSPECT:'데이터 우선 확인'};
  /* Process evidence and data-quality evidence are rendered as two separate
     groups: a process condition and a degraded channel are different problems. */
  /* Pure presentation: every verdict, evidence line, recommendation, gate and
     feed figure is produced by the backend project engine. */
  function renderOperatorActions(){
    const body=$('operatorActionBody');if(!body)return;
    const payload=V2.selectedRow&&V2.selectedRow.operator_actions;
    if(!payload||!Array.isArray(payload.actions)||!payload.actions.length){
      clearOperatorActions('BACKEND OPERATOR ACTION UNAVAILABLE');return;
    }
    const recs=Array.isArray(payload.recommendations)&&payload.recommendations.length
      ?payload.recommendations:null;
    const lamp={ACTION:'t-red',WATCH:'t-amber',NORMAL:'t-green'};
    const label={ACTION:'조치필요',WATCH:'주의',NORMAL:'정상'};
    const card=rec=>'<div class="op-card op-priority-'+escapeHtml(rec.priority)+'">'
      +'<div class="op-card-head"><span class="table-lamp '+(lamp[rec.priority]||'t-gray')+'"></span>'
      +'<b>'+escapeHtml(rec.title)+'</b>'
      +'<span class="op-card-code">'+escapeHtml(label[rec.priority]||rec.priority)+'</span></div>'
      +(rec.evidence&&rec.evidence.length
        ?'<div class="op-block"><div class="op-block-label">근거</div>'
          +rec.evidence.map(line=>'<div class="op-evidence-row">· '+escapeHtml(line)+'</div>').join('')
          +'</div>':'')
      +'<div class="op-block"><div class="op-block-label">권고</div><ol>'
      +(rec.actions||[]).map(line=>'<li>'+escapeHtml(line)+'</li>').join('')+'</ol></div>'
      +(rec.recheck?'<div class="op-recheck">재확인: '+escapeHtml(rec.recheck)+'</div>':'')
      +'</div>';
    // At most three cards on screen, highest severity first; the rest fold away.
    const shown=(recs||[]).slice(0,3),rest=(recs||[]).slice(3);
    const cards=recs
      ?shown.map(card).join('')
      :payload.actions.map(action=>'<div class="op-card op-priority-'+escapeHtml(action.priority)+'">'
        +'<div class="op-card-head"><span class="table-lamp '+(lamp[action.priority]||'t-gray')+'"></span>'
        +'<b>'+escapeHtml(action.title)+'</b></div><ul>'
        +(action.recommendation||[]).slice(0,3).map(line=>'<li>'+escapeHtml(line)+'</li>').join('')
        +'</ul></div>').join('');
    const gate=value=>'<span class="op-gate-'+(value==='CONDITION_VERIFIED'?'VERIFIED':'BLOCKED')+'">'+escapeHtml(value)+'</span>';
    const feed=payload.feed||{};
    const summary='<div class="op-standing"><div><b>운전 상태</b> · '
      +escapeHtml(finite(feed.target_tpd)?'Feed_target '+feed.target_tpd+' t/d':'현재 조건 유지')
      +' · '+escapeHtml(feed.evidence||'')+'</div></div>';
    const details='<details class="op-details"><summary>상세보기 (추가 권고 / RULE / GATE)</summary>'
      +rest.map(card).join('')
      +'<div class="op-provenance">FEED: '+escapeHtml(feed.note||'')+'</div>'
      +'<div class="op-provenance">THERMAL GATE: '+gate(payload.gates&&payload.gates.thermal)
      +' · ALKALI GATE: '+gate(payload.gates&&payload.gates.alkali)
      +' · CONTROL: '+escapeHtml(payload.control_mode)+'</div>'
      +(payload.advisories||[]).map(line=>'<div class="op-provenance">· '+escapeHtml(line)+'</div>').join('')
      +(recs||[]).map(r=>'<div class="op-provenance">'+escapeHtml(r.code)+' · SOURCE '
        +escapeHtml((r.source||[]).join('+'))+'</div>').join('')
      +'</details>';
    const standing=summary+details;
    body.innerHTML=cards+standing;
  }
  function renderDQ(){
    if(!V2.dq)return;const panel=$('dqPanel'),summary=V2.dq.summary;panel.hidden=false;if($('dqDetails'))$('dqDetails').open=false;$('dqStatusBadge').className='dq-status-badge '+((summary.invalid||summary.suspect)?'check':'');text('dqStatusBadge','DQ STATUS: '+((summary.invalid||summary.suspect)?'REVIEW':'VALID'));$('dqSummaryGrid').innerHTML=[['VALID',summary.valid,''],['MISSING',summary.missing,'warn'],['SUSPECT',summary.suspect,'warn'],['INVALID',summary.invalid,'bad'],['ROWS',V2.descriptor.row_count,''],['RAW','PRESERVED',''],['MODEL','NO FIT','']].map(item=>'<div class="dq-summary-cell '+item[2]+'"><span>'+item[0]+'</span><b>'+item[1]+'</b></div>').join('');text('dqDetailsSummary','이상치·결측 상세 보기 ('+V2.dq.flagged_total+')');$('dqTableBody').innerHTML=V2.dq.flagged.slice(0,500).map(row=>'<tr><td>'+row.date+'</td><td>'+row.field+'</td><td>'+(/_A/.test(row.field)?'A':/_B/.test(row.field)?'B':'—')+'</td><td>'+String(row.raw_value===null||row.raw_value===undefined?'':row.raw_value)+'</td><td>'+row.dq_status+'</td><td>'+row.dq_reason+' / ROW '+row.source_row+'</td></tr>').join('');
  }
  async function selectIndex(index){
    if(!V2.datasetId||!V2.history.length)return false;
    const bounded=Math.max(0,Math.min(V2.history.length-1,index)),datasetId=V2.datasetId,date=V2.history[bounded].date,requestVersion=++V2.requestVersion;
    // Claim the target before awaiting, so a burst of clicks steps once per click
    // instead of all reading the same not-yet-updated selectedIndex.
    V2.pendingIndex=bounded;
    try{
      const selectedRow=await window.BioGuardAPI.getRow(datasetId,date);if(requestVersion!==V2.requestVersion||datasetId!==V2.datasetId)return false;
      V2.selectedIndex=bounded;V2.selectedRow=selectedRow;const currentMode=bounded===V2.currentIndex;$('sigDatePicker').value=date;text('sigSelectedReadout','SELECTED: '+date);text('sigModeReadout',currentMode?(forecastAvailable()?'[CURRENT · FORECAST ORIGIN]':'[CURRENT · FORECAST UNAVAILABLE]'):'[HISTORICAL MODE]');text('sigDateValidation','');renderGauges();renderProcessAndParameters(currentMode);renderTelemetry(currentMode);renderOperatorActions();renderCh4TrendRecorder();const health=V2.selectedRow.health;if(health)setLamps(health.lamp,'PROCESS: '+health.process_text,'');return true;
    }catch(error){if(requestVersion!==V2.requestVersion||datasetId!==V2.datasetId)return false;V2.pendingIndex=V2.selectedIndex;console.error('[BioGuard Date Selection] request failed',error);text('sigDateValidation','Backend 날짜 데이터를 불러오지 못했습니다.');return false;}
  }
  function navigationCursor(){
    const cursor=V2.pendingIndex!==null?V2.pendingIndex:V2.selectedIndex;
    return cursor===null?0:cursor;
  }
  function renderStatus(){const review=V2.dq.summary.suspect+V2.dq.summary.invalid;text('sbPlcAddr','LATEST: '+V2.descriptor.latest_data_date);text('sbProtocol','FORECAST ORIGIN: '+(V2.descriptor.forecast_origin_date||'UNAVAILABLE'));text('sbScanRate','DQ: '+(review?'REVIEW '+review:'GOOD'));text('localDataStatus',V2.descriptor.file_name+' · '+V2.descriptor.row_count+' rows · '+(forecastAvailable()?V2.forecast.origin_date+' origin':V2.forecast.status));text('localDataMeta','SOURCE AUTHORITY: UPLOADED EXCEL · BACKEND DATASET');}
  async function loadDataset(datasetId,options){
    const settings=options||{};resetDatasetState('LOADING BACKEND DATASET...');V2.datasetId=datasetId;const loadVersion=V2.requestVersion;
    try{const results=await Promise.all([window.BioGuardAPI.getDataset(datasetId),window.BioGuardAPI.getHistory(datasetId),window.BioGuardAPI.getForecast(datasetId),window.BioGuardAPI.getDataQuality(datasetId)]);if(loadVersion!==V2.requestVersion||datasetId!==V2.datasetId)return false;V2.descriptor=results[0];V2.history=sortUniqueRowsByDate(results[1].rows);V2.forecast=results[2];V2.dq=results[3];const currentDate=forecastAvailable()?V2.forecast.origin_date:V2.descriptor.latest_data_date;V2.currentIndex=V2.history.findIndex(row=>row.date===currentDate);if(V2.currentIndex<0)throw new BioGuardApiError('DATASET_PROCESSING_FAILED','FORECAST ORIGIN NOT FOUND IN DATASET',0);const picker=$('sigDatePicker');picker.min=V2.descriptor.first_date;picker.max=V2.descriptor.latest_data_date;picker.value=currentDate;buildYearButtons();renderTrend();renderCh4TrendRecorder();renderDQ();renderStatus();if(!await selectIndex(V2.currentIndex))throw new BioGuardApiError('DATASET_PROCESSING_FAILED','CURRENT ROW REQUEST DID NOT COMPLETE',0);text('sbProtocol','FORECAST ORIGIN: '+(V2.descriptor.forecast_origin_date||'UNAVAILABLE'));document.documentElement.setAttribute('data-bioguard-v2',forecastAvailable()?'READY':'READY_FORECAST_UNAVAILABLE');if(settings.updateUrl!==false)history.pushState({datasetId},'', '/datasets/'+encodeURIComponent(datasetId));setUploadState('READY','DATASET ONLINE · '+V2.descriptor.file_name+' · '+V2.descriptor.row_count+' rows');return true;}
    catch(error){if(datasetId!==V2.datasetId)return false;resetDatasetState('DATASET LOAD FAILED: '+error.message);setUploadState('ERROR','DATASET_PROCESSING_FAILED · '+error.message);setLamps(null,'SYSTEM STATUS: DATASET UNAVAILABLE',error.message);document.documentElement.setAttribute('data-bioguard-v2','DATASET_LOAD_FAILED');throw error;}
  }
  async function handleBackendFileUpload(file){
    if(!V2.connected){setUploadState('OFFLINE','SERVER OFFLINE · 업로드 불가');return;}
    try{
      validateUploadFileClientSide(file);history.replaceState({},'', '/');resetDatasetState('UPLOADING...');const uploadVersion=V2.requestVersion;setUploadState('PROCESSING','UPLOADING... · VALIDATING DATA · CHECKING DQ · RUNNING MODEL');
      const response=await uploadBioGuardDataset(file);if(uploadVersion!==V2.requestVersion)return;if(!response||!response.dataset_id)throw new BioGuardApiError('DATASET_ID_MISSING','Backend response에 dataset_id가 없습니다.',0);
      await loadDataset(response.dataset_id,{updateUrl:true});
    }catch(error){const code=uploadErrorCode(error);resetDatasetState('UPLOAD FAILED · '+code);setUploadState('ERROR','UPLOAD FAILED · '+code+(error.message?' · '+error.message:''));setLamps(null,'SYSTEM STATUS: DATASET REJECTED',code);document.documentElement.setAttribute('data-bioguard-v2','UPLOAD_FAILED');}
  }
  function bindBackendUpload(){
    const input=$('localDataFileInput'),button=$('btnLocalDataSelect'),rack=$('controlRack');if(!input||!button||!rack)throw new Error('UPLOAD_UI_NOT_FOUND');
    button.addEventListener('click',event=>{event.preventDefault();event.stopImmediatePropagation();if(!V2.connected){setUploadState('OFFLINE','SERVER OFFLINE · 업로드 불가');return;}input.value='';input.click();},true);
    input.addEventListener('change',event=>{event.stopImmediatePropagation();const file=event.target.files&&event.target.files[0];if(file)void handleBackendFileUpload(file);input.value='';},true);
    rack.addEventListener('dragover',event=>{event.preventDefault();event.stopImmediatePropagation();rack.classList.add('dragover');},true);rack.addEventListener('dragleave',()=>rack.classList.remove('dragover'),true);rack.addEventListener('drop',event=>{event.preventDefault();event.stopImmediatePropagation();rack.classList.remove('dragover');const file=event.dataTransfer&&event.dataTransfer.files&&event.dataTransfer.files[0];if(file)void handleBackendFileUpload(file);},true);
  }
  function scheduleBackendRetry(){clearTimeout(V2.healthRetryTimer);if(V2.connected)return;V2.healthRetryTimer=setTimeout(()=>{void connectBackend();},HEALTH_RETRY_MS);}
  async function connectBackend(){
    setUploadState('CONNECTING','CHECKING BACKEND CONNECTION...');
    try{const health=await window.BioGuardAPI.healthCheck();if(health.status!=='CONNECTED')throw new BioGuardApiError('MODEL_NOT_READY',health.model_error||health.status,0);setMode(true);clearTimeout(V2.healthRetryTimer);}
    catch(error){setMode(false);resetDatasetState('SERVER OFFLINE · '+error.message);setUploadState('OFFLINE','SERVER OFFLINE · 업로드 불가');setLamps(null,'SYSTEM STATUS: SERVER OFFLINE','FORECAST UNAVAILABLE');scheduleBackendRetry();return;}
    const match=location.pathname.match(/^\/datasets\/([^/]+)$/);if(match){try{await loadDataset(decodeURIComponent(match[1]),{updateUrl:false});}catch(_) {}}else{resetDatasetState('SERVER CONNECTED · 소화조 데이터를 업로드해주세요.');setUploadState('NO_DATASET','SERVER CONNECTED · 소화조 데이터를 업로드해주세요.');}
  }
  function showView(name){document.querySelectorAll('.view-panel').forEach(panel=>panel.classList.toggle('active',panel.dataset.view===name));document.querySelectorAll('.nav-btn[data-view]').forEach(button=>button.classList.toggle('active',button.dataset.view===name));try{sessionStorage.setItem('bioguardActiveView',name);}catch(_) {}}
  function bind(){
    if($('btnLocalDataReset'))$('btnLocalDataReset').style.display='none';
    $('navMenu').addEventListener('click',event=>{const button=event.target.closest('button[data-view]');if(!button||button.disabled)return;event.preventDefault();event.stopImmediatePropagation();showView(button.dataset.view);},true);
    [['btnSigPrevDate',()=>selectIndex(navigationCursor()-1)],['btnSigNextDate',()=>selectIndex(navigationCursor()+1)],['btnSigCurrent',()=>selectIndex(V2.currentIndex)],['btnSigSelectDate',()=>{const date=$('sigDatePicker').value,index=V2.history.findIndex(row=>row.date===date);if(index>=0)selectIndex(index);else text('sigDateValidation','Backend dataset에 없는 날짜입니다.');}],['btnPrevDate',()=>selectIndex(navigationCursor()-1)],['btnNextDate',()=>selectIndex(navigationCursor()+1)]].forEach(pair=>$(pair[0]).addEventListener('click',event=>{if(!V2.datasetId)return;event.preventDefault();event.stopImmediatePropagation();pair[1]();},true));
    document.querySelectorAll('.range-btn').forEach(button=>button.addEventListener('click',event=>{if(!V2.datasetId)return;event.preventDefault();event.stopImmediatePropagation();V2.trendYear=null;V2.trendMode=button.dataset.range;renderTrend();},true));window.addEventListener('popstate',()=>{const match=location.pathname.match(/^\/datasets\/([^/]+)$/);if(match&&V2.connected)loadDataset(decodeURIComponent(match[1]),{updateUrl:false}).catch(()=>{});else resetDatasetState();});
  }
  document.addEventListener('DOMContentLoaded',()=>{bindBackendUpload();bind();setMode(false);resetDatasetState('CHECKING BACKEND CONNECTION...');void connectBackend();});
})();
