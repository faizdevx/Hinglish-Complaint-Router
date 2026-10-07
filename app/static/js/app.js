/* Vanilla JS for the dashboard. No dependencies; every value shown comes from the API. */
(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var esc = function (v) { return String(v == null ? '' : v).replace(/[&<>"']/g, function (c) { return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); };
  var BANDS = []; try { BANDS = JSON.parse(document.body.dataset.bands || '[]'); } catch (e) {}

  /* ---- theme + nav ---- */
  var themeBtn = $('#theme-btn');
  if (themeBtn) themeBtn.addEventListener('click', function () {
    var cur = document.documentElement.getAttribute('data-theme') ||
      (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    var next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) {}
  });
  var menuBtn = $('#menu-btn'), sidebar = $('#sidebar');
  if (menuBtn) menuBtn.addEventListener('click', function () {
    var open = sidebar.classList.toggle('open'); menuBtn.setAttribute('aria-expanded', open);
  });

  /* ---- toasts ---- */
  function toast(msg, kind) {
    var box = $('#toasts'); if (!box) return;
    var t = document.createElement('div'); t.className = 'toast ' + (kind || ''); t.textContent = msg; box.appendChild(t);
    setTimeout(function () { t.remove(); }, kind === 'error' ? 7000 : 3500);
  }

  /* ---- fetch helper: always resolves to {ok, data, error} ---- */
  function api(path, opts, timeoutMs) {
    var ctl = new AbortController(), timer = setTimeout(function () { ctl.abort(); }, timeoutMs || 120000);
    opts = opts || {}; opts.signal = ctl.signal;
    return fetch(path, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (d) {
        if (!r.ok) return { ok: false, status: r.status, error: (d && d.detail) || 'Prediction service is unavailable.' };
        return { ok: true, data: d };
      });
    }).catch(function (e) {
      return { ok: false, error: e.name === 'AbortError' ? 'Request timed out.' : 'Prediction service is unavailable.' };
    }).then(function (res) { clearTimeout(timer); return res; });
  }
  function postJSON(path, body) {
    return api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  }
  function busy(btn, on) {
    if (!btn) return;
    btn.disabled = on;
    btn.innerHTML = on ? '<span class="spinner" aria-hidden="true"></span>Analyzing…' : esc(btn.dataset.label || 'Run');
  }
  function errToast(res) { toast(res.error, res.status === 503 ? 'warn' : 'error'); }

  /* ---- quick-fill example buttons ---- */
  document.addEventListener('click', function (e) {
    var b = e.target.closest('[data-fill]'); if (!b) return;
    var ta = $('#text'); if (ta) { ta.value = b.dataset.fill; ta.focus(); }
  });

  /* ---- rendering ---- */
  function pct(p) { return (p * 100).toFixed(1) + '%'; }
  function bandOf(p) { return p < BANDS[0][0] ? 'low' : p < BANDS[1][0] ? 'medium' : 'high'; }
  function meter(p) {
    var cells = ''; for (var i = 0; i < 10; i++) cells += '<span class="' + ((i + 0.5) / 10 <= p ? 'on' : '') + '"></span>';
    var band = bandOf(p);
    return '<div class="meter ' + band + '" role="img" aria-label="Escalation probability ' + pct(p) + ', band ' + band + '">' + cells + '</div><span class="band">' + band.toUpperCase() + '</span>';
  }
  function metric(l, v) { return '<div class="metric"><div class="l">' + esc(l) + '</div><div class="v">' + v + '</div></div>'; }
  function bars(items) {
    return '<div class="bars">' + items.map(function (it) {
      return '<div class="barrow"><span class="lab">' + esc(it.label) + '</span><div class="track" role="img" aria-label="' + esc(it.label) + ' ' + it.score.toFixed(3) + '"><div class="fill" style="width:' + (it.score * 100).toFixed(1) + '%"></div></div><span class="num">' + it.score.toFixed(3) + '</span></div>';
    }).join('') + '</div>';
  }
  function explain(r) {
    var top = r.intent_scores[0], urg = r.urgency_scores[r.urgency];
    var li = [
      'Intent is <strong>' + esc(r.intent) + '</strong> because this class received the highest model score (' + top.score.toFixed(3) + ', softmax over ' + 77 + ' classes). Scores are not calibrated probabilities of being correct.',
      'Urgency is <strong>' + esc(r.urgency) + '</strong> because it was the highest-scoring of the 5 urgency levels (' + urg.toFixed(3) + '). This head was trained on ~120 synthetic Hindi/Hinglish texts.',
      'Escalation probability <strong>' + pct(r.escalation_probability) + '</strong> is the sigmoid of the escalation logit' + (r.calibrated ? ' after temperature scaling (raw ' + pct(r.escalation_probability_uncalibrated) + ')' : '') + '. Band ' + bandOf(r.escalation_probability).toUpperCase() + ' uses fixed display thresholds (' + BANDS[0][0] + ' / ' + BANDS[1][0] + '); “high” describes the probability, not the model’s confidence.',
      'Route <strong>' + esc(r.route) + '</strong> is a deterministic lookup from the predicted intent (not a separately learned output).',
      'Language <strong>' + esc(r.language) + '</strong> / script <strong>' + esc(r.script) + '</strong>: ' + esc(r.language_source) + '.'
    ];
    return '<details><summary>Explain result</summary><ul class="explain">' + li.map(function (x) { return '<li>' + x + '</li>'; }).join('') + '</ul></details>';
  }
  function resultCard(r) {
    return '<div class="result-grid">' +
      metric('Intent', esc(r.intent)) + metric('Urgency', esc(r.urgency)) +
      metric('Escalation probability', pct(r.escalation_probability) + '<br>' + meter(r.escalation_probability)) +
      metric('Language / script', esc(r.language) + ' / ' + esc(r.script)) + metric('Route', esc(r.route)) + '</div>' +
      '<h3>Top intent scores (model output)</h3>' + bars(r.intent_scores) +
      '<dl class="kv small" style="margin-top:14px"><dt>Model</dt><dd>' + esc(r.model_name) + '</dd><dt>Checkpoint</dt><dd class="mono">' + esc(r.checkpoint) + '</dd><dt>Device</dt><dd>' + esc(r.device) + '</dd><dt>Inference time</dt><dd>' + r.inference_ms.toFixed(1) + ' ms (tokenise + forward, excludes HTTP)</dd></dl>' + explain(r);
  }
  function compareCard(r) {
    if (!r.ok) return '<section class="card"><h2>' + esc(r.model_name) + '</h2><div class="notice bad">Model unavailable. ' + esc(r.error) + '</div></section>';
    return '<section class="card"><h2>' + esc(r.model_name) + '</h2><dl class="kv"><dt>Intent</dt><dd><strong>' + esc(r.intent) + '</strong></dd><dt>Intent score</dt><dd>' + r.intent_scores[0].score.toFixed(3) + '</dd><dt>Urgency</dt><dd><strong>' + esc(r.urgency) + '</strong></dd><dt>Escalation</dt><dd><strong>' + pct(r.escalation_probability) + '</strong></dd><dt>Route</dt><dd>' + esc(r.route) + '</dd><dt>Language</dt><dd>' + esc(r.language) + '</dd><dt>Latency</dt><dd>' + r.inference_ms.toFixed(1) + ' ms</dd></dl>' + meter(r.escalation_probability) + '<h3>Top intent scores</h3>' + bars(r.intent_scores.slice(0, 3)) + '</section>';
  }

  /* ---- classify ---- */
  var cf = $('#classify-form');
  if (cf) cf.addEventListener('submit', function (e) {
    e.preventDefault();
    var text = $('#text').value;
    if (!text.trim()) { toast('Enter at least one non-empty message.', 'warn'); return; }
    var btn = $('#analyze-btn'); busy(btn, true);
    postJSON('/api/predict', { text: text, model: $('#model').value, language: $('#language').value }).then(function (res) {
      busy(btn, false);
      if (!res.ok) { errToast(res); return; }
      $('#result').hidden = false; $('#result-body').innerHTML = resultCard(res.data); toast('Prediction complete', 'success');
      $('#result').scrollIntoView({ block: 'nearest' });
    });
  });

  /* ---- playground ---- */
  var pf = $('#compare-form');
  if (pf) pf.addEventListener('submit', function (e) {
    e.preventDefault();
    var text = $('#text').value;
    if (!text.trim()) { toast('Enter at least one non-empty message.', 'warn'); return; }
    var btn = $('#compare-btn'); busy(btn, true);
    postJSON('/api/compare', { text: text, language: $('#language').value }).then(function (res) {
      busy(btn, false);
      if (!res.ok) { errToast(res); return; }
      var out = res.data.results, keys = Object.keys(out), html = '', unavailable = 0;
      keys.forEach(function (k) { html += compareCard(out[k]); if (!out[k].ok) unavailable++; });
      $('#compare-out').innerHTML = html;
      if (unavailable) toast('A model is unavailable.', 'warn'); else toast('Prediction complete', 'success');
    });
  });

  /* ---- batch ---- */
  var bf = $('#batch-form'), lastCsv = '';
  if (bf) {
    bf.addEventListener('submit', function (e) {
      e.preventDefault();
      var f = $('#file').files[0];
      if (!f) { toast('Choose a CSV file first.', 'warn'); return; }
      toast('CSV uploaded', '');
      var btn = $('#batch-btn'); busy(btn, true);
      var fd = new FormData(); fd.append('file', f); fd.append('model', $('#model').value);
      api('/api/batch', { method: 'POST', body: fd }, 300000).then(function (res) {
        busy(btn, false);
        if (!res.ok) { errToast(res); return; }
        var d = res.data; lastCsv = d.csv;
        $('#batch-result').hidden = false;
        $('#batch-summary').innerHTML = [['Inputs', d.n_inputs], ['Successful', d.n_success], ['Failed rows', d.n_failed],
          ['Mean inference', d.mean_inference_ms == null ? 'n/a' : d.mean_inference_ms.toFixed(1) + ' ms'], ['Total time', d.total_seconds.toFixed(2) + ' s']]
          .map(function (x) { return '<div class="card"><h2>' + x[0] + '</h2><div class="big">' + esc(x[1]) + '</div></div>'; }).join('');
        $('#batch-failed').innerHTML = d.failed.length ? '<div class="notice bad"><strong>' + d.failed.length + ' row(s) failed:</strong><ul>' + d.failed.slice(0, 10).map(function (x) { return '<li>ID ' + esc(x.id) + ': ' + esc(x.error) + '</li>'; }).join('') + '</ul></div>' : '';
        $('#batch-rows').innerHTML = d.results.slice(0, 200).map(function (r) {
          return '<tr><td>' + esc(r.id) + '</td><td class="text">' + esc(r.text) + '</td><td>' + esc(r.language) + '</td><td>' + esc(r.intent) + '</td><td>' + esc(r.urgency) + '</td><td class="num">' + pct(r.escalation_probability) + '</td><td>' + esc(r.route) + '</td></tr>';
        }).join('');
        toast('Batch completed: ' + d.n_success + '/' + d.n_inputs + ' predicted', d.n_failed ? 'warn' : 'success');
      });
    });
    var dl = $('#download-btn');
    if (dl) dl.addEventListener('click', function () {
      if (!lastCsv) return;
      var a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([lastCsv], { type: 'text/csv;charset=utf-8' }));
      a.download = 'predictions.csv'; document.body.appendChild(a); a.click(); a.remove();
    });
  }

  /* ---- error table ---- */
  var rowsEl = $('#err-rows');
  if (rowsEl) {
    var offset = 0, LIMIT = 50, timer = null;
    var load = function (reset) {
      if (reset) { offset = 0; rowsEl.innerHTML = ''; }
      var p = new URLSearchParams({ limit: LIMIT, offset: offset });
      [['q', '#q'], ['model', '#f-model'], ['language', '#f-lang'], ['intent', '#f-intent']].forEach(function (x) { var v = $(x[1]).value; if (v) p.set(x[0], v); });
      api('/api/errors?' + p.toString()).then(function (res) {
        if (!res.ok) { errToast(res); return; }
        var d = res.data;
        if (!d.available) { $('#err-count').textContent = d.message; return; }
        rowsEl.insertAdjacentHTML('beforeend', d.rows.map(function (r) {
          var esc_ = r.true_escalation === '' ? '–' : (r.true_escalation === '1' ? 'yes' : 'no') + ' / ' + (r.pred_escalation === '1' ? 'yes' : 'no') + ', p=' + Number(r.escalation_probability).toFixed(2);
          return '<tr><td>' + esc(r.model) + '</td><td class="text">' + esc(r.text) + '</td><td>' + esc(r.language) + '</td><td>' + esc(r.true_intent || '–') + '</td><td>' + esc(r.pred_intent || '–') + '</td><td>' + esc(r.true_urgency || '–') + '</td><td>' + esc(r.pred_urgency || '–') + '</td><td>' + esc(esc_) + '</td></tr>';
        }).join(''));
        offset += d.rows.length;
        $('#err-count').textContent = d.total ? 'Showing ' + offset + ' of ' + d.total + ' failed examples' : 'No failed examples match these filters.';
        $('#err-more').hidden = offset >= d.total;
      });
    };
    ['#q', '#f-model', '#f-lang', '#f-intent'].forEach(function (s) {
      $(s).addEventListener('input', function () { clearTimeout(timer); timer = setTimeout(function () { load(true); }, 250); });
    });
    $('#err-filters').addEventListener('submit', function (e) { e.preventDefault(); });
    $('#err-more').addEventListener('click', function () { load(false); });
    load(true);
  }
})();
