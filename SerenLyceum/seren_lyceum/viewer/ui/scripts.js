// ── SerenLyceum viewer ────────────────────────────────────────────────────────
// Snaps onto the SerenMeninges shell, which provides api(), escapeHtml(),
// showTab(), getToken() and the token modal. Written for someone who has never
// trained a model: every choice is a card with a sentence on it, every number
// has a plain name, and a running training says what it is doing in words.

const L = { overview: null, jobs: [], datasets: [], sel: null, selDs: null, pick: {}, poll: null,
            job: null, imported: {}, modal: null };
const $ = (id) => document.getElementById(id);
const STAGE_SAY = { prepare: "Reading your data", download: "Getting the model", train: "Learning",
                    evaluate: "Testing itself", export: "Saving the result" };
const STATE_SAY = { queued: "waiting its turn", starting: "starting", running: "training", done: "done",
                    failed: "stopped with a problem", stopped: "stopped" };
const LIVE = ["queued", "starting", "running"];
const ADV = {
    epochs: ["Passes over the data", "epochs"], learning_rate: ["How big each lesson's step is", "learning rate"],
    lora_rank: ["How much new it can hold", "LoRA rank"], max_length: ["Longest example it reads", "max tokens"],
    batch_size: ["Examples at once", "batch size"], grad_accum: ["Steps added up per update", "gradient accumulation"],
    eval_fraction: ["Share held back to test with", "eval fraction"],
};

function err(msg) { $("globalErr").innerHTML = msg ? `<div class="err">⚠ ${escapeHtml(msg)}</div>` : ""; }

// The shell's api() says "400 Bad Request"; this one says what the service said.
async function call(path, opts = {}) {
    const headers = Object.assign({}, opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
                                  opts.headers || {});
    const tok = getToken();
    if (tok) headers["Authorization"] = "Bearer " + tok;
    const res = await fetch(path, Object.assign({}, opts, { headers }));
    const ct = res.headers.get("content-type") || "";
    const body = ct.includes("application/json") ? await res.json().catch(() => ({})) : await res.text();
    if (!res.ok) throw new Error((body && body.detail) || `${res.status} ${res.statusText}`);
    return body;
}

function ago(ts) {
    if (!ts) return "";
    const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 172800) return `${Math.round(s / 3600)} h ago`;
    return `${Math.round(s / 86400)} days ago`;
}
function dur(sec) {
    if (sec == null) return "";
    sec = Math.round(sec);
    if (sec < 90) return `${sec} s`;
    if (sec < 5400) return `${Math.round(sec / 60)} min`;
    return `${(sec / 3600).toFixed(1)} h`;
}
function human(n) {
    n = Number(n || 0);
    for (const u of ["B", "KB", "MB", "GB", "TB"]) { if (n < 1024 || u === "TB") return (u === "B" ? n : n.toFixed(1)) + " " + u; n /= 1024; }
}
function num(n) { return Number(n || 0).toLocaleString(); }
function localCheck() { return ((L.overview && L.overview.backends.find(b => b.name === "local")) || {}).check || {}; }
// Same families the service marks; for a model typed by hand.
function needsLogin(id) {
    const s = (L.overview.models.find(m => m.id === id) || {});
    if (s.id) return !!s.needs_login;
    return /^(meta-llama\/|google\/gemma|mistralai\/mistral-large)/i.test(id || "");
}

// ── downloads: ask for a short-lived link, then let the browser fetch it ────
// (a click can't carry the token, and fetching a 3 GB model into memory first
// looked like a dead click; see links.py)
async function download(kind, id, path) {
    try {
        const r = await call("/api/link", { method: "POST", body: JSON.stringify({ kind, id, path: path || "" }) });
        const a = document.createElement("a");
        a.href = r.url; a.rel = "noopener";
        document.body.appendChild(a); a.click(); a.remove();
    } catch (e) { err(`Couldn't download it: ${e.message}`); }
}
document.addEventListener("click", (ev) => {
    const el = ev.target.closest("[data-dl]");
    if (!el) return;
    ev.preventDefault();
    download(el.dataset.dl, el.dataset.id, el.dataset.path);
});
function dl(kind, id, label, path = "", cls = "") {
    return `<a href="#" class="${cls}" data-dl="${kind}" data-id="${escapeHtml(id)}" data-path="${escapeHtml(path)}">${label}</a>`;
}

// ── the one modal ─────────────────────────────────────────────────────────────
function openModal(title, body, actions = "", kind = null) {
    $("modalTitle").textContent = title;
    $("modalActions").innerHTML = actions;
    $("modalBody").innerHTML = body;
    $("modal").hidden = false;
    L.modal = kind;
    document.body.classList.add("modal-open");
}
function closeModal() {
    $("modal").hidden = true;
    L.modal = null;
    document.body.classList.remove("modal-open");
}
$("modal").addEventListener("click", (ev) => { if (ev.target === $("modal")) closeModal(); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && !$("modal").hidden) closeModal(); });

// ── the box ───────────────────────────────────────────────────────────────────
async function loadOverview() {
    L.overview = await call("/api/overview");
    const local = L.overview.backends.find(b => b.name === "local");
    const pill = $("boxPill");
    pill.textContent = local ? (local.ready ? `${local.label}: ready` : `${local.label}: not set up`) : "no backends";
    pill.className = "head-pill" + (local && local.ready ? "" : " hot");
    $("count-running").textContent = L.overview.counts.running || "";
}

// The extras (seren_lyceum.extras): every one this Lyceum knows, whether this
// box has what it needs, and a switch for whether New training offers it.
function extrasCard() {
    const xs = L.overview.extras || [];
    return `<div class="box-card wide">
        <div class="bc-h">Extras</div>
        <p>Things a training can switch on that aren't the training itself. None is needed. Switched off here, it isn't
           offered in New training. Add your own to <code>seren-lyceum.yaml</code> under <code>extras:</code> (the sample shows how).</p>
        <ul class="xlist">${xs.map(e => `<li class="${e.available === false ? "not" : "has"}">
            <input type="checkbox" data-extra="${escapeHtml(e.id)}" ${e.on ? "checked" : ""} title="offer it in New training" />
            <div><b>${escapeHtml(e.name)}</b>
                <span class="chip">${e.available === false ? "not on this box" : e.version ? escapeHtml(e.version) : e.available ? "always there" : "not checked"}</span>
                ${e.builtin ? "" : `<span class="chip q">yours</span>`}
                <div class="muted small">${escapeHtml(e.say || "")}</div>
                ${e.available === false && e.install ? `<div class="muted small">To get it: ${escapeHtml(e.install)}, into the training environment, then Check again.</div>` : ""}
            </div></li>`).join("")}</ul>
    </div>`;
}

function loginCard() {
    const h = L.overview.hf_login || {};
    return `<div class="box-card ${h.found ? "ok" : "off"}">
        <div class="bc-h"><span class="dot"></span>Hugging Face login</div>
        <p>${h.found ? `Found: ${escapeHtml(h.where)}. Models that need a login (Llama, Gemma) will use it.`
            : `None on this box. Most models don't need one; Llama and Gemma do, from an account that has accepted
               the model's licence. If node setup was given one it would be here already.`}</p>
        ${loginForm(h.found ? "Replace it" : "Save it")}
    </div>`;
}
function loginForm(label) {
    return `<div class="login-form"><input type="password" class="hfTok" autocomplete="off" spellcheck="false"
        placeholder="hf_…" />
        <button class="btn" onclick="saveLogin(this)">${label}</button>
        <div class="muted small">Make one at huggingface.co/settings/tokens (read access is enough). Kept where Hugging Face keeps it on this box, owner-only; never shown again.</div></div>`;
}
async function saveLogin(btn) {
    const inp = btn.parentElement.querySelector(".hfTok");
    const token = inp.value.trim();
    if (!token) return;
    btn.disabled = true;
    try {
        await call("/api/hf-token", { method: "POST", body: JSON.stringify({ token }) });
        inp.value = "";
        await loadOverview();
        if ($("box").classList.contains("active")) renderBox(); else renderNew();
    } catch (e) { err(e.message); btn.disabled = false; }
}

function renderBox() {
    const o = L.overview;
    if (!o) return;
    $("boxGrid").innerHTML = o.backends.map(b => {
        const c = b.check || {};
        const rows = b.kind === "local" ? [
            ["GPU", c.gpu || "none found"], ["GPU memory", c.gpu_memory_gb ? `${c.gpu_memory_gb} GB` : "-"],
            ["Python", c.python || "-"], ["torch", c.torch || "missing"], ["transformers", c.transformers || "missing"],
            ["peft", c.peft || "missing"], ["trl", c.trl || "missing"],
        ] : [["Cost", b.cost || "-"]];
        return `<div class="box-card ${b.ready ? "ok" : "off"}">
            <div class="bc-h"><span class="dot"></span>${escapeHtml(b.label)}
              <span class="chip">${b.kind === "local" ? "this box" : "cloud"}</span>
              ${b.coming ? `<span class="chip soon">coming next</span>` : ""}</div>
            <p>${escapeHtml(b.say)}</p>
            <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("")}</dl>
            ${b.kind === "local" ? `<button class="btn" onclick="recheck('${b.name}')">Check again</button>` : ""}
        </div>`;
    }).join("") + loginCard() + extrasCard();
}

async function recheck(name) {
    $("boxGrid").innerHTML = `<div class="empty">Asking again…</div>`;
    await call(`/api/backends/${name}/check`, { method: "POST" });
    await loadOverview();
    renderBox();
}

// ── trainings ────────────────────────────────────────────────────────────────
async function loadJobs() {
    L.jobs = (await call("/api/jobs")).jobs;
    renderJobList();
    if (!L.sel && L.jobs.length) selectJob(L.jobs[0].id);
    else if (L.sel) loadJob(L.sel);
}

function stateChip(st) {
    const s = (st || {}).state || "queued";
    return `<span class="state ${s}">${escapeHtml(STATE_SAY[s] || s)}</span>`;
}

function jobProgress(st) {
    if (!st) return 0;
    if (st.state === "done") return 1;
    const si = st.stage_index ?? -1;
    if (si < 0) return 0;
    const inTrain = st.total ? (st.step || 0) / st.total : 0;
    const weights = [0.03, 0.07, 0.75, 0.07, 0.08];
    let p = weights.slice(0, si).reduce((a, b) => a + b, 0);
    p += weights[si] * (si === 2 ? inTrain : 0.5);
    return Math.min(1, p);
}

function renderJobList() {
    const el = $("jobList");
    el.parentElement.classList.toggle("solo", !L.jobs.length);
    if (!L.jobs.length) { el.innerHTML = ""; return; }
    el.innerHTML = L.jobs.map(j => {
        const st = j.status || {};
        const pct = Math.round(jobProgress(st) * 100);
        return `<button class="job-row ${L.sel === j.id ? "sel" : ""}" onclick="selectJob('${j.id}')">
            <div class="jr-top"><span class="jr-name">${escapeHtml(j.name)}</span>${stateChip(st)}</div>
            <div class="jr-sub">${escapeHtml(j.base_model.split("/").pop())} · ${escapeHtml(j.dataset)} · ${ago(j.created_at)}</div>
            <div class="bar"><i style="width:${pct}%"></i></div>
        </button>`;
    }).join("");
}

function selectJob(id) {
    L.sel = id;
    renderJobList();
    loadJob(id);
}

async function loadJob(id) {
    let j;
    try { j = await call(`/api/jobs/${id}`); } catch (e) { if (L.sel === id) L.sel = null; return; }
    if (L.sel !== id) return;
    renderJob(j);
}

// The learning curve. Small in the page, big in the modal; both say what the
// numbers are on both axes.
function lossChart(metrics, big = false) {
    const pts = metrics.filter(m => typeof m.loss === "number" && m.step != null);
    const ev = metrics.filter(m => typeof m.eval_loss === "number" && m.step != null);
    if (pts.length < 2) return `<div class="chart empty-chart">The learning curve appears once it starts learning.</div>`;
    const W = big ? 1000 : 420, H = big ? 440 : 210, PL = 42, PR = 12, PT = 22, PB = big ? 40 : 36;
    const xs = [...pts, ...ev].map(m => m.step), ys = [...pts.map(m => m.loss), ...ev.map(m => m.eval_loss)];
    const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = (y1 - y0) * 0.06 || 0.05;
    y0 = Math.max(0, y0 - pad); y1 += pad;
    const X = s => PL + (W - PL - PR) * ((s - x0) / Math.max(1, x1 - x0));
    const Y = v => H - PB - (H - PT - PB) * ((v - y0) / Math.max(1e-9, y1 - y0));
    const yt = ticks(y0, y1, big ? 6 : 4), xt = ticks(x0, x1, big ? 8 : 4).filter(t => Number.isInteger(t));
    const line = pts.map(m => `${X(m.step).toFixed(1)},${Y(m.loss).toFixed(1)}`).join(" ");
    const ydigits = (y1 - y0) < 0.5 ? 3 : 2;
    return `<svg class="chart ${big ? "big" : ""}" viewBox="0 0 ${W} ${H}" role="img" aria-label="mistakes over the training">
        <text x="${PL}" y="13" class="ax">mistakes (loss) · lower is better</text>
        ${yt.map(t => `<line x1="${PL}" x2="${W - PR}" y1="${Y(t).toFixed(1)}" y2="${Y(t).toFixed(1)}" class="grid"/>
            <text x="${PL - 6}" y="${(Y(t) + 4).toFixed(1)}" class="ax" text-anchor="end">${t.toFixed(ydigits)}</text>`).join("")}
        ${xt.map(t => `<text x="${X(t).toFixed(1)}" y="${H - PB + 16}" class="ax" text-anchor="middle">${num(t)}</text>`).join("")}
        <text x="${(PL + W - PR) / 2}" y="${H - 3}" class="ax" text-anchor="middle">step</text>
        <line x1="${PL}" y1="${H - PB}" x2="${W - PR}" y2="${H - PB}" class="axis"/>
        <line x1="${PL}" y1="${PT}" x2="${PL}" y2="${H - PB}" class="axis"/>
        <polyline points="${line}" class="loss"/>
        ${ev.map(m => `<circle cx="${X(m.step).toFixed(1)}" cy="${Y(m.eval_loss).toFixed(1)}" r="${big ? 5 : 4}" class="evalpt"><title>step ${m.step}, on held-back examples: ${m.eval_loss.toFixed(3)}</title></circle>`).join("")}
        ${big ? pts.map(m => `<circle cx="${X(m.step).toFixed(1)}" cy="${Y(m.loss).toFixed(1)}" r="7" class="hit"><title>step ${m.step}: ${m.loss.toFixed(3)}</title></circle>`).join("") : ""}
    </svg>
    <div class="legend"><span><i class="lg-loss"></i>while learning</span>${ev.length ? `<span><i class="lg-eval"></i>on examples it never saw</span>` : ""}
        </div>`;
}
function ticks(a, b, n) {
    const span = b - a || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => s >= raw) || raw;
    const out = [];
    for (let t = Math.ceil(a / step) * step; t <= b + 1e-9; t += step) out.push(+t.toFixed(10));
    return out;
}
function bigChart() {
    const j = L.job;
    if (!j) return;
    openModal(`${j.spec.name}: how it learned`, lossChart(j.metrics || [], true), "", `chart:${j.id}`);
}

function stageTrack(st) {
    const si = st.state === "done" ? 5 : (st.stage_index ?? -1);
    const failed = st.state === "failed" || st.state === "stopped";
    return `<ol class="stages">${L.overview.stages.map((s, i) => {
        const cls = i < si ? "past" : i === si ? (failed ? "bad" : "now") : "";
        return `<li class="${cls}"><span class="sd"></span>${STAGE_SAY[s] || s}</li>`;
    }).join("")}</ol>`;
}

function logHtml(lines) {
    return lines.slice(-400).map(l => {
        const k = (l.match(/^\[SEREN:(\w+)\]/) || [])[1];
        return k ? `<b class="pl ${k.toLowerCase()}">${escapeHtml(l)}</b>` : escapeHtml(l);
    }).join("\n");
}

function renderJob(j) {
    const st = j.status || {}, sp = j.spec || {}, p = sp.params || {};
    const live = LIVE.includes(st.state);
    let line = st.message || "";
    if (st.stage === "train" && st.total && st.state === "running") {
        line = `Learning: step ${num(st.step)} of ${num(st.total)}` + (st.eta_seconds ? ` · about ${dur(st.eta_seconds)} left` : "");
    }
    if (st.state === "done") {
        const made = (st.artifacts || []).find(a => a.endsWith(".gguf")) || (st.artifacts || [])[0];
        line = `Done in ${dur((st.ended_at || 0) - (st.started_at || 0))}. It learned from ${num(st.rows_train || 0)} examples` +
               (made ? `; what it made is below${made.endsWith(".gguf") ? ` (${made})` : ""}.` : ".");
    }
    const where = (L.overview.backends.find(b => b.name === sp.backend) || {}).label || sp.backend;
    const elapsed = st.started_at ? ((st.ended_at || Date.now() / 1000) - st.started_at) : null;
    const facts = [
        ["Started from", sp.base_model], ["Learning", `${sp.dataset}` + (st.rows_train ? ` · ${num(st.rows_train)} examples, ${num(st.rows_eval || 0)} held back` : "")],
        ...(sp.test_dataset ? [["Tested on", sp.test_dataset]] : []),
        ["Where", where], ["How hard", `${sp.effort} · ${p.epochs} pass${p.epochs === 1 ? "" : "es"} over the data`],
        ["Coming out as", sp.export && sp.export.quant === "none" ? "the merged model + adapter" : `GGUF, ${(sp.export || {}).quant}`],
        ["Time", elapsed ? dur(elapsed) : "-"],
    ];
    if (p.attention === "flash_attention_2" || p.load_4bit) {
        facts.push(["Extras", [p.attention === "flash_attention_2" && "flash attention", p.load_4bit && "4-bit loading"].filter(Boolean).join(", ")]);
    }
    if (st.loss != null) facts.push(["Mistakes now", `${Number(st.loss).toFixed(3)} while learning` + (st.eval_loss != null ? ` · ${Number(st.eval_loss).toFixed(3)} on held-back` : "")]);
    const samples = j.samples || [];
    const logLines = j.log || [];

    // A re-render every beat must not undo what the person did: folds they
    // opened stay open, folds they closed stay closed, and the log stays where
    // they scrolled it (or follows the end, if that's where they were).
    const host = $("jobDetail");
    const same = L.job && L.job.id === j.id;
    const keep = {};
    let logAt = null;
    if (same) {
        host.querySelectorAll("details[data-k]").forEach(d => { keep[d.dataset.k] = d.open; });
        const pre = host.querySelector(".logbox pre");
        if (pre) logAt = { top: pre.scrollTop, bottom: pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24 };
    }
    L.job = j;
    const isOpen = (k, dflt) => (k in keep ? keep[k] : dflt);
    const op = (k, dflt = false) => (isOpen(k, dflt) ? "open" : "");

    host.innerHTML = `
        <div class="jd-head">
            <div><h2>${escapeHtml(sp.name)}</h2><div class="muted small">${escapeHtml(j.id)}${st.fake ? " · fake run" : ""}</div></div>
            <div class="jd-actions">${stateChip(st)}
                ${dl("config", j.id, "Export config", "", "btn ghost")}
                ${live ? `<button class="btn danger" onclick="stopJob('${j.id}')" ${j.stop_requested ? "disabled" : ""}>${j.stop_requested ? "Stopping…" : "Stop"}</button>`
                       : `<button class="btn ghost" onclick="forgetJob('${j.id}')">Forget</button>`}
            </div>
        </div>
        ${stageTrack(st)}
        <div class="now ${st.state}">${escapeHtml(line)}</div>
        ${st.warning ? `<div class="warnbox">${escapeHtml(st.warning)}</div>` : ""}
        <div class="bar big"><i style="width:${Math.round(jobProgress(st) * 100)}%"></i></div>
        <div class="two">
            <div class="panel clickable" onclick="bigChart()" title="Make it bigger">${lossChart(j.metrics || [])}
                <span class="zoom" aria-hidden="true"><svg viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5"/>
                <path d="M15.5 15.5 21 21M10.5 7.5v6M7.5 10.5h6"/></svg></span></div>
            <dl class="facts">${facts.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("")}</dl>
        </div>
        ${samples.length ? `<details class="says" data-k="says" ${op("says", true)}><summary class="h3row"><h3>What it says now</h3>
            <span class="muted small">${num(samples.length)} question${samples.length === 1 ? "" : "s"} it never saw while learning${sp.test_dataset ? `, from ${escapeHtml(sp.test_dataset)}` : ""}</span>
            <span class="grow"></span>
            ${dl("samples", j.id, "Download", "", "btn ghost small-btn")}</summary>
            <div class="samples">${samples.map((s, i) => sampleHtml(s, `s${i}`, isOpen)).join("")}</div></details>` : ""}
        ${madeHtml(j, op)}
        <div class="h3row logrow"><h3>The log</h3><span class="muted small">${num(logLines.length)} lines${live ? ", still writing" : ""}</span>
            <span class="grow"></span>${logLines.length ? dl("log", j.id, "Download the whole log", "", "btn ghost small-btn") : ""}</div>
        <details class="logbox" data-k="log" ${op("log", st.state === "failed")}><summary>${isOpen("log", st.state === "failed") ? "Hide it" : "Show it"}</summary>
            <pre>${logHtml(logLines)}</pre></details>`;

    // the person's click, not the re-render (a <details> parsed open fires
    // "toggle" too, which would throw away where they'd scrolled)
    const box = host.querySelector(".logbox");
    box.querySelector("summary").addEventListener("click", () => setTimeout(() => {
        box.querySelector("summary").textContent = box.open ? "Hide it" : "Show it";
        if (box.open) { const pre = box.querySelector("pre"); pre.scrollTop = pre.scrollHeight; }
    }, 0));
    const pre = box.querySelector("pre");
    if (!logAt || logAt.bottom) pre.scrollTop = pre.scrollHeight;
    else pre.scrollTop = logAt.top;

    if (L.modal === `chart:${j.id}`) $("modalBody").innerHTML = lossChart(j.metrics || [], true);
}

// A thinking model writes <think>...</think> before it answers: the answer
// leads, the thinking folds away.
function sampleHtml(s, k, isOpen = () => false) {
    const got = s.got || "";
    const m = got.match(/^\s*<think>([\s\S]*?)(<\/think>|$)([\s\S]*)$/);
    const think = m ? m[1].trim() : "", said = m ? m[3].trim() : got;
    const op = (kk) => (isOpen(kk, false) ? "open" : "");
    return `<div class="sample"><div class="s-ask"><span>asked</span>${escapeHtml(s.asked || "")}</div>
        <div class="s-got"><span>it said</span>${said ? escapeHtml(said) : `<em class="muted">(it was still thinking when it ran out of room)</em>`}</div>
        ${think ? `<details data-k="${k}-think" ${op(`${k}-think`)}><summary>how it thought</summary><div class="s-exp">${escapeHtml(think)}</div></details>` : ""}
        ${s.expected ? `<details data-k="${k}-exp" ${op(`${k}-exp`)}><summary>the answer you expected</summary><div class="s-exp">${escapeHtml(s.expected)}</div></details>` : ""}</div>`;
}

// What it made, the way a person thinks of it: the model to download first,
// then the two folders behind it, each folded with its files inside.
function madeHtml(j, op) {
    const all = j.artifacts || [];
    if (!all.length) return "";
    const link = a => dl("out", j.id, escapeHtml(a.path.split("/").pop()), a.path);
    const gguf = all.filter(a => a.path.endsWith(".gguf"));
    const group = (dir, title, say) => {
        const files = all.filter(a => a.path.startsWith(dir + "/"));
        if (!files.length) return "";
        const size = files.reduce((n, a) => n + a.bytes, 0);
        return `<details class="made-group" data-k="made-${dir}" ${op(`made-${dir}`)}><summary><b>${title}</b> <span class="muted">${human(size)} · ${say}</span></summary>
            <ul class="files">${files.map(a => `<li>${link(a)}<span>${human(a.bytes)}</span></li>`).join("")}</ul></details>`;
    };
    const rest = all.filter(a => !a.path.endsWith(".gguf") && !a.path.includes("/"));
    return `<h3>What it made</h3>
        ${gguf.map(a => `<div class="made-main"><div><div class="mm-t">Your model</div>${link(a)}
            <div class="muted small">One file. Load it in llama.cpp's llama-server, LM Studio, Ollama, or anything that reads GGUF.</div></div>
            <span class="mm-size">${human(a.bytes)}</span></div>`).join("")}
        ${group("merged", "The full model", "the base model with what it learned merged in; for more training or converting yourself")}
        ${group("adapter", "The adapter", "only what it learned, small; it goes on top of the base model")}
        ${rest.length ? `<ul class="files">${rest.map(a => `<li>${link(a)}<span>${human(a.bytes)}</span></li>`).join("")}</ul>` : ""}`;
}

async function stopJob(id) {
    if (!confirm("Stop this training? It finishes the step it's on and keeps nothing it hasn't saved.")) return;
    try { await call(`/api/jobs/${id}/stop`, { method: "POST" }); } catch (e) { err(e.message); }
    loadJob(id);
}

async function forgetJob(id) {
    if (!confirm("Forget this training? Its folder - and anything it made - is deleted.")) return;
    try { await call(`/api/jobs/${id}`, { method: "DELETE" }); } catch (e) { err(e.message); }
    L.sel = null; L.job = null;
    loadJobs();
}

// ── new training ────────────────────────────────────────────────────────────
function card(group, value, title, say, extra = "", disabled = false) {
    const on = L.pick[group] === value;
    return `<button class="card ${on ? "on" : ""}" ${disabled ? "disabled" : ""} onclick="pick('${group}', '${escapeHtml(value)}')">
        <div class="c-t">${title}</div><div class="c-s">${say}</div>${extra}</button>`;
}

function pick(group, value) {
    L.pick[group] = value;
    if (group === "model") $("modelOther").value = "";
    if (group === "effort") fillAdvanced();
    if (group === "quant") $("quantOther").value = value === L.overview.recommended_quant ? "" : value;
    renderNew();
}

function fillAdvanced(values = {}) {
    const lvl = L.overview.effort[L.pick.effort || "normal"];
    const dflt = { epochs: lvl.epochs, learning_rate: lvl.learning_rate, lora_rank: lvl.lora_rank,
                   max_length: 4096, batch_size: 2, grad_accum: 8, eval_fraction: 0.05 };
    const typed = {};
    document.querySelectorAll("[data-adv]").forEach(i => { if (i.value.trim()) typed[i.dataset.adv] = i.value.trim(); });
    $("advGrid").innerHTML = Object.entries(dflt).map(([k, v]) => `<label><span>${ADV[k][0]} <em>${ADV[k][1]}</em></span>
        <input data-adv="${k}" placeholder="${v}" value="${escapeHtml(values[k] ?? typed[k] ?? "")}" /></label>`).join("");
    renderExtras();
}

// Advanced: the extras switched on in Config. A switch for each this box
// has; the ones it is missing are listed, greyed, with how to get them - a
// switch that is simply absent reads as a feature that does not exist
// (24 Oct 2026). Ticks survive a redraw, and an imported config's are put in.
function renderExtras() {
    const ticked = new Set([...document.querySelectorAll("[data-x]:checked")].map(i => i.dataset.x));
    for (const id of L.pick.extrasIds || []) ticked.add(id);
    delete L.pick.extrasIds;
    const on = (L.overview.extras || []).filter(e => e.on);
    const have = on.filter(e => e.available !== false), lack = on.filter(e => e.available === false);
    $("advExtras").innerHTML = !on.length ? "" :
        (have.length ? `<div class="ex-h">Extras</div>` + have.map(e => `<label class="ex check">
            <input type="checkbox" data-x="${escapeHtml(e.id)}" ${ticked.has(e.id) ? "checked" : ""} />
            <span>${escapeHtml(e.name)} ${e.version ? `<em>${escapeHtml(e.version)}</em>` : ""}
            <div class="muted small">${escapeHtml(e.say || "")}</div></span></label>`).join("") : "") +
        (lack.length ? `<div class="ex-h">Not on this box</div>` + lack.map(e => `<div class="ex dim"><span>${escapeHtml(e.name)}</span>
            <div class="muted small">${escapeHtml(e.say || "")} To get it: ${escapeHtml(e.install || "install what it needs")}, into the training environment, then Check again on Config.</div></div>`).join("") : "");
}

async function loadNew() {
    L.datasets = (await call("/api/datasets")).datasets;
    if (!L.pick.effort) L.pick.effort = "normal";
    if (!L.pick.quant) L.pick.quant = L.overview.recommended_quant || "Q5_K_M";
    if (!L.pick.where) {
        const ready = L.overview.backends.find(b => b.ready);
        if (ready) L.pick.where = ready.name;
    }
    if (!$("advGrid").innerHTML) fillAdvanced();
    renderNew();
}

function renderNew() {
    const o = L.overview;
    if (!o) return;
    const learn = L.datasets.filter(d => d.use !== "questions"), qsets = L.datasets.filter(d => d.use === "questions");
    const ds = learn.find(d => d.id === L.pick.data);
    $("pickData").innerHTML = (learn.length ? learn.map(d => card("data", d.id, escapeHtml(d.name),
        `${num(d.rows)} examples · ${escapeHtml(d.shape_say || "")}` + (d.note ? `<br><span class="c-note">${escapeHtml(d.note)}</span>` : ""))).join("")
        : `<div class="empty">No datasets yet - add one below.</div>`) +
        (ds ? `<div class="peek-row"><button class="btn ghost small-btn" onclick="readDataset('${ds.id}')">Preview ${escapeHtml(ds.name)}</button></div>` : "");
    if (!learn.length) $("addDataBox").open = true;
    // the questions it is tested on: a question set (added as "questions to
    // test with"), or a few of its own examples held back at random
    const tsel = $("testSel"), was = L.pick.test || "";
    tsel.innerHTML = `<option value="">a few of its own examples, held back at random</option>` +
        qsets.map(d => `<option value="${escapeHtml(d.id)}">${escapeHtml(d.name)} (${num(d.rows)} question${d.rows === 1 ? "" : "s"})</option>`).join("") +
        (qsets.length ? "" : `<option value="" disabled>no question sets yet: add one as "questions to test with"</option>`);
    tsel.value = qsets.some(d => d.id === was) ? was : "";
    L.pick.test = tsel.value;
    $("fromHip").innerHTML = (o.hippocampus || {}).configured
        ? `<div class="from-hip"><span class="muted small">Or the hippocampus's lessons: every reviewed drafting call,
            the prompt the small model saw and what finally landed.</span>
            <button class="btn" id="hipBtn" onclick="fromHippocampus()">From the hippocampus</button></div>` : "";

    $("pickModel").innerHTML = o.models.map(m => card("model", m.id, `${escapeHtml(m.id.split("/").pop())} <span class="size">${m.size}</span>`,
        escapeHtml(m.say), (m.fits === false ? `<span class="warn">may not fit here (~${m.need_gb} GB)</span>` : m.fits ? `<span class="fits">fits here</span>` : "") +
        (m.needs_login ? `<span class="${o.hf_login.found ? "fits" : "warn"}">${o.hf_login.found ? "login found" : "needs a login"}</span>` : ""))).join("");
    const model = $("modelOther").value.trim() || L.pick.model;
    // rebuilt only when the question changes, so a half-typed token survives a click elsewhere
    const ask = model && needsLogin(model) && !o.hf_login.found ? model : "";
    if (ask !== L.loginAskFor) $("loginAsk").innerHTML = ask
        ? `<div class="warnbox login-ask"><b>${escapeHtml(model.split("/").pop())} needs a Hugging Face login</b> from an account that has
           accepted its licence on huggingface.co. This box doesn't have one yet.${loginForm("Save it")}</div>` : "";
    L.loginAskFor = ask;

    $("pickWhere").innerHTML = o.backends.map(b => card("where", b.name, escapeHtml(b.label), escapeHtml(b.say),
        `<span class="cost">${escapeHtml(b.cost || "")}</span>`, !b.ready)).join("");
    $("pickEffort").innerHTML = Object.entries(o.effort).map(([k, v]) => card("effort", k, k[0].toUpperCase() + k.slice(1), escapeHtml(v.say))).join("");

    // the recommended way out leads; the rest wait in a list
    const rec = o.recommended_quant || "Q5_K_M";
    $("quantMain").innerHTML = card("quant", rec, `${rec} <span class="chip rec">recommended</span>`, escapeHtml(o.quants[rec] || ""));
    const sel = $("quantOther");
    const others = Object.keys(o.quants).filter(k => k !== rec);
    if (sel.options.length !== others.length + 1) {
        sel.innerHTML = `<option value="">-</option>` + others.map(k => `<option value="${k}">${k === "none" ? "No GGUF" : k}: ${escapeHtml(o.quants[k])}</option>`).join("");
    }
    sel.value = L.pick.quant === rec ? "" : L.pick.quant;
    sel.classList.toggle("on", L.pick.quant !== rec);

    const where = o.backends.find(b => b.name === L.pick.where);
    const missing = [!ds && "what to learn", !model && "a model", !(where && where.ready) && "where to train"].filter(Boolean);
    $("goSummary").innerHTML = missing.length ? `Still needed: ${missing.join(", ")}.`
        : `Teach <b>${escapeHtml(model.split("/").pop())}</b> the <b>${num(ds.rows)}</b> examples in <b>${escapeHtml(ds.name)}</b>, on <b>${escapeHtml(where.label)}</b>, ${escapeHtml(L.pick.effort)} effort, out as <b>${escapeHtml(L.pick.quant === "none" ? "no GGUF" : L.pick.quant)}</b>.`;
    $("goBtn").disabled = missing.length > 0;
}

// The add form, in New training (p = "ds") and on Datasets (p = "im").
async function addDataset(p = "ds") {
    const f = $(p + "File").files[0], name = $(p + "Name").value.trim(), note = $(p + "Note").value.trim(), use = $(p + "Use").value;
    const need = [!name && "a name", !f && "a file", !note && "a description"].filter(Boolean);
    if (need.length) { $(p + "Found").innerHTML = `<div class="err">Give it ${need.join(", ")}.</div>`; return; }
    const fd = new FormData();
    fd.append("name", name); fd.append("note", note); fd.append("use", use); fd.append("file", f);
    $(p + "Found").innerHTML = `<div class="muted">Reading it…</div>`;
    let body;
    try { body = await call("/api/datasets", { method: "POST", body: fd }); }
    catch (e) { $(p + "Found").innerHTML = `<div class="err">Couldn't use it: ${escapeHtml(e.message)}</div>`; return; }
    const d = body.dataset;
    $(p + "Found").innerHTML = foundReport(d);
    if (use === "questions") L.pick.test = d.id; else L.pick.data = d.id;
    $(p + "Name").value = ""; $(p + "File").value = ""; $(p + "Note").value = "";
    await loadOverview();
    if (p === "im") { L.selDs = d.id; await loadDatasets(); } else await loadNew();
}

async function fromHippocampus() {
    const btn = $("hipBtn");
    btn.disabled = true;
    $("dsFound").innerHTML = `<div class="muted">Asking the hippocampus for its lessons…</div>`;
    try {
        const r = await call("/api/datasets/from-hippocampus", { method: "POST", body: JSON.stringify({ name: $("dsName").value.trim(), note: $("dsNote").value.trim() }) });
        $("dsFound").innerHTML = foundReport(r.dataset);
        L.pick.data = r.dataset.id;
        $("dsName").value = ""; $("dsNote").value = "";
        await loadOverview();
        await loadNew();
    } catch (e) {
        $("dsFound").innerHTML = `<div class="err">Couldn't get them: ${escapeHtml(e.message)}</div>`;
    } finally { if ($("hipBtn")) $("hipBtn").disabled = false; }
}

function foundReport(d) {
    return `<div class="found"><b>Found ${num(d.rows)} ${d.use === "questions" ? "questions" : "examples"}</b>: ${escapeHtml(d.shape_say || "")}.
        About ${num(d.tokens_est)} tokens in all (the pieces a model reads, about three quarters of a word each),
        ${num(d.chars_avg)} characters per example on average.
        ${d.unusable ? `<div class="warn">${num(d.unusable)} line${d.unusable === 1 ? "" : "s"} couldn't be used:
          <ul>${d.unusable_shown.map(b => `<li>line ${b.line}: ${escapeHtml(b.why)}</li>`).join("")}</ul></div>` : ""}</div>`;
}

function collectAdvanced() {
    const adv = { ...L.imported };
    document.querySelectorAll("[data-adv]").forEach(i => { if (i.value.trim()) adv[i.dataset.adv] = Number(i.value); });
    // an imported lora_alpha belongs to the rank it came with; a new rank gets its own
    const rank = adv.lora_rank ?? L.overview.effort[L.pick.effort].lora_rank;
    if (adv.lora_alpha != null && rank !== L.importedRank) delete adv.lora_alpha;
    adv.extras = [...document.querySelectorAll("[data-x]:checked")].map(i => i.dataset.x);
    return adv;
}

async function startTraining() {
    const body = { dataset: L.pick.data, base_model: $("modelOther").value.trim() || L.pick.model,
                   backend: L.pick.where, effort: L.pick.effort, quant: L.pick.quant,
                   name: $("jobName").value.trim(), notes: $("jobNotes").value.trim(), advanced: collectAdvanced(),
                   test_dataset: L.pick.test || "" };
    $("goBtn").disabled = true;
    try {
        const r = await call("/api/jobs", { method: "POST", body: JSON.stringify(body) });
        L.sel = r.id;
        $("jobName").value = ""; $("jobNotes").value = "";
        L.imported = {}; $("cfgNote").textContent = "";
        switchTab("trainings");
    } catch (e) {
        err(`Couldn't start it: ${e.message}`);
        $("goBtn").disabled = false;
    }
}

// A config exported from a training fills the walkthrough back in. What it
// can't fill (a dataset this box doesn't have) it says, and leaves to pick.
async function importConfig(file) {
    let c;
    try { c = JSON.parse(await file.text()); } catch (_) { $("cfgNote").textContent = "That file isn't a config (it isn't JSON)."; return; }
    if (!c || c.kind !== "seren-lyceum-config") { $("cfgNote").textContent = "That file isn't a Lyceum config."; return; }
    const o = L.overview, said = [];
    if (c.effort && o.effort[c.effort]) L.pick.effort = c.effort;
    if (c.quant && o.quants[c.quant]) L.pick.quant = c.quant;
    if (c.base_model) {
        if (o.models.some(m => m.id === c.base_model)) { L.pick.model = c.base_model; $("modelOther").value = ""; }
        else { delete L.pick.model; $("modelOther").value = c.base_model; }
    }
    if (c.test_dataset && L.datasets.some(d => d.id === c.test_dataset)) L.pick.test = c.test_dataset;
    else if (c.test_dataset) said.push(`its test dataset (${c.test_dataset}) isn't here; it will hold back a few at random`);
    if (c.dataset && L.datasets.some(d => d.id === c.dataset)) L.pick.data = c.dataset;
    else if (c.dataset) { delete L.pick.data; said.push(`its dataset (${c.dataset}) isn't here; pick one`); }
    const b = o.backends.find(x => x.name === c.backend);
    if (b && b.ready) L.pick.where = b.name;
    else if (c.backend) said.push(`${c.backend} can't train right now; pick where`);
    const adv = c.advanced || {};
    L.imported = {};
    L.importedRank = adv.lora_rank;
    for (const k of ["lora_alpha", "warmup_ratio", "seed"]) if (adv[k] != null) L.imported[k] = adv[k];
    const lvl = o.effort[L.pick.effort];
    const values = {};
    for (const k of Object.keys(ADV)) {
        const dflt = k in lvl ? lvl[k] : { max_length: 4096, batch_size: 2, grad_accum: 8, eval_fraction: 0.05 }[k];
        if (adv[k] != null && adv[k] !== dflt) values[k] = adv[k];
    }
    // an older config said attention / load_4bit; today they are extras
    L.pick.extrasIds = [...(adv.extras || []), ...(adv.attention === "flash_attention_2" ? ["flash_attention"] : []),
                        ...(adv.load_4bit ? ["load_4bit"] : [])];
    document.querySelectorAll("[data-x]").forEach(i => { i.checked = false; });
    fillAdvanced(values);
    if (Object.keys(values).length) $("advBox").open = true;
    $("jobName").value = c.name ? `${c.name} again` : "";
    $("jobNotes").value = c.notes || "";
    $("cfgNote").textContent = `Filled in from ${c.name || file.name}.` + (said.length ? ` But ${said.join("; ")}.` : "");
    renderNew();
}

// ── datasets ─────────────────────────────────────────────────────────────────
async function loadDatasets() {
    L.datasets = (await call("/api/datasets")).datasets;
    const el = $("dsList");
    el.innerHTML = L.datasets.length ? L.datasets.map(d => `<button class="job-row ${L.selDs === d.id ? "sel" : ""}" data-ds="${d.id}" onclick="showDataset('${d.id}')">
        <div class="jr-top"><span class="jr-name">${escapeHtml(d.name)}</span>${d.use === "questions" ? `<span class="chip q">questions</span>` : ""}</div>
        <div class="jr-sub">${num(d.rows)} ${d.use === "questions" ? "to test with" : "to learn from"} · ${ago(d.added_at)}</div></button>`).join("")
        : `<div class="empty">No datasets yet. Import one above.</div>`;
    if (L.selDs) showDataset(L.selDs);
}

function rowHtml(s, clip = 1200) {
    const cut = t => escapeHtml(clip ? String(t).slice(0, clip) + (String(t).length > clip ? "…" : "") : String(t));
    return `<div class="sample">${s.text != null ? `<div class="s-got">${cut(s.text)}</div>`
        : (s.messages || []).map(m => `<div class="msg ${m.role}"><span>${m.role}</span>${cut(m.content)}</div>`).join("")}</div>`;
}

async function showDataset(id) {
    L.selDs = id;
    let r;
    try { r = await call(`/api/datasets/${id}`); } catch (e) { err(e.message); return; }
    const d = r.dataset;
    $("dsDetail").innerHTML = `<div class="jd-head"><div><h2>${escapeHtml(d.name)}</h2>
        <div class="muted small">${escapeHtml(d.source || "")}</div></div>
        <div class="jd-actions"><button class="btn" onclick="readDataset('${d.id}')">Read it all</button>
            ${dl("dataset", d.id, "Download", "", "btn ghost")}
            <button class="btn ghost" onclick="dropDataset('${d.id}')">Delete</button></div></div>
        ${d.note ? `<p class="ds-note">${escapeHtml(d.note)}</p>` : `<p class="ds-note muted">No description.</p>`}
        ${foundReport(d)}
        <h3>A few examples</h3>
        ${r.sample.map(s => rowHtml(s)).join("")}`;
    document.querySelectorAll("#dsList .job-row").forEach(b => b.classList.toggle("sel", b.dataset.ds === id));
}

// The whole dataset, a page at a time, in the modal.
const PAGE = 25;
async function readDataset(id, offset = 0) {
    let r;
    try { r = await call(`/api/datasets/${id}/rows?offset=${offset}&limit=${PAGE}`); } catch (e) { err(e.message); return; }
    const d = L.datasets.find(x => x.id === id) || { name: id };
    const last = Math.min(r.total, offset + r.rows.length);
    const pager = `<div class="pager">
        <button class="btn ghost small-btn" ${offset <= 0 ? "disabled" : ""} onclick="readDataset('${id}', ${Math.max(0, offset - PAGE)})">← Earlier</button>
        <span class="muted small">${num(offset + 1)}-${num(last)} of ${num(r.total)}</span>
        <button class="btn ghost small-btn" ${last >= r.total ? "disabled" : ""} onclick="readDataset('${id}', ${offset + PAGE})">Later →</button></div>`;
    openModal(`${d.name}: every example`,
              `${pager}${r.rows.map((s, i) => `<div class="s-n muted small">example ${num(offset + i + 1)}</div>${rowHtml(s, 0)}`).join("")}${pager}`,
              dl("dataset", id, "Download", "", "btn ghost small-btn"), `dataset:${id}`);
    $("modalBody").scrollTop = 0;
}

async function dropDataset(id) {
    if (!confirm("Delete this dataset? Trainings that already used it keep their own copy.")) return;
    try { await call(`/api/datasets/${id}`, { method: "DELETE" }); } catch (e) { err(e.message); return; }
    L.selDs = null;
    $("dsDetail").innerHTML = `<div class="empty">Pick a dataset to see what's in it.</div>`;
    loadDatasets();
    loadOverview();
}

// ── tabs and the heartbeat ───────────────────────────────────────────────────
async function switchTab(tab) {
    showTab(tab);
    err("");
    try {
        if (!L.overview) await loadOverview();
        if (tab === "trainings") await loadJobs();
        if (tab === "new") await loadNew();
        if (tab === "datasets") await loadDatasets();
        if (tab === "box") { await loadOverview(); renderBox(); }
    } catch (e) { err(e.message); }
}

function heartbeat() {
    clearInterval(L.poll);
    L.poll = setInterval(async () => {
        if (!document.getElementById("trainings").classList.contains("active")) return;
        try {
            const was = Object.fromEntries(L.jobs.map(j => [j.id, `${j.status.state}/${j.status.step}`]));
            L.jobs = (await call("/api/jobs")).jobs;
            const now = Object.fromEntries(L.jobs.map(j => [j.id, `${j.status.state}/${j.status.step}`]));
            if (JSON.stringify(now) !== JSON.stringify(was)) renderJobList();
            const j = L.sel && L.jobs.find(x => x.id === L.sel);
            // a live one every beat; a finished one once more, the beat it changed
            if (j && (LIVE.includes(j.status.state) || now[j.id] !== was[j.id])) loadJob(L.sel);
            $("count-running").textContent = L.jobs.filter(j => LIVE.includes(j.status.state)).length || "";
        } catch (_) { /* the next beat tries again */ }
    }, 2000);
}

function reload() { L.overview = null; boot(); }

async function boot() {
    try {
        await loadOverview();
        const active = document.querySelector(".view.active");
        await switchTab(active ? active.id : "trainings");
    } catch (e) {
        err(`${e.message}. If this Lyceum has a token, set it with 🔑 Token.`);
    }
}

$("dsAdd").onclick = () => addDataset("ds");
$("imAdd").onclick = () => addDataset("im");
$("boxGrid").addEventListener("change", async (e) => {
    const x = e.target.closest("[data-extra]");
    if (!x) return;
    try { await call(`/api/extras/${encodeURIComponent(x.dataset.extra)}`, { method: "PUT", body: JSON.stringify({ on: x.checked }) }); await loadOverview(); }
    catch (er) { err(er.message); x.checked = !x.checked; }
});
$("goBtn").onclick = startTraining;
$("modelOther").addEventListener("input", () => { if ($("modelOther").value.trim()) delete L.pick.model; renderNew(); });
$("testSel").addEventListener("change", () => { L.pick.test = $("testSel").value; renderNew(); });
$("quantOther").addEventListener("change", () => { L.pick.quant = $("quantOther").value || L.overview.recommended_quant; renderNew(); });
$("cfgFile").addEventListener("change", () => { const f = $("cfgFile").files[0]; if (f) importConfig(f); $("cfgFile").value = ""; });
heartbeat();
boot();
