"use strict";

/* RareCare front-end: hash router, 3D tilt, questionnaire wizard, tabbed results.
   All dynamic text is set with textContent. No HTML injection anywhere. */
(() => {
  const $ = (id) => document.getElementById(id);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const TIER_LABELS = {
    URGENT: "See a doctor within 2 weeks",
    SOON: "Book a check-up soon",
    WATCH: "Keep an eye on it",
    NEED_MORE_INFO: "One quick question",
  };
  const MODE_LABELS = { neural: "trained model", "rule-baseline": "rule baseline", unavailable: "trained, not deployed (see Research)" };

  function el(tag, props = {}, children = []) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "text") n.textContent = v;
      else if (k === "class") n.className = v;
      else n.setAttribute(k, v);
    }
    children.forEach((c) => c != null && n.append(c));
    return n;
  }

  /* ------------------------------------------------------------ quick exit */
  const leave = () => window.location.replace("https://www.google.com/search?q=weather");
  $("quick-exit").addEventListener("click", leave);
  let lastEsc = 0;
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const now = Date.now();
    if (now - lastEsc < 600) leave();
    lastEsc = now;
  });

  /* ---------------------------------------------------------------- router */
  const ROUTES = ["home", "check", "how", "research", "privacy"];
  const menuBtn = $("menu-btn");
  const mobileMenu = $("mobile-menu");

  function setMenu(open) {
    menuBtn.setAttribute("aria-expanded", String(open));
    menuBtn.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    mobileMenu.hidden = !open;
  }
  menuBtn.addEventListener("click", () => setMenu(mobileMenu.hidden));

  function route() {
    const name = (location.hash.replace(/^#\/?/, "") || "home").split("?")[0];
    const page = ROUTES.includes(name) ? name : "home";
    $$(".page").forEach((p) => (p.hidden = p.dataset.page !== page));
    $$("[data-route]").forEach((a) => {
      if (a.dataset.route === page) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    setMenu(false);
    window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
    $("view").focus({ preventScroll: true });
    if (page === "check") ensureQuestionnaire();
  }
  window.addEventListener("hashchange", route);

  /* ----------------------------------------------------------- 3D effects */
  if (!reduceMotion && window.matchMedia("(pointer: fine)").matches) {
    const stage = $("scene")?.querySelector(".stage");
    if (stage) {
      window.addEventListener("pointermove", (e) => {
        const x = e.clientX / window.innerWidth - 0.5;
        const y = e.clientY / window.innerHeight - 0.5;
        stage.style.setProperty("--ry", `${-14 + x * 18}deg`);
        stage.style.setProperty("--rx", `${8 - y * 12}deg`);
      }, { passive: true });
    }
    document.addEventListener("pointermove", (e) => {
      const card = e.target.closest?.(".tilt");
      $$(".tilt[data-tilting]").forEach((c) => {
        if (c !== card) { c.style.removeProperty("--tx"); c.style.removeProperty("--ty"); c.removeAttribute("data-tilting"); }
      });
      if (!card) return;
      const r = card.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width - 0.5;
      const py = (e.clientY - r.top) / r.height - 0.5;
      card.style.setProperty("--ty", `${px * 8}deg`);
      card.style.setProperty("--tx", `${-py * 8}deg`);
      card.setAttribute("data-tilting", "");
    }, { passive: true });
  }

  /* ------------------------------------------------------------- API calls */
  async function call(url, options) {
    let resp;
    try {
      resp = await fetch(url, options);
    } catch {
      throw new Error("You seem to be offline. Please check your connection and try again.");
    }
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data?.error?.message || "Something went wrong. Please try again.");
    return data;
  }

  /* ---------------------------------------------------------------- wizard */
  const wizard = $("wizard");
  const STEPS = 4; // steps 0..3 collect input; 4 is the result
  let step = 0;
  let questionnaire = null;
  let loadingQuestionnaire = null;
  let sessionId = null;

  function ensureQuestionnaire() {
    if (questionnaire) return Promise.resolve();
    if (!loadingQuestionnaire) {
      loadingQuestionnaire = call("/api/v1/questionnaire").then((q) => {
        questionnaire = q;
        $("areas").replaceChildren(...q.areas.map((a) => el("label", { class: "choice", "data-area": a.id }, [
          el("input", { type: "checkbox", name: "area", value: a.id }),
          el("span", { class: "face" }, [el("strong", { text: a.title }), el("small", { text: a.subtitle })]),
          el("span", { class: "check", "aria-hidden": "true" }),
        ])));
        q.age_bands.forEach((b) => $("age_band").append(
          el("option", { value: b, text: b.startsWith("<") ? `Under ${b.slice(1)}` : b.endsWith("+") ? `${b.slice(0, -1)} or over` : b })));
        q.durations.forEach((d) => $("duration").append(el("option", { value: d.value, text: d.label })));
      }).catch((err) => { loadingQuestionnaire = null; showError(err.message); });
    }
    return loadingQuestionnaire;
  }

  const selectedAreas = () => $$('input[name="area"]:checked', wizard).map((i) => i.value);
  const selectedSymptoms = () => $$('input[name="symptoms"]:checked', wizard).map((i) => i.value);

  function buildSymptomStep() {
    const keep = new Set(selectedSymptoms());
    const areas = selectedAreas();
    const groups = questionnaire.areas.filter((a) => areas.includes(a.id));
    $("symptom-groups").replaceChildren(...groups.map((g) => el("div", { class: "sym-group" }, [
      el("h3", { text: g.title }),
      el("div", { class: "sym-list" }, g.symptoms.map((s) => {
        const input = el("input", { type: "checkbox", name: "symptoms", value: s.concept });
        if (keep.has(s.concept)) input.checked = true;
        return el("label", { class: "sym" }, [input, el("span", { text: s.label })]);
      })),
    ])));
  }

  function showError(msg) {
    const box = $("form-error");
    box.textContent = msg || "";
    box.hidden = !msg;
  }

  function goTo(n, focus = true) {
    step = n;
    $$(".step", wizard).forEach((s) => (s.hidden = Number(s.dataset.step) !== n));
    $$("#stepper li").forEach((li) => {
      const i = Number(li.dataset.step);
      li.classList.toggle("done", i < n);
      if (i === n) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
    });
    $("progress-bar").style.width = `${(n / STEPS) * 100}%`;
    $("back").style.visibility = n === 0 ? "hidden" : "visible";
    $("next").textContent = n === STEPS - 1 ? "See my result" : "Continue";
    showError("");
    if (focus) {
      const legend = wizard.querySelector(`.step[data-step="${n}"] .step-title`);
      legend?.setAttribute("tabindex", "-1");
      legend?.focus({ preventScroll: true });
    }
  }

  $("next").addEventListener("click", async () => {
    if (step === 0) {
      await ensureQuestionnaire();
      if (!questionnaire) return;
      if (selectedAreas().length === 0) {
        showError("Select at least one area. If unsure, choose the closest match; you can add details later.");
        return;
      }
      buildSymptomStep();
    }
    if (step < STEPS - 1) { goTo(step + 1); return; }
    submit();
  });
  $("back").addEventListener("click", () => { if (step > 0) goTo(step - 1); });

  const text = $("text");
  text.addEventListener("input", () => { $("text-count").textContent = `${text.value.length} / 2000`; });

  const drop = $("drop");
  const file = $("image");
  const DROP_DEFAULT = $("drop-text").textContent;
  file.addEventListener("change", () => { $("drop-text").textContent = file.files[0]?.name || DROP_DEFAULT; });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("drag")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) { file.files = e.dataTransfer.files; file.dispatchEvent(new Event("change")); }
  });

  async function submit() {
    const symptoms = selectedSymptoms();
    if (!symptoms.length && !text.value.trim()) {
      showError("Select at least one symptom or add a description.");
      return;
    }
    const form = new FormData();
    form.append("text", text.value);
    symptoms.forEach((s) => form.append("symptoms", s));
    if ($("age_band").value) form.append("age_band", $("age_band").value);
    if ($("duration").value) form.append("duration", $("duration").value);
    if (file.files[0]) form.append("image", file.files[0]);

    $("next").disabled = true;
    showResultLoading();
    try {
      const data = await call("/api/v1/assess", { method: "POST", body: form });
      sessionId = data.session_id;
      wizard.hidden = true;
      $("progress-bar").style.width = "100%";
      $$("#stepper li").forEach((li) => { li.classList.add("done"); li.removeAttribute("aria-current"); });
      $$("#stepper li")[STEPS].setAttribute("aria-current", "step");
      render(data.result);
    } catch (err) {
      $("result").hidden = true;
      showError(err.message);
    } finally {
      $("next").disabled = false;
    }
  }

  async function answer(variable, value) {
    if (!sessionId) return;
    $$("#q-options button").forEach((b) => (b.disabled = true));
    try {
      const data = await call(`/api/v1/session/${encodeURIComponent(sessionId)}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ variable, value }),
      });
      sessionId = data.session_id;
      render(data.result);
    } catch (err) {
      $("q-hint").textContent = err.message;
      $$("#q-options button").forEach((b) => (b.disabled = false));
    }
  }

  function showResultLoading() {
    $("result").hidden = false;
    $("result-loading").hidden = false;
    ["verdict", "rtabs", "build", "result-actions"].forEach((id) => ($(id).hidden = true));
  }

  /* --------------------------------------------------------------- results */
  function render(r) {
    $("result-loading").hidden = true;
    const verdict = $("verdict");
    verdict.dataset.tier = r.tier;
    $("tier-label").textContent = TIER_LABELS[r.tier] || r.tier;
    $("headline").textContent = r.headline;

    const q = r.question;
    $("followup").hidden = !q;
    if (q) {
      $("q-text").textContent = q.text;
      $("q-hint").textContent = "Only questions that change the answer are asked. You can skip.";
      $("q-options").replaceChildren(...q.options.map((o) => {
        const b = el("button", { type: "button", text: o.label });
        if (o.value === "__decline__") b.className = "decline";
        b.addEventListener("click", () => answer(q.variable, o.value));
        return b;
      }));
    }
    verdict.hidden = false;
    verdict.style.animation = "none"; void verdict.offsetWidth; verdict.style.animation = "";
    verdict.focus({ preventScroll: true });
    verdict.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });

    const final = !q;
    $("rtabs").hidden = !final;
    $("build").hidden = !final;
    $("result-actions").hidden = !final;
    if (!final) return;

    $("cancer-cards").replaceChildren(...r.cancers.map((c) => el("div", { class: "cancer-card" }, [
      el("h4", { text: c.label }),
      el("span", { class: `tier-chip ${c.tier}`, text: TIER_LABELS[c.tier] }),
    ])));
    if (!r.cancers.length) {
      $("cancer-cards").replaceChildren(el("p", { class: "hint", text: "No specific cancer area was flagged by what you shared." }));
    }
    $("safety-list").replaceChildren(...r.safety_net.map((s) => el("li", { text: s })));

    $("chain").replaceChildren(...r.evidence.map((ev) => el("li", {}, [
      el("div", { class: "flowline" }, [
        el("span", { class: "quote", text: ev.from_questionnaire ? "You ticked this" : `“${ev.span}”` }),
        el("span", { class: "arrow", text: "→" }),
        el("span", { class: "concept", text: ev.concept_label }),
        el("span", { class: "arrow", text: "→" }),
        el("span", { class: "crit", text: ev.criterion_id }),
      ]),
      el("p", {
        class: "why",
        text: `${ev.action} Source: ${ev.source}.` +
          (ev.probability < 1 ? ` Applies with ${Math.round(ev.probability * 100)}% likelihood, given details not shared.` : ""),
      }),
    ])));
    $("chain-empty").hidden = r.evidence.length > 0;

    $("source-list").replaceChildren(...r.citations.map((c) => el("li", {}, [
      el("a", { href: c.url, target: "_blank", rel: "noopener noreferrer", text: c.title }),
      el("p", { text: c.snippet }),
    ])));
    if (!r.citations.length) {
      $("source-list").replaceChildren(el("li", {}, [el("p", { text: "No guideline passages were needed for this result." })]));
    }

    const hasImage = !!r.image;
    $("tab-image").hidden = !hasImage;
    if (hasImage) {
      $("image-msg").textContent = r.image.message;
      const img = $("heatmap");
      if (r.image.heatmap_png_b64) { img.src = `data:image/png;base64,${r.image.heatmap_png_b64}`; img.hidden = false; }
      else img.hidden = true;
    }
    $("modes").replaceChildren(...Object.entries(r.component_modes).map(([k, v]) =>
      el("li", { text: `${k.replaceAll("_", " ")}: ${MODE_LABELS[v] || v}` })));
    selectTab("tab-summary");
  }

  /* WAI-ARIA tabs: arrow keys, Home/End, roving tabindex */
  const tabs = $$('[role="tab"]');
  function selectTab(id) {
    tabs.forEach((t) => {
      const on = t.id === id;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      $(t.getAttribute("aria-controls")).hidden = !on;
    });
  }
  tabs.forEach((t) => {
    t.addEventListener("click", () => selectTab(t.id));
    t.addEventListener("keydown", (e) => {
      const visible = tabs.filter((x) => !x.hidden);
      const i = visible.indexOf(t);
      const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: visible.length - 1 }[e.key];
      if (next === undefined) return;
      e.preventDefault();
      const target = visible[(next + visible.length) % visible.length];
      selectTab(target.id);
      target.focus();
    });
  });

  $("restart").addEventListener("click", () => {
    if (sessionId) fetch(`/api/v1/session/${encodeURIComponent(sessionId)}`, { method: "DELETE" }).catch(() => {});
    sessionId = null;
    wizard.reset();
    wizard.hidden = false;
    $("text-count").textContent = "0 / 2000";
    $("drop-text").textContent = DROP_DEFAULT;
    $("result").hidden = true;
    buildSymptomStep();
    goTo(0);
  });

  goTo(0, false);
  route();
})();
