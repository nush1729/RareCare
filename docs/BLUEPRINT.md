# RareCare — Project Blueprint

**Stigma-robust, guideline-grounded triage for women's cancers — with guaranteed sensitivity, calibrated uncertainty, and minimal-disclosure questioning.**

> Research prototype. Not a medical device, not a diagnostic tool, not clinically validated.

---

## 1. Concept

RareCare is a screening-triage system for three women's cancers where **stigma and symptom normalisation — not model accuracy — drive late diagnosis**:

| Cancer | Why it presents late | System role |
|---|---|---|
| **Breast** | Body taboo, marriage/social fear, "it's just a lump" | Image + text fusion (ultrasound available in clinics) |
| **Cervical** | Post-coital bleeding and abnormal discharge are frequently under-reported | Lay-language-robust understanding; disclosure-cost-aware questioning |
| **Ovarian** | Vague symptoms (bloating, early satiety, pelvic pain) dismissed as "normal women's problems" | Adaptive questioning under high uncertainty |

Post-menopausal bleeding (NICE NG12 endometrial criterion) is handled as a rule inside the gynae pathway — no separate model.

### Inputs
- **Text (required):** free-text symptom description in clinical, lay, euphemistic, or code-mixed (Hinglish, stretch) language.
- **Image (optional):** breast ultrasound; cervical Pap cytology; ovarian ultrasound (clinic-side).
- **Minimal demographics:** age band, menopausal status.

### Outputs — referral-urgency tier, aligned to NICE NG12 (3% PPV threshold)

| Tier | Meaning |
|---|---|
| **Urgent** | Matches a suspected-cancer referral criterion — see a doctor within 2 weeks |
| **Soon** | Needs routine clinical review / screening |
| **Watch** | Low risk now; explicit red-flag safety-net advice |
| **Need more info** | Uncertainty too high → agent asks the next best question |
| **Retake photo** | Image quality too low; specific retake guidance |
| **Unsupported image** | Recognised as a medical image type RareCare does not assess (e.g. chest X-ray) |
| **Not a medical image** | Selfie, document, etc. |

Image rejections **never end the assessment** — the text path always completes.

Every decision carries an **evidence chain**: user phrase → clinical concept → NG12 criterion → tier, plus image heatmap and cited guideline passages.

### Users
1. **Women doing an anonymous self-check**, often on a **shared family phone** → discreet UI, quick-exit button, neutral tab title, no history, no stored raw input.
2. **ASHA / community health workers** (all women) doing home visits.

---

## 2. Problem and why it matters

- Breast and cervical cancer are the two most common cancers in women across India and much of the Global South; ovarian cancer is typically diagnosed late because its symptoms are normalised.
- The failure point is **disclosure**: symptoms are described indirectly or not at all, and existing tools don't understand indirect language, don't bound their misses, and ask intimate questions up front.
- All three have meaningful early-detection windows; a tool that lowers disclosure cost and never silently under-triages targets the actual bottleneck.

---

## 3. USP

### What existing systems do
- **Symptom checkers** (Ada, Buoy, WebMD-style): structured clinical symptoms → ranked differential. Triage correct ~57% in Semigran et al., BMJ 2015.
- **Imaging AI** (MedMNIST benchmarks, mammography CAD, NCI Automated Visual Evaluation for cervix): single-modality, AUROC on curated in-distribution test sets.
- **Medical LLMs** (Med-PaLM M, AMIE-style agents): fluent, but uncalibrated, no formal guarantees, hard to audit.

### Their limitations
1. Assume **clinical vocabulary** — biomedical encoders degrade on lay/euphemistic text.
2. **Uncalibrated point estimates** — no bound on miss rate, the costly error in screening.
3. Require **all modalities** or fuse naively; no principled missing-modality handling.
4. **Never abstain** — OOD images and vague text still get confident answers.
5. Ignore **disclosure cost** — intimate questions asked regardless of need.

### What RareCare does differently — three defensible claims
1. **Stigma-robust language understanding.** Lay/euphemism → clinical-concept alignment, measured on a new benchmark (**StigmaSymp-W**) that quantifies the *euphemism gap* in women's-health language.
2. **Guaranteed sensitivity under missing modalities.** Class-conditional conformal prediction bounds the Urgent miss rate at α per subgroup; an OR-rule across modalities preserves each modality's guarantee **without paired text+image calibration data**. Evidential (subjective-logic) fusion ranks, detects modality conflict, and explains inside that safety envelope.
3. **Disclosure-minimising questioning.** Next question chosen by *expected information gain ÷ sensitivity cost* over a guideline knowledge graph — confident decisions with the fewest intimate questions.

### "Why would anyone care when healthcare AI already exists?"
Because the existing systems optimise the wrong thing for this population. They are accurate on clean clinical input and silent about their failures. RareCare is built for how women actually describe gynaecological and breast symptoms, gives a *provable* bound on missed urgent cases, refuses rather than guesses on inputs it can't assess, and asks the least-invasive questions needed. Each of these is measurable and each is absent from existing tools.

### Publishable contributions
| | Contribution | Anchored on | Venue fit |
|---|---|---|---|
| **A** | StigmaSymp-W benchmark + euphemism-gap analysis of biomedical encoders | Cervical, ovarian | BioNLP / ClinicalNLP workshops |
| **B** | Conformal evidential fusion with missing modalities and subgroup guarantees | Breast (paired data) | ML4H, CHIL, MICCAI UNSURE |
| **C** | Disclosure-cost-aware adaptive questioning | Cervical (high cost), ovarian (high uncertainty) | ML4H, CHI health |

---

## 4. Technical components (each earns its place)

| Component | Why |
|---|---|
| **Contrastive lay→clinical alignment** (SapBERT-style) | Fixes the dominant failure mode: euphemistic input |
| **Knowledge graph** (NetworkX): symptom ↔ SNOMED/CHV concept ↔ NG12 criterion ↔ cancer ↔ risk factor, with per-question disclosure cost | Single structure for criterion mapping, question generation, explanation. ~15–20 criteria, hand-curated |
| **Evidential heads** (Dirichlet) + subjective-logic fusion (Trusted Multi-view Classification, Han et al. 2021) | Missing modality = vacuous opinion → graceful degradation; measurable conflict |
| **Temperature scaling + class-conditional / Mondrian conformal** | Calibrated risk + per-subgroup miss-rate guarantee |
| **Prevalence-aware recalibration** (label-shift prior correction from GLOBOCAN) | NG12 is UK-calibrated; India needs prior correction |
| **Image gatekeeping stack** (see §4.1) | Refuse what can't be assessed; stay robust on what can |
| **Adaptive questioning agent** — deterministic Bayesian planner + LLM phrasing | Planner decides; LLM only phrases gently |
| **RAG explanations** — MedCPT retriever over NG12, WHO guidance, NCI PDQ | Cited, grounded, faithfulness-measured |
| **XAI** — Grad-CAM++ (images), Integrated Gradients + extracted spans (text) | Quantitatively evaluated: pointing game / IoU on BUSI masks; ERASER comprehensiveness/sufficiency |
| **Privacy by design** — Presidio PII scrub, stateless sessions, no raw-input logs, discreet UI, optional in-browser text inference (transformers.js) | Shared-phone reality of the target users |

**Deliberately excluded:** federated learning (no partner sites), end-to-end LLM diagnosis (unauditable), unsupervised online self-training (label-noise, poisoning, silent drift).

### 4.1 Handling unknown images

Unknown images are used in training so the model **recognises** them — not so it produces a diagnosis for them. Four cases, four behaviours:

| Case | Example | Behaviour | Training |
|---|---|---|---|
| Poor quality | Blurry / dark / glare | "Retake photo" + specific guidance | Quality classifier on synthetic blur, low light, glare |
| Near-OOD, in scope | Breast US from a new machine | **Give a result** with widened conformal set — must *not* be rejected | Heavy augmentation, MedMNIST-C corruptions, cross-site training, test-time BN adaptation |
| Unsupported medical | Chest X-ray, MRI, retina | "Unsupported image type", continue with text | Explicit router class trained on other MedMNIST sets (Chest, OCT, Retina, Organ, Blood) |
| Not medical | Selfie, cat, document | "Not a medical image" | Outlier exposure (ImageNet/COCO) + energy-score threshold as backstop |

**Learning from unknowns over time (safely):** consented rejected / high-uncertainty images → review queue → expert labels prioritised by active learning → periodic retrain → evaluation gate → new versioned model on HF Hub. The prototype *simulates* this loop by holding out a site and measuring recovery vs. number of labels.

---

## 5. Datasets

### Text
| Dataset | Role |
|---|---|
| **MACCROBAT** (200 annotated case reports) | Symptom/duration/body-site extractor pretraining |
| **NICE NG12** (breast, gynae criteria — manually encoded) | Criteria + urgency ground truth |
| **Consumer Health Vocabulary + SNOMED CT subset** | Lay → clinical concept lexicon |
| **StigmaSymp-W** (ours, ~3k items) | Vignettes per NG12 criterion across clinical / lay / euphemistic / noisy / Hinglish registers |
| **HealthCareMagic-100k / MedDialog** (women's-health subset) | Unlabelled real patient language for domain-adaptive pretraining |
| **Symptom2Disease, DDXPlus** | Out-of-scope negatives; symptom-checker baselines |

**StigmaSymp-W leakage control:** training paraphrases from one open-LLM family; **test set human-written, no LLM** (5–10 volunteer writers, labels fixed before writing); target a gynaecologist/oncologist review of a sample.

### Vision
| Role | Train | External test |
|---|---|---|
| Breast ultrasound | **BUSI** (780 imgs, with masks) | **BUS-BRA** (~1.9k imgs, Brazil) |
| Breast fusion testbed (real pairing) | **BrEaST-Lesions-USG** (TCIA 2024: US + age, signs/symptoms, BI-RADS, histology) | — |
| Cervical cytology | **SIPaKMeD** (~4k cells) | **Herlev** (917 cells) |
| Ovarian ultrasound (clinic-side, optional) | **MMOTU** (~1.5k 2D imgs) | held-out split |
| Backbone benchmark / robustness | BreastMNIST (MedMNIST+ 224px), **MedMNIST-C** | — |
| Router "unsupported medical" class | ChestMNIST, OCTMNIST, RetinaMNIST, OrganAMNIST, BloodMNIST | — |
| Outlier exposure | ImageNet / COCO subset | — |

**Fallbacks:** PAD-UFES-20 if BrEaST metadata is insufficient; IARC colposcopy/VIA image bank is a stretch goal pending access.

---

## 6. Models

| Component | Choice | Ablations / baselines |
|---|---|---|
| Text encoder | **BioBERT v1.2** (`dmis-lab/biobert-base-cased-v1.2`) — heads: token classification (symptom/duration/site/severity) + assertion/negation; multi-label NG12-criterion (evidential); scope classifier | PubMedBERT, Bio_ClinicalBERT, SapBERT |
| Concept normaliser | **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`), contrastively fine-tuned on (euphemism, concept) pairs | Lexicon-only |
| Vision backbone | **ConvNeXt-Tiny** (shared), per-organ evidential heads + router/OOD/quality heads; 3-member ensemble | ResNet-18/50 (MedMNIST standard), BiomedCLIP linear probe |
| Serving vision | **EfficientNet-B0** distilled student → ONNX int8 | — |
| Uncertainty | Deep ensemble + evidential | MC-dropout |
| Fusion | Subjective-logic combination + conformal OR-rule | Prob. averaging, early concat (paired data only) |
| Retriever | **MedCPT** + FAISS | BM25 |
| Phrasing LLM | Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct, 4-bit via MLX (dev); templates by default on Space | — |

---

## 7. End-to-end architecture

```
             ┌──────────── TEXT PATH ─────────────┐     ┌──────────── IMAGE PATH (optional) ─────────────┐
User text →  PII scrub (Presidio) → normalise        Image → quality head ──(low)──► "Retake photo"
             → BioBERT extractor                           → router: breast-US / cytology / ovarian-US /
               (spans + negation + duration)                 unsupported-medical / non-medical
             → SapBERT + alignment → concept IDs             + energy OOD score ──(reject)──► tier + continue text
             → KG: concepts → NG12 criteria                → ConvNeXt-T organ head (ensemble, evidential)
             → BioBERT criterion classifier (evidential)     + test-time BN adaptation
             → prevalence-adjusted risk + conformal set    → Grad-CAM++ │ calibrated p │ conformal set
             └──────────────────┬───────────────────┘     └──────────────────────┬─────────────────────────┘
                                ▼                                                ▼
                  ┌────────────── FUSION & SAFETY LAYER ──────────────────────────────┐
                  │ Subjective-logic fusion (missing modality = vacuous opinion)      │
                  │ Conflict score → escalate on disagreement                         │
                  │ Conformal OR-rule → Urgent recall ≥ 1−α (per subgroup)            │
                  │ Tier: Urgent / Soon / Watch / Need-more-info                      │
                  └───────────────┬───────────────────────────────────────────────────┘
          uncertain? ──yes──► Question planner: argmax EIG(q) / disclosure_cost(q) over KG
                  │                → LLM/template phrasing → answer → update → loop
                  ▼ no
       Explanation: evidence chain + heatmap + MedCPT-cited guideline passages + safety-net advice
```

---

## 8. Improvement over existing approaches

| Limitation | RareCare mechanism | Evidence |
|---|---|---|
| Fails on euphemistic text | Contrastive alignment + CHV lexicon | Euphemism gap before/after on StigmaSymp-W |
| Unbounded misses | Temperature scaling + class-conditional conformal | Empirical Urgent recall ≥ 1−α, per subgroup |
| Needs paired data | Evidential fusion + conformal OR-rule | Works text-only / image-only / both; validated vs true pairing on BrEaST |
| Always answers | Quality + router + OOD gate, Need-more-info tier | OOD AUROC; risk–coverage; low false-rejection on shifted in-scope images |
| Intimate questions up front | EIG ÷ cost planner | Questions & cumulative disclosure cost to decision vs. full questionnaire |
| Black-box LLM | Deterministic KG decision path | Traceable chains; citation precision |
| UK thresholds everywhere | Prevalence-aware recalibration | Decision-curve net benefit under UK vs India priors |

---

## 9. Evaluation

### Metrics
- **Extraction:** strict/relaxed span F1; negation accuracy. **Normalisation:** acc@1/@5.
- **Criteria:** micro/macro F1. **Triage:** macro-F1, **under-triage rate (primary, pre-registered at α = 0.05)**, over-triage rate.
- **Images:** AUROC, AUPRC, specificity @ 90/95% sensitivity.
- **Calibration:** ECE, Brier, reliability diagrams. **Conformal:** coverage, set size, per-subgroup coverage.
- **Selective prediction:** AURC. **OOD:** AUROC, FPR@95TPR, **false-rejection rate on near-OOD in-scope images**.
- **Clinical utility:** decision-curve analysis (net benefit).
- **XAI:** pointing game / IoU on BUSI masks; ERASER comprehensiveness/sufficiency. **RAG:** citation precision, NLI faithfulness.
- **Agent:** questions-to-decision, cumulative disclosure cost, under-triage.
- **Active-learning sim:** labels needed to recover performance on a held-out site.

### Baselines
1. NG12 keyword checklist
2. Vanilla BioBERT (no alignment)
3. Symptom2Disease-style classifier
4. ResNet-18 + published MedMNIST v2 numbers
5. Probability averaging / early concat fusion
6. **Zero-shot open LLM triage** (calibration + under-triage comparison)

### Ablations
Remove: alignment · KG · conformal (plain threshold) · evidential fusion (averaging) · OOD gate · cost term in planner · test-time adaptation. Swap: text encoders, vision backbones, 28 vs 224 px.

### Robustness & fairness
- **Text:** clinical → lay → euphemistic → typos → Hinglish; negation flips; paraphrase.
- **Image:** MedMNIST-C corruptions; BUSI → BUS-BRA; SIPaKMeD → Herlev.
- **Subgroups:** age band, menopausal status, acquisition site. Report sensitivity gaps and Mondrian coverage.

### What makes results convincing
External test sets · human-written text test set · 5 seeds with bootstrap CIs · clinically chosen operating points · pre-registered primary metric · honest failure analysis.

---

## 10. Deployment

- **Flask + gunicorn** in a **Hugging Face Space (Docker SDK, port 7860)**.
  - `POST /api/v1/assess` · `POST /api/v1/session/{id}/answer` · `GET /health` · `GET /model-card`
  - Server-rendered discreet UI: neutral branding, quick-exit, no persistent cookies.
- **Hugging Face Hub:** versioned model repos (ONNX int8 BioBERT, EfficientNet-B0 student, conformal thresholds, KG JSON) with model cards; StigmaSymp-W dataset with datasheet.
- **Inference:** ONNX Runtime CPU, dynamic int8; target < 1.5 s per assessment on free CPU tier.
- **Also required:** bundled FAISS index; Presidio; stateless TTL sessions; anonymised decision-metadata audit log only; rate limiting; strict CSP.
- **CI:** GitHub Actions — unit tests + **evaluation regression gate** (fail on worse under-triage or calibration).
- **Experiment tooling:** Weights & Biases, Hydra.
- **Stretch:** in-browser BioBERT (transformers.js) privacy mode.

---

## 11. Roadmap (~14 weeks)

| Phase | Weeks | Deliverables | Gate |
|---|---|---|---|
| **0. Setup** | 1 | Repo, envs (MPS local / Colab train), Hydra, W&B; **verify BrEaST metadata fields and MMOTU access** | Core datasets downloadable; fallback decided |
| **1. Data + KG** | 2–3 | Loaders, splits; NG12 breast/gynae criteria in KG with disclosure costs; CHV/SNOMED lexicon; StigmaSymp-W generation; human test-set collection starts | ~15–20 criteria encoded; test protocol frozen |
| **2. Text** | 3–5 | BioBERT extractor (MACCROBAT → StigmaSymp-W); SapBERT alignment; evidential criterion classifier; euphemism-gap results | Alignment measurably closes the gap |
| **3. Vision** | 4–6 (parallel) | ConvNeXt-T multi-organ + ensemble; quality head; router with unsupported-medical class; outlier exposure; energy OOD; Grad-CAM++; external tests | External AUROC reported; OOD works; low false-rejection on shifted in-scope |
| **4. Uncertainty + fusion** | 7–8 | Temp scaling, Mondrian conformal, evidential fusion, OR-rule, BrEaST paired validation, prevalence recalibration, test-time BN adaptation | Coverage guarantee holds empirically |
| **5. Agent + RAG** | 9–10 | EIG ÷ cost planner; simulated-patient eval; MedCPT RAG; explanation composer | Fewer questions / lower disclosure cost at equal under-triage |
| **6. Evaluation** | 11–12 | Baselines, ablations, robustness, fairness, decision curves, LLM baseline, active-learning sim | Results tables frozen |
| **7. Deployment** | 13 | ONNX export + quantisation, Flask API + UI, Docker Space, cards, CI gate | Space live; latency met |
| **8. Paper + demo** | 14 | Workshop paper draft, 3-min demo video | Submission-ready |

### Demo personas
1. **Cervical:** post-coital bleeding with abnormal vaginal discharge → Soon (clinical examination), with evidence chain; planner asked only low-cost questions.
2. **Breast:** ASHA worker uploads ultrasound + "lump in left side, 3 weeks" → fused decision, heatmap, conformal set.
3. **Ovarian:** "stomach always feels full and bloated lately" → Need-more-info → two low-sensitivity questions (duration, age) → Soon/Urgent per NG12.
4. **Gatekeeping:** uploads a chest X-ray → "Unsupported image type", text assessment still completes.

---

## 12. Summary

| | |
|---|---|
| **Scope** | Breast, cervical, ovarian (+ post-menopausal bleeding rule) |
| **Tech stack** | PyTorch, HF Transformers, timm, Optimum/ONNX Runtime, FAISS, NetworkX, Presidio, MAPIE/custom conformal, Captum, pytorch-grad-cam, Hydra, W&B, Flask, gunicorn, Docker, HF Hub/Spaces, GitHub Actions, MLX |
| **Models** | BioBERT v1.2, SapBERT (aligned), MedCPT, ConvNeXt-Tiny ensemble → EfficientNet-B0 student, optional 7B instruct LLM |
| **Datasets** | MACCROBAT, NG12, CHV/SNOMED, StigmaSymp-W (ours), HealthCareMagic; BUSI, BUS-BRA, BrEaST-Lesions-USG, SIPaKMeD, Herlev, MMOTU, MedMNIST+/C |
| **USP** | Understands how women actually describe stigmatised symptoms; provable bound on missed urgent cases even with unpaired / missing modalities; refuses rather than guesses on unassessable images; asks the least-intimate questions necessary |
| **Expected outputs** | Live HF Space; model + data cards; StigmaSymp-W on HF Hub; results report (external validation, fairness audit, decision curves); workshop paper; demo video |

### Limitations
- Not clinically validated; not a medical device.
- Training text partly synthetic; human test set is small.
- Imaging datasets small and few-site.
- Paired text+image data exists only for breast (BrEaST); cervical/ovarian fusion relies on the unpaired OR-rule.
- NG12 is UK-specific; prevalence recalibration is a partial fix.
- Conformal guarantee is marginal and assumes calibration/deployment exchangeability.
- Cervical cytology is lab-side, not what a patient can capture; VIA/colposcopy depends on IARC access.

### Future scope
- Prospective pilot with ASHA workers.
- Voice input in Indian languages (IndicWhisper).
- VIA / colposcopy module; HPV self-sampling pathway integration.
- Federated fine-tuning across clinics.
- Longitudinal symptom tracking for persistent ovarian symptoms.
