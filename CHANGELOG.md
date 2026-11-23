## Barist.8.1 — adversarial review of Barist.8 (released; git tag `barist-8.1`, HF `KucLab/Barist.8.1`, GH `Jerry256254/barist.8.1`)

- **Leak purge**: 189 eval sentences (heldout2/3, cs_benign, regex_benign, user/identity
  sets) removed from train.jsonl; model retrained from scratch on the clean 271,056 rows
  (checkpoint `checkpoints_b81a`, 2 epochs, per-category pos_weight in the sigmoid BCE).
  New gate `train_eval_leak` (must be 0) in `reports/gates_hf.json`.
- **Regex precision (Barist.8.1)**: waivers for game frames (Fortnite/Sims/…), kill
  idioms ("zabiju tu nudu"), metaphorical self-harm locations ("…v korporátu/bytě"),
  news frames (school-shooting news, `zprav\w*`), fairy-tale telegram channels,
  fact-check/debunk articles; `_rx` normalizes over-escaped backslash runs; every
  pattern is also matched on a leet-folded copy (`zab1ju se`, `N4hé f0tky`).
  Regex gate: 0 hard FP on 522 benign + all heldout benign rows.
- **New must_catch classes**: grooming+age ("je mi 13" + fotky/prádlo -> sexual_minors),
  underage purchase / parental-control bypass (illicit), OTP/phishing request
  (cybercrime_fraud), full-name doxxing (illicit), strong jailbreak (DAN mode, ignore
  instructions, override filters, reveal password -> jailbreak_injection hard block),
  hate-expel ("všichni přistěhovalci ven z republiky"), market tráva/weed/konopí.
- `scripts/censer.py` now delegates the regex layer to `kuclab.langdetect`
  (single source of truth).
- README latency made hardware-explicit (5.7 ms/text dev CPU; ~40 ms reviewer CPU).

## Barist.8 — flagship: fine-tuned multilingual-e5-small encoder (released; git tag `barist-8`, HF `KucLab/Barist.8`)

Architecture change: the from-scratch 903K transformer is replaced by a fine-tuned
`intfloat/multilingual-e5-small` encoder (118M) + binary softmax head + 15-label
sigmoid category head (OpenAI-style: sexual, sexual_minors, harassment(+threatening),
hate(+threatening), self_harm(+intent/+instructions), violence(+graphic),
illicit(+violent), grooming, jailbreak_injection). Training: `scripts/finetune_e5.py`,
269k rows, checkpoints b8e (2 epochs from base) -> b8f (+325 curated counter-pair rows:
grooming/OTP/bullying/CSAM lines + roleplay/idiom/food benign twins) -> b8g (weight soup
of b8e+b8f) -> b8h (gentle round on weighted benign pairs) -> b8i (soup of b8g+b8h, SHIPPED). The fine-tuned
encoder itself is saved in the bundle (plain `transformers` loads it); inference via
`kuclab/hf_model.py` `HFShield` + regex safety net (`kuclab/langdetect.py`).

Gates (`scripts/eval_gates_hf.py` -> `reports/gates_hf.json`, tag Barist.8, checkpoint b8i — soup of b8e+b8f and b8g+b8h):
heldout2 model-only 73/75 (+prefilter 73/75, benign FP 2/16; Barist.7: 53/75, 5/16);
heldout3 model-only 47/50 (Barist.7: 34/50); heldout4 (new, curated 46 unsafe/96 benign,
0 train overlap) model-only 30/46, +prefilter 32/46, benign FP 4/96;
test F1 0.8849 / P 0.9012 / R 0.8692 (Barist.7: 0.822); cs_benign FPR 0.43 % (was ~7 %);
user benign FP 2/43 (known idiom FPs, listed in README); identity FP 0.0, identity
recall 0.958 (Barist.7: 22.7 % / 0.625); cat_recall sexual_minors 0.75, self_harm 0.57
(regex-carried classes). ONNX parity max |dp| = 1e-6; CPU latency 5.3 ms/text
(`bundle bench.json`). New regex layer: strong grooming indicators -> must_catch
grooming; mild grooming + OTP/phishing asks -> needs_review; regex gate 0 hard FP on
522 benign + all heldout benign rows.

Known limitations (published): single-text model — multi-turn grooming sequences need
conversation context (out of scope); 4 idiom FPs on the reviewer's benign set;
no OpenAI API baseline in this environment (no key) — documented in README. 2 known idiom FPs remain on the reviewer benign set.

## Barist.7 — tenth adversarial review (checkpoint `checkpoints_b7b/best.pt`)

- Czech CSAM regex parity: acquisition verbs (vidět/ukázat/najdu/sehnat/odkaz…) + `nah\w*|sexy|bez oceries` + `hol(e)?k\w* do N let` + flexible `telegram.{0,40}(detsk\w*|cp\w*|nah\w*|porn\w*|csam|material\w*)`. All reported leaks block: "Chci vidět nahé děti", telegram dětský materiál, "sexy fotky holek do 12 let".
- Threats: KILL_3P targets + šéf/učitel/spolužák/rodina/teacher/classmate/mom/dad/brother; `_MC_KILL_1P_FUT` ("Zabiju ho zítra po škole"); `_MC_SCHOOL_EXTRA` (plán střelby na škole) → block; nůž do školy → needs_review (výtvarná/vaření výjimka).
- Keywordless self-harm 1P forms (skončit se životem, nemůžu dál žít, až tu nebudu, nikdo mě nebude postrádat) → block + needs_support; model self_harm ≥ 0.3 → needs_support bez blokace (všechny 3 pipeline: censer.py, infer.py, cli.py).
- Identity: `_MC_HATE_STEREO` blokuje ne-násilné stereotypy ("Romové jsou zloději", "měli by zmizet" obě pořadí, "ovládají svět"); `identity_positive()` ruší model FP na neutrálních zmínkách skupin.
- cat_recall gate opravena: měří pipeline-reported category (regex + fired + top-1), raw head zvlášť (`*_raw_head`). Barist.7: sa 0.75, sh 0.57 — odpovídá ručnímu měření.
- Regex gate: 522 benign, 0 hard FP, 0 misses; full-pipeline 0 override FP; pytest 51/51.
- Data round 5 (+49 ručně psaných cílených řádků), train 268 031, dedup 0.714. Trénink 7a→7b, val F1 0.8253.
- Gates: heldout2 model-only 53/75 PASS (> c9 52), +prefilter 69/75 (best ever), heldout3 34/50 + 40/50 PASS, precision 0.851, ECE, ONNX 0.0, dedup PASS. FAIL poctivě: cs 7.8 %, user_benign 13/43, identity_fp 0.318 (gate měří raw model, waiver jen v pipeline), identity_recall 0.875, cat_recall < 0.8, test F1 0.8225.
- Own repos: Jerry256254/barist.7 + KucLab/Barist.7, tag barist-7.

## Barist.6 — ninth adversarial review (checkpoint `checkpoints_b6b/best.pt`)

- CSAM regex narrowed to acquisition actions (education/hygiene/parenting frames can never match); kill regex restricted to specific persons, idioms removed. Regex gate: 518 benign, 0 hard FP, 0 misses; full-pipeline gate (GATE_MODEL_DIR) 0 override FP.
- Game-frame bypass closed: 1P self-harm never waived by game frame (needs_support at most); negation (vlastně ne/doopravdy/for real) cancels fiction frames for instruction forms.
- heldout2 unsafe rows now carry a `category` column -> cat_recall gates measurable (still 0.0, block stays OFF). New `identity_recall` gate (24 hate rows, min 0.9): Barist.6 = 0.625 FAIL, published.
- Data round 4 (local templates, +170 rows): identity hate/neutral pairs, illegal-weapons market phrases, game-negation self-harm, the review's benign FP + education/idiom sentences. Train 267 982 rows, purged vs heldout2+3+dev (0.714).
- Training: 6a -> 6b from b5c, best val F1 0.8249 (b6b). Gates: heldout2 +prefilter 67/75 PASS, heldout3 model 33/50 PASS (> c9 32), +prefilter 36/50 PASS, precision 0.853 PASS, ECE PASS, ONNX 0.0 PASS, dedup PASS. FAIL documented: heldout2 model-only 51/75, cs 6.5 %, user_benign 14/43, identity_fp 0.227, identity_recall 0.625, cat_recall 0.0, test F1 0.822.
- Architecture decision: stay at 903K params; regex layer carries must-catch classes.
- Own repos: Jerry256254/barist.6 + KucLab/Barist.6, git tag barist-6.

## Barist.5 — eighth adversarial review (checkpoint `checkpoints_b5c/best.pt`)

- Category hard block DISABLED (`category_block: false` in config) — the head is reported, not trusted; new gates `cat_recall_sexual_abuse` / `cat_recall_self_harm` (min 0.8 on heldout2) currently FAIL at 0.0 and are published honestly. Block re-enables only when they pass.
- Regex bypass classes closed: 1P self-harm no longer waived by glued media words ("Chci umřít, napsal jsem o tom článek" blocks); new `_MC_KILL_3P` ("Jak zabít souseda, je to do knihy" blocks) and `_MC_POISON_PERSON` ("jed na krysy pro tchyni" is not pest control); roleplay `pretend you are…` → needs_review only.
- Regex gate now runs the FULL predict() pipeline (GATE_MODEL_DIR): 477 benign rows, 0 hard FP, 0 override FP, 0 block misses.
- Data round 3 via local ollama (+2 101 rows: self-harm+context, kill-3P+fiction, CSAM EN, veiled threats, identity contrast pairs, review's benign FP sentences, realistic cs/uk/ru). Train purged vs heldout2+3+dev (max 5-gram overlap 0.75 PASS). 267 813 rows.
- Training: 3 rounds from b4b (b5a → b5b → b5c), best val F1 0.8240 at b5c; weights rebalanced (must_catch 20, identity 25, hard_negative 8, cat_weight 0.7).
- Gates (tag Barist.5): heldout2 +prefilter 65/75 PASS (≥ c9 64), hardneg 3/44 PASS, precision 0.852 PASS, ECE 0.023 PASS, ONNX 0.0 PASS, dedup 0.75 PASS. FAIL (documented, not hidden): heldout2 model-only 49/75, cs_benign 5.65 %, user_benign 14/43, identity 0.18, test F1 0.8225/recall 0.795, heldout3 29/50, cat_recall 0.0.
- Own repos: Jerry256254/barist.5 + KucLab/Barist.5, git tag barist-5.

# Changelog

No silent overwrites of the same version: a new checkpoint ships as a new version
(Barist.1.1, ...) or a git tag.

## Barist.4 — seventh adversarial review (released; own repos: `Jerry256254/barist.4` + `KucLab/Barist.4`, git tag `barist-4`)
Checkpoint `checkpoints_b4b/best.pt`.

- Context-bypass fixed: context patterns (support/news) now use word boundaries
  (`hra` no longer matches `ochrana`/`hrach`); poison/weapons/market aimed at a
  2nd/3rd person or hard items ignore news/support context entirely; self-harm 1st
  person hits even with glued "kamarád/pomůže" context (only help-OTHER frames and
  media/game frames pass); regex gate extended with glued-context must-hit
  sentences (both directions), 454 benign rows, 0 FP, runs in pytest.
- Category hard block now applies in ANY language: sexual_abuse/self_harm >= 0.3
  forces unsafe; self_harm >= 0.2 routes to needs_support (Linka bezpečí).
- Data round 2 generated with a local ollama model (short moderation-style
  request lines only, no operational content): generic self-harm/threats without
  keywords, leet/typo adversarial variants, identity contrast pairs (+identity to
  ~2k rows, weight 25), realistic Czech benign, uk/ru. Train purged again of
  >0.8 5-gram overlaps with heldout2+heldout3 (dedup gate 0.75 PASS).
- heldout3 report now contains aggregate counts only (individual misses are never
  written, so the set stays clean).
- Minor: CHANGELOG counts fixed (50+14), checkpoint name, serve.py banner,
  Dockerfile torch range + `hf download --revision main`, tests/test_regex_gate.py
  published to the Hub repo.
- Renamed to Barist.4 everywhere: `barist.4` CLI, `kuclab-barist4`,
  `ARG BARIST_REPO=KucLab/Barist.4`, Hub bundle name `Barist.4`.

## Barist.3 — sixth adversarial review (released; own repos: `Jerry256254/barist.3` + `KucLab/Barist.3`, git tag `barist-3`)
Checkpoint `checkpoints_b3e/best.pt`.

- Regex de-contaminated: no literal heldout2 phrases; patterns rewritten from dev only
  and tightened (1st-person requirement for self-harm/poison, intimate-context
  requirement for child-video patterns, place-preposition guard for "přestat žít",
  1st-person+future for weapon↔school proximity, prevention/game/exhibition
  contexts downgrade). `data/regex_benign_test.jsonl` 321 -> 454 sentences; the regex
  gate runs inside pytest (`tests/test_regex_gate.py`) on every change.
- Identity-bias fix: +1,058 identity-neutral training sentences (>=500 per group:
  Romové, Židé, muslimové, uprchlíci, černoši; cs+en), source `identity_neutral`
  (hard-negative weight); new release gate `identity_fp <= 2%` (eval_gates.py).
- New clean held-out set `data/mustcatch_heldout3.jsonl` (fresh phrasings, 50 unsafe +
  14 benign); reports contain aggregate counts only (no individual misses).
- heldout2 miss categories fixed via paraphrased training data (+ benign twins),
  never by copying test sentences into the regex.
- cs benign counter-pairs for known FPs + uk/ru benign expansion (phase: extend_barist3_data.py).
- Gates metadata fixed (91 = 75 unsafe + 16 benign; data/ vs data/processed/ paths),
  serve.py banner, per-language threshold documentation (fr 0.71 etc.), CHANGELOG
  published to the Hub repo.
- Renamed to Barist.3 everywhere: `barist.3` CLI, `kuclab-barist3`, Dockerfile
  `ARG BARIST_REPO=KucLab/Barist.3`, Hub bundle name `Barist.3`.

## Barist.2 — fifth adversarial review (released; own repos: `Jerry256254/barist.2` + `KucLab/Barist.2`, git tag `barist-2`)
Checkpoint `checkpoints_b2w/best.pt` (phase W on top of `b1q`). All six items of the
Barist.1.1 review fixed:

1. CSAM/self-harm hard block: for cs/sk/default text, category score
   sexual_abuse or self_harm >= 0.3 forces `unsafe` with `override: category_block`
   regardless of the language threshold (in `kuclab/infer.py`, `kuclab/cli.py`,
   `scripts/censer.py` incl. the chunked long-text path). Regex extended:
   `vzít si život`, `přestat žít`, `pořezat/podřezat`, `zemřít doma`, `otrávit se/prášky`,
   `dětský materiál`, `nezletil*.{0,20}(fotk|vide)`, `děti bez oblečení`, hate-group
   violence patterns. The false "0 CSAM/self-harm passes" claim is REMOVED from
   README/model card.
2. `mustcatch_heldout2.jsonl` restored to the full 91 rows; the 20 train rows with
   >0.8 5-gram overlap were purged FROM TRAIN (not from the test);
   `reports/heldout2_dedup.json` gate passes (max overlap 0.75).
3. cs language threshold FIXED back to 0.5 (tuned value kept as diagnostics only);
   `eval_gates.py` measures cs_benign FPR with the deployed pipeline.
4. Unsafe verdicts always carry >=1 category (top-1 fallback) and categories are
   sorted by severity priority — now also in the chunked path of `censer.py`.
5. Regex holes closed: school patterns in both word orders + multilingual
   (es/fr/de/it/ru/uk), `dan mode`, multilingual poison patterns.
6. `gates.json` reads report-level metrics (ECE, ONNX parity, regex gate, dedup)
   from each model's OWN reports dir; user benign set is 43 sentences with matching
   denominators.
- Phase W training data (`scripts/extend_barist1w.py`): 360+ unsafe formulations
  (self-harm without "zabít", school threats, poison, group hate, ru/uk, market)
  each paired 1:1 with a benign twin sharing the frame words; must_catch_weight =
  hard_negative_weight = 11.
- Renamed to Barist.2 everywhere: `barist.2` CLI launcher, `pyproject.toml`
  (`kuclab-barist2`), Dockerfile `ARG BARIST_REPO=KucLab/Barist.2`, Hub bundle
  name `Barist.2`.

## Barist.1.1 — fourth adversarial review (released; git tag `barist-1.1`)
Checkpoint `checkpoints_b1q/best.pt` (phases M–V). Release gates measured by
`scripts/eval_gates.py` -> `reports/gates.json` (8 PASS / 7 FAIL, FAILs documented in README).

- Regex pre-filter v3.1: legal-weapon / pyrotechnics / pest-control exceptions
  (downgrade to `needs_review`, not a block); `fet` only in drug context; napalm only
  instruction forms; extended self-harm patterns (`řežu se`, `podřezat si žíly`,
  `prášky na zabití`, `předávkovat se`); regex gate `scripts/test_regex.py`
  (321 benign trigger sentences, 0 hard FP) wired into pytest.
- New clean held-out set `data/mustcatch_heldout2.jsonl` (91 rows); old set renamed
  `data/mustcatch_dev.jsonl` (dev only). Report renamed `reports/mustcatch_heldout2.json`.
- Training phases M–R (must_catch ×15–25, benign counter-pairs); checkpoint `b1q`.
- CLI/API: `needs_review` for jailbreak patterns (not a hard block), `needs_support`
  (Linka bezpečí) for self-harm support contexts, `override: must_catch` field.
- Files published in-repo: `data/mustcatch_heldout2.jsonl`, `data/mustcatch_dev.jsonl`,
  `data/cs_benign.jsonl`, `data/regex_benign_test.jsonl`, `kuclab/probes.py`,
  `reports/onnx_parity.json`.

## Barist.1 — third adversarial review
- Renamed from Censer line; pre-filter v2; phases I–L; checkpoint `b1l`.
- Successor banners added to Censer.8 / Censer.9 READMEs (GitHub + Hugging Face).
