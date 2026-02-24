# ALLES (Gesamtplanung) und „ohne Fallback“ – wissenschaftliche Prüfung

**Stand:** Nach Implementierung der SOTA 300M Gesamtplanung.  
**Prüfdatum:** Erneute Verifikation durch Abgleich Plan ↔ Code (inference/, training/, data/, model/, evaluation/).  
**Fragen:** (1) Ist **alles** aus der Gesamtplanung (hohe/mittlere Priorität) korrekt eingebunden? (2) Ist **ohne Fallback** durchgängig eingehalten?

---

## Quellenverifikation (Code-Abgleich)

- **run_chat.py:** Zeilen 31–35 `ValueError` bei fehlendem config/model; Zeile 148–149 `FileNotFoundError` wenn `_j.exists()` False.
- **run_torch.py:** Zeile 77–78 `FileNotFoundError` wenn Tokenizer-Datei fehlt; kein Laden ohne existierenden Pfad.
- **run_mlx.py:** Zeile 131 `--model` required; Zeile 139 `FileNotFoundError` bei Load-Fehler; Zeile 118–119 `ValueError` bei require_model und fehlendem Modell.
- **rl_train.py:** Zeile 42 Kommentar „kein Fallback“ (EOS/PAD); Zeilen 138–141 AERO Zero-Advantage-Skip ohne backward/step; Zeilen 183, 209, 224, 271 `FileNotFoundError`/`ValueError`/`AttributeError` bei fehlenden RL-Daten/Tokenizer.
- **train.py:** Zeilen 233–234 `data_loader is None` → `FileNotFoundError`.
- **data/dataloader.py:** Zeilen 50–52 `FileNotFoundError`; Zeile 139 `ValueError` bei fehlendem pad/eos.
- **data/sft_dataloader.py:** Zeilen 21, 66, 71 `FileNotFoundError`; Zeilen 76, 78 `ValueError`; Kommentar Zeile 72 „kein Fallback/Dummy“.

---

## Teil 1: ALLES – Abgleich Plan ↔ Codebase

### 1.1 Hohe Priorität (Plan §D)

| Maßnahme | Plan | Umsetzung | Referenz |
|----------|------|-----------|----------|
| **MoA in Attention** | Sparse-Patterns pro Head/Layer; 8K→32K+ Kontext | `model/moa.py` (make_moa_sparse_mask, apply_moa_mask); `model/config.py` use_moa, moa_patterns; `model/gpt.py` Attention.forward bei use_moa | ✅ |
| **2-GRPO oder AERO** | Weniger Rollouts; Zero-Advantage vermeiden | `training/rl_train.py`: --num-candidates (z. B. 2), skip_zero_advantage (AERO); kein Backward bei std_reward < 1e-6 | ✅ |
| **S* mit distinguishing inputs** | Tests, die Kandidaten unterscheiden | `inference/test_time_evolution.py`: generate_differentiating_tests(), s_star_generate(..., differentiating_generator=...) | ✅ |
| **SelfCodeAlign-Style Instruction** | Konzepte aus Code → Sandbox-Filter → nur grüne Paare | `data/instruction_data.py`: extract_concepts_from_code, generate_teacher_response_multi, filter_selfcodealign_green; `data/generate_instruction_data.py`: --selfcodealign, --seed_code_file, --tests_file | ✅ |
| **RLM-Loop vollständig** | REPL: Code → Sandbox → Kontext → Iteration | `inference/run_mlx.py`: rlm_repl_loop(), rlm_generate(), _execute_code_safely(), _extract_code_block(); --model required | ✅ |
| **Speculative Decoding** | Draft 15–30M für 300M; 2–4× schnellere Decode | `inference/speculative.py`: speculative_decode, draft_generate, speculative_decode_step; `inference/run_torch.py`: --draft-checkpoint, --speculative-k | ✅ |
| **LongCodeBench-Eval + Repair Rate** | 32K/128K; Repair Success Rate | `evaluation/eval_lcb.py`: run_longcodebench_with_repair(), repair_success_rate aus eval_repair; CLI --data, --context-lengths | ✅ |

### 1.2 Mittlere Priorität (Plan §D)

| Maßnahme | Plan | Umsetzung | Referenz |
|----------|------|-----------|----------|
| **ORPO / SimPO** | Optional nach SFT; kein Reference-Modell | Nicht implementiert (Plan: „optional“) | ⏳ optional |
| **Genetic Instruct / Auto Evol-Instruct** | Evolutions-Algorithmen für Instructions | Evolve-Instruct + SelfCodeAlign vorhanden; vollständiger Genetic Instruct (Crossover, Judge) nicht | ⏳ teilweise |
| **KV-Cache-Optimierung** | Cache komprimieren/prunen | Nicht implementiert | ⏳ optional |
| **BitNet Median-Skalierung** | Robustere Skalierung für kleine Modelle | `model/bitnet.py`: use_median_scaling; `model/config.py`: use_bitnet_median_scaling; gpt.py _linear(..., use_median_scaling) | ✅ |
| **IA2 (ICL Activation Alignment)** | Aktivierungen vor SFT an ICL anpassen | Nicht implementiert | ⏳ optional |

### 1.3 Bestehend (Plan §A/D – unverändert gefordert)

| Komponente | Status |
|------------|--------|
| BLT, Mamba-2-Hybrid, LEAM++, BitNet, L-MTP, QK-Norm, Value Residual | ✅ config/gpt/bitnet/mamba_hybrid/leam |
| NorMuon, WSD, Stage-LRs, EMA, CODA, CodeDenoise, RegMix | ✅ training/, data/ |
| SFT (ChatML, Assistant-Only), GRPO, CodeRL+, Execution-Reward | ✅ sft_train, rl_train, execution_reward, coderl_plus |
| S* (Kandidaten + Sandbox), LEAM Constrainer; Stubs PoT, AB-MCTS, DaJ | ✅ test_time_evolution, run_chat, run_torch |

**Fazit ALLES:** Alle **hohen** Prioritäten der Gesamtplanung sind umgesetzt. Mittlere: BitNet-Median und SelfCodeAlign/Evol-Instruct anteilig; ORPO/SimPO, KV-Cache, IA2 bewusst optional. Keine stillen Lücken im kritischen Pfad.

---

## Teil 2: Ohne Fallback – systematische Prüfung

Vorgabe (PLAN_COMPLIANCE, DESIGN): **Keine Dummy-/Fallback-Configs oder -Batchs**; fehlende Ressourcen → sofortiger Abbruch mit `FileNotFoundError` / `ValueError`.

### 2.1 Inferenz

| Datei | Erwartung | Prüfung |
|-------|-----------|---------|
| **run_chat.py** | Checkpoint muss `config` und `model` enthalten; Tokenizer-Pfad muss existieren | ✅ `ValueError` bei fehlendem config/model; `FileNotFoundError` wenn Tokenizer-Datei fehlt |
| **run_torch.py** | Tokenizer-Pfad Pflicht; kein Fallback-Tokenizer | ✅ Tokenizer aus --tokenizer/Path; kein Default auf HF-Auto |
| **run_mlx.py** | RLM ohne stillen Fallback (z. B. grep) | ✅ `--model` required; bei Load-Fehler `FileNotFoundError`; `require_model=True` → bei fehlendem Modell `ValueError` (kein grep-Fallback im Produktionspfad) |

### 2.2 Training

| Datei | Erwartung | Prüfung |
|-------|-----------|---------|
| **rl_train.py** | EOS/PAD nur aus ModelConfig; AERO: kein Optimizer-Step bei Zero Advantage | ✅ Kommentar „kein Fallback“; eos/pad aus model.config; skip_zero_advantage → return ohne backward/step |
| **train.py** | Dataloader Pflicht; kein Fallback-Batch | ✅ `data_loader is None` → `FileNotFoundError` |
| **sft_train.py** | Instruction-JSONL und Tokenizer Pflicht | ✅ SFTDataLoader erhebt FileNotFoundError/ValueError bei fehlendem Pfad/Tokenizer |

### 2.3 Daten & Tokenizer

| Datei | Erwartung | Prüfung |
|-------|-----------|---------|
| **data/dataloader.py** | Tokenizer nur aus konfiguriertem Pfad/vocab; kein HF-Auto-Fallback. pad/eos aus Tokenizer, sonst Abbruch | ✅ `load_tokenizer_for_training` → FileNotFoundError wenn nichts gefunden; CodeDataLoader: pad_id/eos aus Tokenizer, sonst `ValueError` (Kommentar: „no fallback“) |
| **data/sft_dataloader.py** | Special-Token-IDs nur aus Tokenizer; kein Dummy | ✅ pad/eos via getattr(tokenizer, …); wenn _pad/_eos None → ValueError. Kommentar: „kein Fallback/Dummy“ |

### 2.4 Sonderfälle (kein Verstoß gegen „ohne Fallback“)

- **Optionales `getattr(..., None)` für Config-Optionen** (z. B. use_leam, use_moa, moa_patterns): Legt nur Defaults für **optionale** Features fest; es werden keine kritischen Ressourcen ersetzt. ✅  
- **pad_token_id or eos_token_id**: Erlaubt „pad = eos“, wenn der Tokenizer nur einen davon setzt. Kein Dummy-Wert, sondern Konvention aus dem Tokenizer. ✅  
- **verify_plan.py**: RLM-Test mit `require_model=False` nur für Modul-Test ohne echtes Modell; Produktionsaufruf (main) bleibt mit `--model` required und require_model=True. ✅  

**Fazit ohne Fallback:** In allen kritischen Pfaden (Inferenz, Training, Daten, Tokenizer, RL, RLM) gibt es **keinen** stillen Fallback auf Dummy-Configs, Platzhalter-Batchs oder Ersatz-Ressourcen. Fehlende Ressourcen führen zu klarem Abbruch.

---

## Teil 3: AERO (Zero-Advantage)

- **AERO** (arXiv:2602.14338): Zero-Advantage-Situationen vermeiden; kein Update ohne Lernsignal.
- **Umsetzung:** `skip_zero_advantage=True` (Default); bei `std_reward < 1e-6` wird **kein** `backward()` und **kein** `optimizer.step()` ausgeführt; explizit **kein Fallback** auf einen nutzlosen Step.
- **Referenz:** In rl_train.py mit arXiv:2602.14338 referenziert.

---

## Teil 4: Kurzfassung

| Prüfpunkt | Ergebnis |
|-----------|----------|
| **ALLES (Hohe Priorität)** | MoA, 2-GRPO/AERO, S* distinguishing, SelfCodeAlign, RLM REPL, Speculative Decoding, LongCodeBench+Repair, BitNet-Median: in Codebase eingebunden und referenziert. |
| **ALLES (Mittlere/Optionale)** | ORPO/SimPO, KV-Cache, IA2 optional; Genetic Instruct anteilig (Evol-Instruct + SelfCodeAlign). |
| **Ohne Fallback** | run_chat, run_torch, run_mlx, rl_train, train, sft_train, dataloader, sft_dataloader: keine stillen Fallbacks; Abbruch bei fehlenden Ressourcen. |
| **AERO** | Zero-Advantage-Skip korrekt; kein Fallback auf Optimizer-Step ohne Lernsignal. |

**Wissenschaftliche Bewertung:** Die Gesamtplanung ist in den gesetzten Prioritäten umgesetzt; „ohne Fallback“ ist im kritischen Pfad durchgängig eingehalten und mit der Plan-Compliance abgeglichen. Die obigen Zeilennummern ermöglichen eine direkte Nachprüfung im Code.
