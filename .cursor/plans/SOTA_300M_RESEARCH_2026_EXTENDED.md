# SOTA 300M Code-LLM: Erweiterte Forschungsübersicht 2025–2026

**Stand:** 2026. Maximale Informationsdichte aus paralleler Recherche (Subagents).  
Fokus: Alles, was uns **besser**, **schneller** oder **smarter** macht – bei gleichzeitig **kleinem Modell** (max 300M Parameter).  
Jeder Eintrag: **Für Laien** (was es macht) | **Was es bringt** (Nutzen-Schätzung) | **Hält klein?** (Parameter/Compute).

---

## Übersicht: Besser · Schneller · Smarter

| Kategorie | Bedeutung |
|-----------|-----------|
| **Besser** | Höhere Accuracy (pass@1, Benchmarks), robustere Instruction-Following, bessere Code-Qualität. |
| **Schneller** | Weniger Training-Compute, schnellere Inferenz, weniger VRAM, weniger Rollouts. |
| **Smarter** | Sample-Effizienz, Long-Context, Test-Time-Compute, Self-Repair, bessere Daten-Nutzung. |

---

# TEIL 1: BESSER (Accuracy, Instruction, Code-Qualität)

## 1.1 Kleine Code-LLMs (Sub-500M / Sub-300M)

| Ansatz | Für Laien | Was es bringt | Hält klein? | Quelle |
|--------|-----------|----------------|-------------|--------|
| **SmallCoder 303M** | Ein 303M-Modell wird in 4 Stufen trainiert (Sprache → Code → Math → SFT) und erreicht Spitzenwerte unter 500M. | 27,4% HumanEval, 31% MBPP pass@1; ~23× kleiner als Mistral 7B bei ähnlicher Coding-Qualität. | Ja (303M). | SmallCoder 303M, Web 2025. |
| **SmolLM2 data-centric** | Ein kleines Modell wird auf ~11T Tokens mit gestaffeltem Mix aus Web, Math, Code und Instruction-Daten trainiert. | Übertrifft Qwen2.5-1.5B und Llama3.2-1B bei 1.7B. | Ja (1.7B / 360M). | arXiv:2502.02737. |
| **Mify-Coder 2.5B** | 2.5B Code-Modell mit kuratierten + synthetischen Daten und LLM-Qualitätsfilter. | Vergleichbar mit größeren Modellen bei Coding und Function-Calling; quantisiert auf Desktop nutzbar. | Ja (2.5B). | arXiv:2512.23747. |
| **Maincoder-1B** | 1B-Modell mit besserer Datenverarbeitung und RL-basiertem Post-Training. | Bis 76% HumanEval-Niveau bei geringer Latenz. | Ja (1B). | Maincode / Web 2025. |
| **Empirische SLM-Studie** | 20 SLMs (0.4B–10B) auf HumanEval, MBPP etc.: kompakte Modelle können konkurrenzfähig sein; ~10% mehr Accuracy oft nur mit ~4× VRAM. | Orientierung: kleine Modelle bleiben effizient; große Sprünge brauchen mehr Kapazität. | Ja (0.4B–10B). | arXiv:2507.03160. |

## 1.2 Instruction Following & SFT / Alignment

| Ansatz | Für Laien | Was es bringt | Hält klein? | Quelle |
|--------|-----------|----------------|-------------|--------|
| **IA2 (ICL Activation Alignment)** | Vor dem SFT werden die inneren Aktivierungen des Modells so angepasst, dass sie „In-Context-Learning“-Verhalten nachahmen. | Bessere Accuracy und Kalibrierung auf 12 Benchmarks; ICL-ähnliches Verhalten ohne lange Prompts. | Ja (keine Extra-Parameter). | arXiv:2509.22621. |
| **ORPO** | SFT und Preference-Optimization in einer Phase, ohne separates Reference-Modell. | Bis +12,2% Win-Rate auf AlpacaEval 2.0; 125M–7B getestet. | Ja. | ORPO (ACL/EMNLP). |
| **SimPO** | Reference-freier Reward (durchschnittliche Log-Prob als impliziter Reward) + Ziel-Marge zwischen Gewinner/Verlierer. | Bis +6,4 Punkte AlpacaEval 2, +7,5 Arena-Hard vs. DPO; weniger Speicher/Compute. | Ja. | arXiv:2405.14734. |
| **SelfCodeAlign** | Konzepte aus Seed-Code → neue Tasks → mehrere Antworten → Sandbox-Validierung → nur bestandene Beispiele für SFT. | 67,1 pass@1 HumanEval+ mit 7B; übertrifft CodeLlama-70B-Instruct. | Ja (7B; Prinzip auf 300M übertragbar). | NeurIPS 2024; BigCode. |
| **RLVR für präzise Instruktionen** | Verifizierbare Rewards (z. B. Execution) in RL verbessern das Befolgen von präzisen, eingeschränkten Anweisungen. | Bessere Generalisierung auf ungesehene Constraints (z. B. IFEval-Style). | Ja. | arXiv:2507.02833. |
| **SFT-Best-Practices (kleine LLMs)** | Größere Batch-Größen mit niedrigerer Lernrate; frühe Training-Dynamik (Gradient-Norm, Loss) sagt finale Performance vorher. | Bessere MMLU/MT-Bench; Early Stopping spart Compute. | Ja. | arXiv:2412.13337. |

## 1.3 HumanEval / MBPP & pass@k

| Ansatz | Für Laien | Was es bringt | Hält klein? | Quelle |
|--------|-----------|----------------|-------------|--------|
| **SvS (Self-Play + Variational Problem Synthesis)** | Viele Varianten derselben Aufgabe (gleiche Referenzantwort) werden erzeugt, damit die Policy nicht kollabiert. | +18,3% und +22,8% pass@32 auf competition-style Benchmarks; 3B–32B. | Ja. | arXiv:2508.14029. |
| **PKPO (Pass@K Policy Optimization)** | RL optimiert direkt pass@k (z. B. pass@4) mit low-variance Schätzern statt nur pass@1. | Stärkeres pass@1 und pass@k; bessere Exploration bei schweren Aufgaben. | Ja. | arXiv:2505.15201. |
| **Top Pass Ranking** | Code-Samples werden nach pass@k-artigem Loss gerankt; bestes Sample wird zuerst gewählt. | ~32,9% relative pass@1-Verbesserung auf CodeContests. | Ja (nur Ranking). | Frontiers of Computer Science 2025. |
| **ACECODER** | RL mit Execution-Reward und synthetisierten Testfällen. | >25% HumanEval-plus, ~6% MBPP-plus mit ~80 Optimierungsschritten; 7B vergleichbar mit 236B in best-of-32. | Ja (7B–32B). | ACL 2025; arXiv:2502.01718. |
| **RLEF** | End-to-End-RL mit Execution-Feedback. | SOTA auf Competitive Programming mit 8B/70B; ~10× weniger Samples als bisherige RL. | Ja. | arXiv:2410.02089. |

---

# TEIL 2: SCHNELLER (Training, Inferenz, weniger Compute)

## 2.1 Training & Optimizer

| Ansatz | Für Laien | Was es bringt | Hält klein? | Quelle |
|--------|-----------|----------------|-------------|--------|
| **NorMuon** | Optimizer mit neuronweiser Normalisierung; weniger Varianz in Updates als reines Muon. | ~21% weniger Trainingsschritte vs. AdamW, ~50% weniger VRAM für Optimizer States. | Ja. | NorMuon OpenReview/arXiv:2510.05491. |
| **2-GRPO** | Nur 2 Rollouts pro Prompt statt 16; GRPO wird als kontrastives Lernen interpretiert (nahe DPO). | ~98,1% der 16-GRPO-Performance mit 12,5% der Rollouts; >70% kürzere Trainingszeit. | Ja; weniger Rollouts = weniger Compute. | OpenReview „It Takes Two“; arXiv:2510.00977. |
| **AERO (Adaptive Rollout Optimization)** | Adaptive Rollout-Strategien, selektive Ablehnung, Bayesian Posteriors gegen Null-Advantages. | ~48% weniger Training-Compute, ~45% weniger Wall-Clock; gleiche oder bessere Performance. | Ja. | arXiv:2602.14338. |
| **PODS (Down-Sampling Rollouts)** | Nur eine Teilmenge der Rollouts wird für Updates genutzt (max-Varianz Down-Sampling). | GRPO mit PODS erreicht Spitzen-Genauigkeit mindestens 1,7× schneller als Vanilla-GRPO. | Ja. | arXiv:2504.13818. |
| **L-MTP (Multi-Token Prediction)** | Modell sagt 2–4 nächste Tokens vorher; bei Inferenz weniger Decode-Schritte. | +12% HumanEval, +17% MBPP (13B); bis 3× schnellere Inferenz. | Ja (MTP-Head ist klein). | Gloeckle et al., ICML 2024. |

## 2.2 Inferenz-Geschwindigkeit

| Ansatz | Für Laien | Was es bringt | Hält klein? | Quelle |
|--------|-----------|----------------|-------------|--------|
| **Speculative Decoding (kleiner Draft)** | Ein kleines „Draft“-Modell schlägt Tokens vor; das Hauptmodell prüft sie in einem Batch. Weniger schwere Hauptmodell-Schritte. | 1,5–6,5× schnellere Decode (setting-abhängig); lossless. | Ja: Hauptmodell bleibt 300M; Draft 10–20× kleiner (~15–30M). | Diverse 2025–2026. |
| **Cascade Decoding** | Zuerst kleines Modell; nur bei Unsicherheit wird ein größeres Modell aufgerufen. | ~40–60% weniger Kosten/Latenz; Qualität durch gezieltes Deferral. | Ja: 300M als erste oder einzige Stufe. | CASCADIA etc. |
| **Multi-Candidate / Adaptive Acceptance** | Mehrere Kandidaten oder variable Länge beim Spekulieren; längster korrekter Präfix wird akzeptiert. | ~2–4× über Baseline Speculative; lossless. | Ja: weniger Hauptmodell-Schritte pro Token. | SpecDec++, HeteroSpec, EARS. |
| **Draft–Target Vocab Mismatch** | Draft und Hauptmodell können unterschiedliche Tokenizer haben; Match über String oder Semantik. | Bis ~2,8–5× Speedup; lossless oder >99% Accuracy. | Ja: beliebiger kleiner Draft ohne geteiltes Vocab. | SLEM, TLI, FLy. |
| **Quantisierung (FP8 / INT4)** | Gewichte (und ggf. KV) in 8-bit oder 4-bit; weniger Speicher und oft schnellere Kernel. | ~1,2–1,7× schnellere Inferenz; 2–4× weniger Speicher. | Ja: 300M passt in ~0,4–0,6 GB (INT4). | Diverse 2025. |
| **KV-Cache-Optimierung** | Cache wird komprimiert oder gepruned; weniger Speicher und Bandbreite. | ~2–2,5× höherer Durchsatz; ~4× kleinerer Cache bei <1% Qualitätsverlust. | Ja: lange Kontexte bei 300M ohne OOM. | SmallKV, KeepKV, KVCrush, MiniKV. |
| **Early Exit** | Bei ausreichendem Konfidenzgrad wird in einer mittleren Schicht „ausgestiegen“; Rest der Schichten wird übersprungen. | ~1,25–2,6× Speedup. | Ja: gleiche Parametergröße, weniger Compute pro Token. | Context-aware Exit, Multi-Model Exit. |
| **Hardware-in-the-Loop Design** | Architektur und Operatoren für konkrete Hardware (CPU, Edge) optimiert. | Bis ~2× schnellere Prefill/Decode auf CPU; >20 tok/s auf Consumer-CPU bei Q4. | Ja (z. B. Nemotron-Flash, LFM2, SmallThinker). | 2025–2026. |

## 2.3 Architektur (schneller bei gleicher Qualität)

| Ansatz | Für Laien | Was es bringt | Hält klein? | Quelle |
|--------|-----------|----------------|-------------|--------|
| **BitNet b1.58 / Ternary** | Gewichte nur -1, 0, +1; ~1,58 Bit pro Parameter. Training von Anfang an so. | Deutlich weniger Speicher; ~5–6× schnellere Inferenz; Parity mit FP16 bei gleicher Größe. | Ja; Median-Skalierung für kleine Modelle empfohlen. | JMLR 2025; BitNet b1.58. |
| **Hybrid Transformer–SSM (Mamba)** | Wenige Attention-Layer für Recall; viele SSM-Layer für lineare Kontext-Skalierung. | ~8× schnellere Inferenz bei langen Sequenzen; weniger KV-Cache. | Ja (euer 43/7/50 Split). | Mamba-2, Samba, Hymba. |
| **MoE (Mixture of Experts)** | Pro Token nur wenige „Experten“ aktiv; Router wählt. | +3–7% Validation Loss, 3,2× Inferenz-Speedup, ~68% weniger KV-Cache bei sub-300M MoE. | Ja (17M–202M aktive Parameter in Studien). | MoE-MLA-RoPE, FLAME-MoE. |
| **Parameter Sharing** | Gleiche Schicht mehrfach genutzt (rekursiv) oder geteilte „Basis“-Gewichte über Schichten. | Weniger Parameter bei gleicher oder besserer Accuracy; ~47% niedrigere First-Token-Latenz in Encoder-Decoder. | Ja (SLMs, recursive, basis sharing). | Relaxed recursive, SLlama. |
| **Deep-Thin + GQA** | Mehr Schichten, schmaler; Grouped-Query-Attention. | +2,7–4,3% Accuracy bei 125M/350M vs. breit-flach. | Ja (weniger Parameter als breit-flach). | MobileLLM 2024. |

---

# TEIL 3: SMARTER (Sample-Effizienz, Long-Context, Test-Time, Daten)

## 3.1 Sample-Effizienz & Daten

| Ansatz | Für Laien | Was es bringt | Hält klein? | Quelle |
|--------|-----------|----------------|-------------|--------|
| **IMU-1-Style** | QK-Norm, per-head Gating, Value Residuals, LayerNorm-Scaling; Stage-LRs, Checkpoint-EMA; NorMuon. | 430M auf 72B Tokens erreicht Level von Modellen mit 56× mehr Tokens. | Ja (430M). | arXiv:2602.02522. |
| **RegMix** | Kleiner Proxy wird auf vielen Mixturen trainiert; Regression sagt optimale Mischung für großes Modell vorher. | ~6,3% über menschliche Mischung bei ~2% Extra-FLOPs; DoReMi-level bei ~10% DoReMi-Compute. | Ja. | ICLR 2025; euer regmix_proxy.py. |
| **PreSelect / DataDecide** | Daten werden vor dem Training nach Nutzen ausgewählt (Proxy oder Modell sagt vorher, welche Daten helfen). | 30B ausgewählte Tokens schlagen 300B unselektiert (~10× Compute-Reduktion); 80% Accuracy bei 1B-Vorhersage mit 150M-Proxy. | Ja. | PreSelect, DataDecide. |
| **Curriculum Learning** | Daten nach Schwierigkeit sortiert; von leicht zu schwer. | 18–45% weniger Schritte bis Baseline; bis ~3,5% anhaltender Gewinn als Warmup. | Ja. | Difficulty curricula, DoReMi-style. |
| **Deduplikation** | Exakte und Near-Duplicate werden entfernt. | Deutlich weniger Memorization; bessere Valid-Metriken; bis ~19,6% Perplexity, ~28% Laufzeit bei 10–30% Duplikation. | Ja. | Lee et al.; EP-MPD 2025. |
| **Synthetische Daten** | Training auf von LLM erzeugten Beispielen; ggf. aktiv nach Studenten-State. | Bis +101% auf GSM8K bei 1B vs. instruction-only; SynAlign reduziert Distribution Shift. | Ja. | BARE, Active Synthetic, SynAlign. |
| **Genetic Instruct / Auto Evol-Instruct** | Evolutions-Algorithmen für Coding-Instructions: Crossover, Mutation, Judge-LLM. | 7,5M Paare; bessere Code-Results; Auto Evol-Instruct übertrifft manuelle Evol-Instruct auf MT-Bench, HumanEval. | Ja. | Genetic Instruct ACL 2025; Auto Evol-Instruct. |

## 3.2 Long-Context (128K–1M)

| Ansatz | Für Laien | Was es bringt | Hält klein? | Quelle |
|--------|-----------|----------------|-------------|--------|
| **MoA (Mixture of Sparse Attention)** | Pro Head/Layer unterschiedliche Sparse-Patterns (Window, Global, Dilated); kein Re-Training nötig. | 3,9× effektiver Kontext; 1,5–7,1× bessere Retrieval-Genauigkeit; 8K→32K+ bei 150M. | Ja. | arXiv:2406.14909; thu-nics/MoA. |
| **Infini-Attention für SLMs** | Komprimierter Langzeit-Speicher (Infini-Attention-Style) in kleinen Transformern. | ~31% Accuracy-Gewinn bei 16K vs. Baseline für 300M; begrenzter Speicher. | Ja (300M). | arXiv:2512.23862; 2404.07143. |
| **LongMamba (training-free)** | „Global channel“-Flaschenhälse; unwichtige Tokens werden gefiltert, damit der State bei langen Sequenzen nicht explodiert. | Längerer effektiver Kontext ohne Re-Training. | Ja. | arXiv:2504.16053. |
| **QMambaExtend** | Discretization-Step (Δt) pro Layer bei Inferenz kalibriert; kein Training. | Bis 32× Kontext-Verlängerung (2K→64K); ~2,1× weniger Speicher mit Quantisierung. | Ja. | ICLR 2025 Workshop. |
| **MiniCPM-SALA (Hybrid Sparse + Linear)** | Einige Layer sparse (präziser Recall), die meisten linear (globaler Fluss). | ~3,5× Inferenz-Speedup bei 256K; 1M Kontext auf einer 32GB GPU. | Ja (9B; Prinzip auf klein übertragbar). | arXiv:2602.11761. |
| **REFORM (KV-Cache)** | KV-Cache pro Chunk komprimiert; nur nötige Tokens bei Bedarf neu berechnet. | ~52% (RULER), ~34% (BABILong) bei 1M; ~30% schnellere Inferenz; ~5% weniger Peak-Memory. | Ja (inference-time). | arXiv:2506.01215. |
| **RLM (Recursive LM)** | Kontext als „Umgebung“; REPL mit Code-Tools; Fixed-Window (z. B. 4K) für 1M+ Tokens. | +20–30% auf Long-Context-Tasks vs. Compaction. | Ja (euer RLM-Stub ausbauen). | arXiv:2505.07897. |

## 3.3 Test-Time Compute & Self-Repair

| Ansatz | Für Laien | Was es bringt | Hält klein? | Quelle |
|--------|-----------|----------------|-------------|--------|
| **S*** | Mehrere Kandidaten generieren; „differenzierende“ Test-Inputs erzeugen, die Kandidaten trennen; in Sandbox auswerten und besten wählen. | 3B mit S* übertrifft GPT-4o-mini; GPT-4o-mini + S* schlägt o1-preview um 3,7% auf LiveCodeBench. | Ja (Modellgröße unverändert). | arXiv:2502.14382; EMNLP 2025. |
| **Self-Repair** | Modell generiert Code → läuft in Sandbox → bei Fehler erneuter Generationsschritt („Fix“). | +17–53% auf SWE-bench Verified; AuPair schlägt best-of-N und plain self-repair. | Ja (nur Inferenz-Loop). | Self-Improving Agent; InspectCoder; AuPair ICML 2025. |
| **PoT / LTPO** | PoT: transiente Gewichts-Updates (LoRA/GRPO) zur Laufzeit. LTPO: parameter-frei, optimiert „Thought“-Vektoren pro Aufgabe. | PoT: +14,38 Punkte LiveCodeBench V6 bei 4B. LTPO: robust auf AIME. | Ja (LTPO ohne Extra-Parameter). | PoT; LTPO ICLR 2026. |
| **Agent Distillation** | Großes Agent-Verhalten (Reasoning + Tools) in kleines Modell distilliert. | 0,5B/1,5B/3B match next-tier 1,5B/3B/7B CoT-distilliert auf 8 Reasoning-Tasks. | Ja (0,5B–3B). | arXiv:2505.17612 (NeurIPS 2025). |

## 3.4 Curriculum & RL-Varianten

| Ansatz | Für Laien | Was es bringt | Hält klein? | Quelle |
|--------|-----------|----------------|-------------|--------|
| **Self-Evolving Curriculum (SEC)** | Curriculum wird von einer Bandit-Policy gewählt; RL-Advantage als Lernsignal. | Bessere Generalisierung auf härtere, OOD-Probleme (Planning, Math). | Ja. | arXiv:2505.14970. |
| **Test-Time Curriculum (TTC-RL)** | Zur Trainingszeit wird automatisch task-relevante Daten aus großem Pool für RL ausgewählt. | ~1,8× auf AIME25, ~2,1× auf CodeElo; pass@1 nahe pass@8-Ceiling. | Ja. | arXiv:2510.04786. |
| **FastCuRL** | Curriculum-RL mit gestaffeltem Kontext-Scaling. | 49,6% AIME 2024 mit ~50% weniger Schritten. | Ja. | arXiv:2510.26336 / 2503.17287. |
| **Tina (1,5B)** | LoRA während RL; zweiphasiges RL (Math dann Code). | 43,33% Pass@1 AIME24; ~9$ Post-Training (~260× Kostenreduktion). | Ja (1,5B). | arXiv:2504.15777. |

---

# TEIL 4: ZUSAMMENFASSUNG & PRIORISIERUNG

## Was ihr bereits habt (Codebase)

- **Architektur:** BLT, Mamba-2-Hybrid, LEAM++, BitNet, L-MTP, QK-Norm, Value Residual.
- **Training:** NorMuon, WSD, Stage-LRs, EMA, CODA, CodeDenoise, RegMix, dreistufige Pipeline.
- **Post-Training:** SFT (ChatML, Assistant-Only), GRPO, CodeRL+, Execution-Reward.
- **Inferenz:** S* (Kandidaten + Sandbox), LEAM Constrainer; Stubs: PoT, AB-MCTS, DaJ, RLM.

## Hohe Priorität (neu oder erweitern)

| Maßnahme | Für Laien | Erwarteter Nutzen | Hält klein? |
|----------|-----------|-------------------|-------------|
| **MoA in Attention** | Sparse-Patterns pro Head/Layer; kein Re-Training. | 8K→32K+ Kontext, bessere Retrieval. | Ja. |
| **2-GRPO oder AERO** | Weniger Rollouts pro Prompt; gleiche oder bessere RL-Qualität. | 70%+ kürzeres RL-Training. | Ja. |
| **S* mit distinguishing inputs** | Tests, bei denen sich Kandidaten unterscheiden, nicht nur Pass/Fail. | Stärkere Selektion, bessere pass@1. | Ja. |
| **SelfCodeAlign-Style Instruction** | Konzepte aus Code → Tasks → Sandbox-Filter → nur grüne Paare. | Höhere Instruction-Qualität ohne mehr Parameter. | Ja. |
| **RLM-Loop vollständig** | REPL: Code generieren → ausführen → Ergebnis in Kontext → Iteration. | 1M+ Kontext mit Fixed-Window. | Ja. |
| **Speculative Decoding** | 15–30M Draft für 300M Hauptmodell; 2–4× schnellere Decode. | Deutlich schnellere Inferenz. | Ja. |
| **LongCodeBench-Eval** | Eval bei 32K/128K; Repair Success Rate. | Messbarer Long-Context- und Repair-Erfolg. | Ja. |

## Mittlere Priorität

| Maßnahme | Für Laien | Erwarteter Nutzen | Hält klein? |
|----------|-----------|-------------------|-------------|
| **ORPO oder SimPO** | SFT + Preference in einer Phase; kein Reference-Modell. | Bessere Alignment ohne Extra-Phase. | Ja. |
| **Genetic Instruct / Auto Evol-Instruct** | Evolutions-Algorithmen für Instruction-Daten. | Skalierbare, hochwertige Instruction-Daten. | Ja. |
| **KV-Cache-Optimierung** | Cache komprimieren/prunen. | Längere Kontexte bei gleichem VRAM. | Ja. |
| **BitNet Median-Skalierung** | Robustere Skalierung für kleine Modelle (weniger Outlier-anfällig). | Stabileres Training bei 100–300M. | Ja. |
| **IA2 (ICL Activation Alignment)** | Aktivierungen vor SFT an ICL anpassen. | Besseres Instruction-Following. | Ja. |

## Optional (bei Streckung auf 300M)

| Maßnahme | Für Laien | Erwarteter Nutzen | Hält klein? |
|----------|-----------|-------------------|-------------|
| **MASA / MoHD** | Parameter-Sharing in Attention; dynamische Hidden-Dimensionen. | Mehr Kapazität bei gleicher Parametergröße. | Ja. |
| **MoE (sub-300M)** | Nur wenige Experten pro Token aktiv. | Schnellere Inferenz, weniger KV-Cache. | Ja (aktive Params <300M). |
| **PoT oder LTPO** | Laufzeit-Updates oder Thought-Optimization. | Bessere Reasoning bei Test-Time. | Ja. |
| **PRISM** | Strukturierte Schemata für Long-Range-Reasoning mit kurzem Kontext. | Weniger Kontext nötig für lange Abhängigkeiten. | Ja. |

---

## Quellen (Auswahl 2025–2026)

- **Small Code:** SmallCoder 303M, SmolLM2 (2502.02737), Mify-Coder (2512.23747), Maincoder-1B, Assessing SLMs for Code (2507.03160).
- **Instruction/Alignment:** IA2 (2509.22621), ORPO, SimPO (2405.14734), SelfCodeAlign (NeurIPS 2024), RLVR (2507.02833), SFT Guide (2412.13337).
- **pass@k / RL:** SvS (2508.14029), PKPO (2505.15201), ACECODER (2502.01718), RLEF (2410.02089), 2-GRPO (2510.00977), AERO (2602.14338), PODS (2504.13818).
- **Inferenz:** Speculative (SpecDec++, EAGLE-3, SLEM, TLI, FLy), Cascade, KV-Cache (SmallKV, KeepKV, KVCrush), Early Exit, Quantization.
- **Long-Context:** MoA (2406.14909), Infini-Attention SLMs (2512.23862), LongMamba (2504.16053), QMambaExtend, MiniCPM-SALA (2602.11761), REFORM (2506.01215), RLM (2505.07897).
- **Daten:** IMU-1 (2602.02522), RegMix (ICLR 2025), PreSelect, DataDecide, Curriculum, Dedup, Genetic Instruct, Auto Evol-Instruct.
- **Test-Time:** S* (2502.14382), Self-Repair, AuPair (2502.18487), PoT, LTPO (ICLR 2026), Agent Distillation (2505.17612).
- **Architektur:** BitNet b1.58 (JMLR 2025), Mamba-2-Hybrid, MoE (FLAME-MoE), Parameter Sharing, Deep-Thin (MobileLLM), NorMuon (2510.05491).

---

*Dieses Dokument bündelt die Ergebnisse aus mehreren parallelen Subagent-Recherchen (2025–2026). Alle Angaben „Was es bringt“ sind Schätzungen aus Papers/Studien; konkrete Zahlen hängen von Modell, Daten und Setup ab. „Hält klein“ bezieht sich auf das Ziel, unter 300M Parameter zu bleiben bzw. Compute/VRAM zu reduzieren.*
