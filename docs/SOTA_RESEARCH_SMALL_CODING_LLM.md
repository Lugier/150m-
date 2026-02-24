# SOTA-Recherche: Kleine Coding-LLMs (≤300M Parameter), Instruction & Long Context

Tiefe, investigative Recherche zu wissenschaftlich belegten Ansätzen für ein High-Class-SOTA-Coding-LLM mit maximal ~300M Parametern, starker Instruction-Performance und Long-Context-Fähigkeit. Stand: Forschung 2024/2025.

---

## 1. Was ihr bereits drin habt (Bewertung & Evidenz)

| Element | Evidenz / Forschung | Empfehlung |
|--------|----------------------|------------|
| **Deep-Thin (d_model 384, n_layer 44)** | MobileLLM (2024): Deep-Thin + Embedding-Sharing + GQA bringt 2.7–4.3 % Genauigkeitsverbesserung bei 125M/350M; deutlich effizienter als breite, flache Netze. | Beibehalten; für <300M weiterhin SOTA-tauglich. |
| **L-MTP (mtp_n=2)** | Gloeckle et al. (ICML 2024): 4-Token-Prediction bei 13B: +12 % HumanEval, +17 % MBPP; Effekte bei Code besonders stark. Inferenz bis 3× schneller durch Self-Speculative Decoding. | mtp_n=2 oder 4 beibehalten; bei 250M eher 2–4 für Regularisierung ohne Overhead. |
| **Mamba-2-Hybrid** | Samba/Jamba (2024/25): Linearer Kontext, weniger KV-Cache; Jamba 256K bei 10× weniger KV-Speicher. Mamba-Length-Generalization: Mamba Modulation (2025) adressiert OOD bei längerem Kontext. | Beibehalten für Long Context; ggf. Mamba-Modulation (Spectrum Scaling) für bessere Längen-Generalisation prüfen. |
| **QK-Norm** | Stabilisiert Training (Logit-Drift, Attention-Entropy-Collapse); ermöglicht robustere Lernraten. | Beibehalten. |
| **GRPO + Execution Reward** | GRPO = gruppenrelativer Vorteil ohne separaten Critic (DeepSeek R1, Code/Math). CodeRL+ (2024): Execution + Semantics-Alignment; RLEF: Execution Feedback, SOTA bei 8B/70B. | Beibehalten; Semantics nur bei exec_score>0 (Verifiable RL) wie umgesetzt. |
| **SFT: EOS bei Truncation** | Ohne EOS am Sequenzende lernt das Modell Abbruchverhalten; mit EOS klares Stopp-Signal. | Beibehalten. |
| **Vocab-Resize (Embed/LM-Head)** | Ohne manuelles Kopieren gehen Pre-Train-Gewichte bei Vocab-Erweiterung verloren (strict=False überspringt Shape-Mismatch). | Beibehalten. |
| **Vocab 16448 (8/64-Alignment)** | Tensor Cores (Ampere/Ada) effizient bei Vielfachen von 8 (besser 64); Vermeidung von Padding/ineffizienten Kerneln. | Beibehalten. |
| **NorMuon** | NorMuon (2024): ~21 % bessere Training-Effizienz vs. AdamW, ~11 % vs. Muon bei 1.1B; neuronweise Normalisierung behebt Muon-Nachteile. | Beibehalten. |

---

## 2. Bewährte SOTA-Bausteine für ≤300M Code-LLMs

### 2.1 Curriculum Learning (Multi-Stage)

- **SmallCoder (303M)**: 4 Stufen – (1) Linguistic Base, (2) Code Specialization, (3) Math & Knowledge, (4) SFT mit EOS-fixierten Daten. Gesamt ~29.8B Tokens.
- **Curriculum-Guided Layer Scaling (CGLS)**: Steigende Daten-Schwierigkeit mit wachsendem Modell (z. B. 100M→1.2B); bessere Wissens- und Reasoning-Tasks.
- **Für euch**: Pre-Train → Code-spezialisiert → SFT (Instruction) ist konsistent mit SmallCoder; optional explizite „Code-Difficulty“-Stufen oder Daten-Mix nach Schwierigkeit.

### 2.2 Instruction & SFT

- **Assistant-Only Loss**: Nur Assistant-Tokens mit Loss; User/System mit `labels=-100`. Standard für Chat/Instruction.
- **Data Quality**: GRAPE (Response-Prob unter Zielmodell), Superfiltering (Weak-to-Strong), CLEAR – „best data = those that fit“; weniger, aber passende Daten oft besser als viel ungefiltert.
- **Mify-Coder (2.5B)**: CPT + SFT, kuratierte + synthetische Daten, LLM-basiertes Quality-Filtering; kompakte Modelle erreichen frontier-nahe Code-Intelligenz.

### 2.3 RL & Execution

- **CodeRL+**: Execution + Semantics-Alignment; Semantics nur bei lauffähigem Code (Verifiable RL) – verhindert Reward Hacking; ~4.6 % pass@1, +15.5 % Code-Reasoning.
- **RLEF**: Execution Feedback, starke Ergebnisse bei 8B/70B, weniger Samples nötig.
- **GRPO**: Kein Critic, gruppenrelativer Vorteil; gut für Code/Math, weniger VRAM.

### 2.4 Architektur & Effizienz

- **Mamba-Hybrid**: Samba 3.7× Throughput bei 128K Prompts; Jamba 256K, 10× weniger KV. Für Long Context bei kleinem Modell fast Pflicht.
- **BitNet b1.58**: 1.58-bit (ternär) von Anfang an; Microsoft 2B auf 4T Tokens; auch für 100K–48M gezeigt. Option für noch kleinere Deployment-Kosten.
- **Speculative Decoding**: Mit MTP (n>1) Self-Speculative möglich; 3× Speedup in der Literatur. Euer MTP-Head direkt für Draft-Verification nutzbar.

### 2.5 Long Context

- **Mamba Length Generalization**: Mamba Modulation (2025) verbessert Generalisation auf längere Kontexte durch Spectrum Scaling der State-Transition-Matrix.
- **Samba**: 4K Pre-Train → 1M Zero-Shot Perplexity, 256K Fine-Tune mit guter Recall. Klare Mehrstufen-Strategie für Kontextlänge.

---

## 3. Konkrete Erweiterungen / Optionen (nach Evidenz)

1. **Curriculum im Pre-Train**: Stufenweise von allgemeinem Code → schwierigerem Code / Math (evtl. mit Difficulty-Scorer wie in Curriculum-Code-LLM-Arbeiten).
2. **Data Filtering für SFT**: GRAPE- oder Weak-to-Strong-Filterung auf Instruction-Daten; nur Beispiele die „zum Modell passen“.
3. **MTP auf 4 erhöhen** (wenn VRAM/Compute erlauben): Größere Code-Gains und Self-Speculative-Speedup; bei 250M oft mtp_n=2 als Kompromiss.
4. **Mamba Modulation** (wenn Mamba-Hybrid): Paper 2025 zu Length Generalization prüfen und ggf. in eurer Mamba-Integration übernehmen.
5. **BitNet-Option** (config bereits vorhanden): Für extrem kleines Deployment; Training von Scratch in W1.58A8.
6. **Self-Speculative Decoding in Inferenz**: MTP-Head als Draft-Modell nutzen, Target = Hauptmodell; 2–3× Speedup ohne zweites Modell.

---

## 4. Was ihr vermeiden solltet

- **Keine Fallbacks/Dummies** für Token-IDs oder Modell-Config: Single Source of Truth (Tokenizer/ModelConfig); fehlende Werte → klarer Fehler.
- **Keine Semantics-Rewards bei exec_score=0**: Verifiable RL wie umgesetzt beibehalten.
- **Kein hartes Truncation ohne EOS**: Immer EOS am Ende bei Abschneiden setzen.
- **Vocab-Größe nicht beliebig**: 8/64-Alignment für Tensor Cores einhalten (z. B. 16448).

---

## 5. Quellen (Auswahl)

- SmallCoder 303M, Mify-Coder (arXiv 2512.23747), Mify training pipeline.
- Better & Faster LLMs via Multi-Token Prediction (Gloeckle et al., ICML 2024).
- Samba (arXiv 2406.07522), Jamba (Hybrid Transformer-Mamba), Mamba Modulation (arXiv 2509.19633).
- CodeRL+, RLEF (arXiv 2510.18471, 2410.02089).
- GRPO (NeMo RL, RLinf, DeepSeek R1).
- NorMuon (OpenReview/arXiv 2510.05491), QK-Norm (Emergent Mind, Ross Taylor).
- BitNet b1.58 (Microsoft), MobileLLM (Deep-Thin).
- Curriculum: CGLS (arXiv 2506.11389), Curriculum Code LLM (ACL 2024 Workshop).
- Instruction Data: GRAPE, Superfiltering, CLEAR (arXiv 2502.04194, ACL 2024).

---

*Dokument dient als Living Reference für SOTA-Entscheidungen im Projekt; bei neuen Papers relevante Abschnitte aktualisieren.*
