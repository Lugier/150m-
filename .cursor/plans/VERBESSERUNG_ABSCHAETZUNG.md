# Was wir verbessert haben – Abschätzung

**Kontext:** Umgesetzte Punkte der SOTA-300M-Gesamtplanung (hohe + ausgewählte mittlere Priorität).  
Zwei Teile: **(1) technisch einfach verständlich**, **(2) inhaltlich quantifiziert**.

---

## Teil 1: Technisch einfach verständlich

### MoA (Mixture of Sparse Attention)

- **Was es ist:** In der Attention schaut nicht jeder Token auf alle anderen (teuer, Speicher wächst quadratisch). MoA nutzt pro Head feste Muster: z. B. nur ein Fenster der letzten 512 Tokens, die ersten 8 („global“) oder jeden zweiten (dilated). So bleibt das Modell klein, kann aber längere Kontexte „effektiv“ nutzen.
- **Was es bringt:** Längerer nutzbarer Kontext (z. B. 8K→32K+) ohne Full-Attention und ohne Re-Training; weniger Speicher pro Schritt.

### 2-GRPO und AERO (RL-Training)

- **2-GRPO:** Statt 16 Code-Kandidaten pro Aufgabe werden nur 2 erzeugt und verglichen. Weniger Rollouts = deutlich weniger Rechenzeit pro Schritt, bei nahezu gleicher Lernqualität (kontrastives Lernen mit 2 Antworten reicht oft).
- **AERO (Zero-Advantage-Skip):** Wenn alle Kandidaten dasselbe Ergebnis liefern (alle richtig oder alle falsch), gibt es kein sinnvolles Lernsignal. Statt trotzdem einen Optimizer-Schritt zu machen (Verschwendung), wird der Schritt übersprungen. So wird keine Compute für nutzlose Updates verbraucht.

### S* mit distinguishing inputs

- **Was es ist:** Beim S*-Verfahren werden mehrere Code-Kandidaten erzeugt und in der Sandbox getestet. „Distinguishing inputs“ sind zusätzliche Tests, bei denen sich die Kandidaten unterscheiden (nicht nur Pass/Fail), sodass man den wirklich besten zuverlässiger auswählen kann.
- **Was es bringt:** Bessere Auswahl des besten Kandidaten → höhere pass@1 bei gleicher Kandidatenzahl; weniger Zufall bei gleichen Test-Ergebnissen.

### SelfCodeAlign-Style Instruction-Daten

- **Was es ist:** Aus bestehendem Code werden Konzepte (z. B. Funktionsnamen, Docstrings) gezogen und daraus neue Aufgaben formuliert. Der Teacher erzeugt mehrere Antworten; nur solche, die in der Sandbox alle Tests bestehen, kommen ins SFT-Dataset.
- **Was es bringt:** Höhere Qualität der Instruction-Daten (nur „grüne“ Beispiele), bessere Instruction-Following und Code-Qualität ohne mehr Parameter.

### RLM REPL-Loop (vollständig)

- **Was es ist:** Das Modell generiert Code → der Code wird in einer sicheren Umgebung ausgeführt → die Ausgabe (stdout/Fehler) wird wieder in den Kontext geschrieben → das Modell generiert den nächsten Schritt. So kann über viele Schritte hinweg mit langem „effektivem“ Kontext gearbeitet werden (Fixed-Window, z. B. 4K pro Schritt).
- **Was es bringt:** Nutzung von 1M+ Token Kontext (z. B. großes Repo) durch wiederholte kurze Fenster; kein einzelner Riesen-Kontext nötig.

### Speculative Decoding

- **Was es ist:** Ein kleines Draft-Modell (z. B. 15–30M Parameter) schlägt mehrere nächste Tokens vor; das große Modell (300M) prüft sie in einem Durchlauf und akzeptiert den längsten korrekten Präfix. So werden pro Runde mehrere Tokens „abgehakt“, obwohl das große Modell nur einmal läuft.
- **Was es bringt:** Deutlich schnellere Textausgabe bei gleicher Qualität (lossless), besonders spürbar bei längeren Antworten.

### LongCodeBench-Eval und Repair Success Rate

- **Was es ist:** Evaluation bei festen Kontextlängen (32K, 128K) mit pass@1 und „Repair Success Rate“: Wie oft schafft es das Modell, nach einem ersten fehlgeschlagenen Lauf den Code in wenigen Reparatur-Versuchen zu korrigieren?
- **Was es bringt:** Messbare Fortschritte bei Long-Context und Self-Repair; klare Metriken für 32K/128K und Reparaturfähigkeit.

### BitNet Median-Skalierung

- **Was es ist:** Bei BitNet (ternäre Gewichte -1, 0, +1) wird pro Schicht ein Skalierungsfaktor β benötigt. Statt dem Mittel der Beträge (anfällig für Ausreißer) wird der **Median** verwendet.
- **Was es bringt:** Stabileres Training bei kleinen Modellen (100–300M), weniger Empfindlichkeit gegenüber einzelnen großen Gewichten.

### Ohne Fallback (durchgängig)

- **Was es ist:** Keine stillen Ersatzlösungen: Fehlt der Tokenizer, Checkpoint-Config, RL-Daten oder Dataloader, bricht das Programm sofort mit klarer Fehlermeldung ab (kein Dummy-Tokenizer, kein Platzhalter-Batch).
- **Was es bringt:** Reproduzierbarkeit, keine versteckten Qualitätsverluste durch Fallback-Daten oder -Configs; Fehler treten sofort und nachvollziehbar auf.

---

## Teil 2: Inhaltlich quantifizierte Abschätzung

Angaben in **[...]** beziehen sich auf die in der Gesamtplanung zitierten Papers/Studien; unsere Pipeline kann je nach Daten und Setup etwas abweichen. **„Wir“** = erwarteter Effekt bei Nutzung der neuen Komponenten in eurer 300M-Pipeline.

| Verbesserung | Grobe Quantifizierung (Orientierung) | Quelle / Anmerkung |
|--------------|--------------------------------------|---------------------|
| **MoA** | ~3,9× effektiver Kontext; 1,5–7,1× bessere Retrieval-Genauigkeit bei langem Kontext; 8K→32K+ ohne Full-Attention | arXiv:2406.14909; bei 150M getestet, Prinzip auf 300M übertragbar |
| **2-GRPO** | ~98 % der Performance von 16-GRPO bei 12,5 % der Rollouts; **>70 % kürzere RL-Trainingszeit** | arXiv:2510.00977 („It Takes Two“) |
| **AERO (Zero-Advantage-Skip)** | In Studien: **~48 % weniger Training-Compute**, ~45 % kürzere Wall-Clock bei gleicher/ besserer Performance; bei uns: Reduktion nutzloser Steps (Anteil abhängig von Daten) | arXiv:2602.14338 |
| **S* mit distinguishing inputs** | In Studien: 3B mit S* übertrifft GPT-4o-mini; **+3,7 %** vs. o1 auf LiveCodeBench; bei uns: bessere pass@1 durch schärfere Kandidatenauswahl | arXiv:2502.14382, EMNLP 2025 |
| **SelfCodeAlign-Style** | In Studien: 67,1 % pass@1 HumanEval+ (7B); übertrifft CodeLlama-70B-Instruct; bei uns: **bessere Instruction-Qualität** durch nur test-grüne Paare (Größenordnung wenige bis niedrige zweistellige %-Verbesserung auf pass@1 je nach Baseline) | NeurIPS 2024, BigCode |
| **RLM REPL** | In Studien: **+20–30 %** auf Long-Context-Tasks vs. reine Compaction; bei uns: 1M+ Kontext nutzbar mit Fixed-Window | arXiv:2505.07897 |
| **Speculative Decoding** | **1,5–6,5×** schnellere Decode (setting-abhängig), lossless; bei uns: typisch **2–4×** mit kleinem Draft | Diverse 2025–2026 |
| **LongCodeBench 32K/128K + Repair** | Keine direkte „+X %“-Zahl für euer Modell; **messbare Metriken** für 32K/128K pass@1 und Repair Success Rate; Vergleich vorher/nachher und zwischen Checkpoints möglich | Plan §9; eval_lcb.py |
| **BitNet Median-Skalierung** | Robustere Skalierung für kleine Modelle; **stabileres Training** (weniger Instabilität durch Outlier); keine einzelne Benchmark-Zahl im Plan | JMLR 2025, BitNet b1.58; Plan: „Median-Skalierung für kleine Modelle“ |
| **Ohne Fallback** | **0 %** versteckte Qualitätsverluste durch Dummy-Daten/Config; **100 %** Reproduzierbarkeit bei korrekter Konfiguration; sofortige Fehlererkennung bei fehlenden Ressourcen | PLAN_COMPLIANCE, DESIGN |

---

## Kurz: Was bringt es uns?

- **Schneller:** RL-Training (2-GRPO + AERO) deutlich kürzer und effizienter; Inferenz mit Speculative Decoding 2–4× schneller; weniger verschwendete RL-Steps.
- **Besser:** Länger nutzbarer Kontext (MoA), bessere Kandidatenauswahl (S* distinguishing), höhere Instruction-/Code-Qualität durch SelfCodeAlign-Style-Daten; stabileres BitNet-Training bei kleinen Modellen.
- **Smarter:** RLM REPL für 1M+ Kontext; klare Long-Context- und Repair-Metriken (LongCodeBench 32K/128K, Repair Success Rate); keine stillen Fallbacks, dafür reproduzierbare und saubere Fehlerbehandlung.

Die genauen Zahlen hängen von eurem konkreten Modell, Datenumfang und Hardware ab; die Tabelle gibt eine **Größenordnung** aus der Literatur und dem Plan für die eingebauten Komponenten.
