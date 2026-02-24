---
name: Instruction SFT + RL Pipeline
overview: Erweiterung der bestehenden 150M-Code-Pipeline um Instruction-SFT (Chat-Format, Loss-Masking, Evol-Instruct-Daten) und execution-basiertes RL (GRPO/PPO, Sandbox-Reward), sodass das Modell auf Anweisungen wie „Schreibe einen Terminal-Taschenrechner“ reagiert und unter 250M Parametern bleibt.
todos: []
isProject: false
---

# Instruction SFT + RL für 150–250M Code-Modell

## Ziel

- **Bestehender Stack:** Pre-Training (bereits in [training/train.py](training/train.py)), Deep-Thin-Architektur ([model/config.py](model/config.py)), ~150M Parameter.
- **Neu:** (1) **Instruction SFT** mit Chat-Format (z. B. ChatML), Loss nur auf Assistant-Token, synthetische Daten (Evol-Instruct/Teacher). (2) **Execution-basiertes RL** (Sandbox, Reward aus Lauf/Tests), GRPO oder PPO. Modell bleibt **≤250M** Parameter und soll instruktionsfähig sein („Schreibe einen Python-Taschenrechner im Terminal“ → nutzbarer Code).

## Abhängigkeiten im bestehenden Code

- **Loss-Masking:** In [model/gpt.py](model/gpt.py) (Zeilen 210–213) wird `labels` mit `ignore_index=-100` in `F.cross_entropy` verwendet. Alles, was im Label-Tensor `-100` ist, wird ignoriert. Für SFT reicht es also, `input_ids` und `labels` so zu bauen, dass nur die **Assistant-Token** echte Label-Werte haben; System/User und Chat-Steuer-Token werden auf `-100` gesetzt.
- **Sandbox:** [evaluation/eval_repair.py](evaluation/eval_repair.py) bietet bereits `run_tests_in_sandbox(code, tests)` und `run_repair_attempt`. Diese Logik kann für RL-Rewards (Syntax/Runtime/Tests) wiederverwendet werden.
- **SFT-Trajektorien:** [post_training/sft_trajectories.py](post_training/sft_trajectories.py) lädt bereits User/Assistant-Trajektorien; es fehlen ChatML-Format, Tokenizer-Integration und explizites **Assistant-Only-Labels** für den Pretraining-Trainer.

---

## 1. Chat-Format und Tokenizer

**Anforderung:** Einheitliches Chat-Format (z. B. ChatML), damit das Modell klar System/User/Assistant trennt. Kleine Modelle sind format-empfindlich; ein festes Template reduziert Rauschen.

**Vorgehen:**

- **Special Tokens:** Im aktuellen Setup ist `vocab_size=16384` in [model/config.py](model/config.py); `pad/bos/eos` sind 0,1,2. Für ChatML braucht man z. B. `<|im_start|>`, `<|im_end|>`, `<|user|>`, `<|assistant|>`, `<|system|>`. Option A: Diese als normale BPE-Subwörter trainieren (Tokenizer-Training mit Beispiel-Chats). Option B: Zusätzliche Special-Token-IDs (z. B. 16384–16389) und Tokenizer erweitern; dann `vocab_size` und Embedding in Config anpassen (bleibt unter 250M wenn nur wenige hinzugefügt werden).
- **Konkret:** Neue Config `data/chat_template.yaml` oder Eintrag in [data/config_data.yaml](data/config_data.yaml): Template-String (ChatML), Reihenfolge System → User → Assistant. Hilfsfunktion `data/chat_format.py`: `format_chat(system, user, assistant) -> str` und `parse_chat_to_message_spans(text)` (optional, für Label-Masking).
- **Tokenizer:** Entweder bestehenden BPE ([data/tokenizer_train.py](data/tokenizer_train.py)) um Beispiel-Chats erweitern und neu trainieren, oder ein HF-Tokenizer mit `add_special_tokens` nutzen. Entscheidung: Wenn du beim aktuellen 16k-BPE bleibst, reicht ein festes ChatML-String, der als normale Tokens encodiert wird; dann keine Vocab-Änderung nötig.

---

## 2. Instruction-SFT-Daten (Evol-Instruct / Teacher)

**Anforderung:** 100k–500k hochwertige Instruction–Code-Paare, möglichst mit Evol-Instruct (einfache Prompts iterativ verkomplizieren) und Teacher-generierten Antworten (z. B. DeepSeek-Coder, Qwen2.5-Coder, oder API).

**Vorgehen:**

- **Datensatz-Format:** JSONL, eine Zeile pro Beispiel. Pro Zeile: `{"instruction": "...", "output": "..."}` oder `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`. Optional: `system` für System-Prompt.
- **Generierung:** Neues Modul `data/instruction_data.py` (oder Unterordner `data/instruction/`):
  - **Evol-Instruct-Logik:** Funktion `evolve_instruction(instruction: str, level: str) -> str` (z. B. „Schreibe eine Addition“ → „Schreibe einen Terminal-Taschenrechner mit Fehlerbehandlung“). Entweder heuristisch (Templates) oder per Aufruf eines Teacher-Modells/API.
  - **Teacher-Antworten:** Funktion `generate_teacher_response(instruction: str, api_or_model) -> str`. Integration mit HF Pipeline oder OpenAI-ähnlicher API für Code-Completion; Ausgabe = nur Code oder Code+Erklärung je nach gewünschtem Format.
  - **Skript:** `data/generate_instruction_data.py`: Liest Seed-Instructions (Datei oder einfache Liste), wendet Evol-Instruct an, ruft Teacher auf, schreibt ChatML-formatierte Zeilen nach `data/processed/instruction_sft.jsonl`.
- **Alternativ:** Nutzung öffentlicher Datensätze (z. B. evol-instruct-code, CodeAlpaca, oder gefilterte Code-Chat-Daten von HF), Konvertierung in das gleiche JSONL-Format und ChatML-String.

---

## 3. SFT-Dataloader und Assistant-Only-Loss

**Anforderung:** Dataloader, der Instruction-JSONL lädt, in ChatML-String umwandelt, tokenisiert und **Labels** so setzt, dass nur Assistant-Token für den Loss zählen (Rest `-100`).

**Vorgehen:**

- **Neues Modul:** `data/sft_dataloader.py` (oder Erweiterung in [data/dataloader.py](data/dataloader.py) mit `mode="sft"`).
  - Eingabe: Pfad zu `instruction_sft.jsonl`, Tokenizer, `chat_format` (ChatML-Template), `max_seq_len`, `batch_size`.
  - Pro Zeile: `messages` → ChatML-String bauen → tokenisieren → `input_ids`.
  - **Label-Masking:** Für jede Position: Wenn das Token zu einem **Assistant**-Segment gehört, Label = Token-ID; sonst Label = `-100`. Dafür muss die Grenze „Start Assistant“ / „End Assistant“ im tokenisierten Verlauf bekannt sein (entweder über Byte-Offsets nach dem Tokenizer oder über feste ChatML-Delimiters und Token-ID-Ranges). Ausgabe: `{"input_ids": Tensor, "labels": Tensor}`.
- **Batch:** Padding auf `max_seq_len`, `pad_token_id`; in `labels` Padding-Positionen ebenfalls `-100`.
- **Integration:** [training/train.py](training/train.py) verwendet aktuell `get_training_dataloader` und erwartet Batches mit `input_ids` (und optional `labels`). Für SFT: Entweder neuer Entrypoint `training/sft_train.py`, der nur den SFT-Dataloader nutzt und das gleiche `train_step` (mit `labels` aus dem Dataloader) aufruft, oder `train.py` um ein Argument `--mode sft` und einen zweiten Dataloader-Pfad erweitern. Empfehlung: eigener `sft_train.py`, der ein Pre-Train-Checkpoint lädt und nur SFT durchführt (weniger Risiko, Pretraining-Logik zu brechen).

---

## 4. SFT-Training-Skript

**Anforderung:** Skript, das Pre-Train-Checkpoint lädt, nur auf Instruction-Daten mit Assistant-Only-Loss trainiert, Checkpoints speichert.

**Vorgehen:**

- **Neue Datei:** `training/sft_train.py`.
  - Argumente: `--checkpoint` (Pre-Train-Checkpoint, z. B. `checkpoints/final.pt`), `--instruction_data` (Pfad zu `instruction_sft.jsonl`), `--chat_template` (Pfad zu YAML oder Inline ChatML), `--output_dir`, `--batch_size`, `--max_steps`, `--lr`, `--max_seq_len`, `--tokenizer_path`.
  - Modell: Wie in [training/train.py](training/train.py) aus Checkpoint laden (gleiche `ModelConfig` / `CodeGPTLMHeadModel`).
  - Dataloader: Aus `data/sft_dataloader.py` mit Assistant-Only-Labels.
  - Optimizer: AdamW (oder gleiche Gruppen wie Pretraining); LR z. B. 1e-5 bis 5e-5.
  - Loop: Wie `train_step` in train.py; keine EMA/L-MTP nötig, nur nächster-Token-Loss. Optional: Gradient Clipping, Logging, Evaluation auf einem kleinen Holdout.
  - Speichern: `sft_checkpoints/` mit `final_sft.pt` und optional safetensors.
- **Config:** `training/config_sft.yaml` für SFT-spezifische Defaults (batch_size, lr, max_steps, max_seq_len, chat_template_path).

---

## 5. Execution-basiertes RL (GRPO / PPO)

**Anforderung:** Nach SFT das Modell mit RL verfeinern: Prompt (Instruction) → Modell generiert N Code-Kandidaten → Sandbox führt Code aus / führt Tests aus → Reward (Syntax ok / Lauf ok / Tests bestanden) → Policy-Update (GRPO oder PPO).

**Vorgehen:**

- **Reward-Funktion:** Modul `post_training/execution_reward.py` (oder Erweiterung von [post_training/coderl_plus.py](post_training/coderl_plus.py)).
  - Eingabe: `code: str`, optional `tests: list[str]` (wenn aus Aufgabe bekannt).
  - Logik: (1) Syntax-Check (z. B. `ast.parse`); bei Fehler return -1.0. (2) Optional: Code in Sandbox ausführen ([evaluation/eval_repair.py](evaluation/eval_repair.py) `run_tests_in_sandbox`); bei Timeout/Exception return -0.5; bei Erfolg ohne Tests return 0.0 oder kleiner Bonus. (3) Wenn Tests vorhanden: `run_tests_in_sandbox(code, tests)`; passed → +1.0, sonst -0.5. Ausgabe: ein Float pro Sample.
- **RL-Daten:** Gleiche Instruction-Liste wie SFT (oder Teilmenge); pro Instruction N Generierungen (z. B. N=4 oder 8).
- **GRPO:** Group Relative Policy Optimization: Pro Prompt eine Gruppe von K Samples; Reward pro Sample; Normalisierung innerhalb der Gruppe (z. B. advantage = reward - mean(rewards)). Loss = -log_prob * advantage, nur für die generierten Token. Implementierung: Entweder in PyTorch von Hand (generate, dann loss über alle generierten Token mit maskierten log_probs und advantage), oder Anbindung an z. B. `trl` (GRPO) / `openrlhf` falls gewünscht.
- **Neues Skript:** `training/rl_train.py` (oder `post_training/grpo_train.py`): Lädt SFT-Checkpoint, lädt Instruction-Daten, pro Step: Batch von Prompts, für jeden Prompt N Generierungen, Reward berechnen, GRPO-Loss, Backward, Optimizer-Step. PPO-Alternative: Klassischer PPO-Update mit Value-Head (dann kleines Value-Head-Modul am LM); bei 150M oft GRPO einfacher ohne Value-Head.
- **Sandbox-Sicherheit:** Für RL nur vertrauenswürdige Aufgaben nutzen; Sandbox mit Timeout und ohne Netzwerk (wie in eval_repair) beibehalten.

---

## 6. Pipeline-Reihenfolge und Parameter-Obergrenze

- **Reihenfolge:** Pre-Training (bestehend) → SFT (`sft_train.py` auf Instruction-Daten) → RL (`rl_train.py` mit execution reward). Inferenz dann mit demselben Chat-Format (User-Nachricht eingeben, Modell antwortet mit Assistant-Teil).
- **Parameter:** Modellgröße bleibt durch [model/config.py](model/config.py) und [training/config_train.yaml](training/config_train.yaml) gesteuert. Aktuell ~150M (44L, d_model=384). Für bis 250M: z. B. `n_layer=48` und/oder `d_model=448` innerhalb der gleichen Architektur; in einer gemeinsamen Config (z. B. `config_train.yaml` oder neue `config_instruction.yaml`) explizit dokumentieren: „Max 250M Parameter“.

---

## 7. Konfiguration und Repo-Struktur

- **Neue/angepasste Dateien (Übersicht):**
  - `data/chat_format.py` – ChatML-Formatierung und Message-Spans für Label-Masking.
  - `data/instruction_data.py` oder `data/instruction/` – Evol-Instruct + Teacher-Aufruf (oder Adapter für HF-Datensätze).
  - `data/generate_instruction_data.py` – Skript zum Erzeugen von `instruction_sft.jsonl`.
  - `data/sft_dataloader.py` – Instruction-Dataloader mit Assistant-Only-Labels.
  - `training/config_sft.yaml` – SFT-Hyperparameter.
  - `training/sft_train.py` – SFT-Training ab Pre-Train-Checkpoint.
  - `post_training/execution_reward.py` – Reward aus Sandbox/Tests (oder in coderl_plus integrieren).
  - `training/rl_train.py` – GRPO/PPO-Loop mit Execution-Reward.
- **Bestehende Wiederverwendung:** [model/gpt.py](model/gpt.py) (labels, ignore_index=-100), [evaluation/eval_repair.py](evaluation/eval_repair.py) (Sandbox), [training/train.py](training/train.py) (Optimizer/Step-Logik als Referenz).

---

## 8. Kurz: Was du am Ende hast

- **Pre-Train** (wie heute): Code-only, 3-Stufen-Daten, Stage-LRs, EMA → `final.pt`.
- **SFT:** `python training/sft_train.py --checkpoint checkpoints/final.pt --instruction_data data/processed/instruction_sft.jsonl ...` → Modell spricht im Chat-Format und beachtet Anweisungen; Loss nur auf Assistant.
- **RL:** `python training/rl_train.py --checkpoint sft_checkpoints/final_sft.pt ...` → Modell wird auf funktionale Korrektheit (Sandbox/Tests) optimiert.
- **Inferenz:** Gleiches Chat-Format: User-Prompt z. B. „Schreibe einen Python-Taschenrechner im Terminal“ → Modell generiert Assistant-Antwort (Code). Dafür muss `inference/run_torch.py` (oder ein neues `inference/run_chat.py`) so erweitert werden, dass ein ChatML-Prompt gebaut wird und nur die Assistant-Antwort ausgegeben wird.

Damit ist die Pipeline wissenschaftlich auf dem Stand von Evol-Instruct, Loss-Masking, execution-based RL (GRPO/PPO) und bleibt unter 250M Parameter mit deinem bestehenden Deep-Thin-Code-Modell.