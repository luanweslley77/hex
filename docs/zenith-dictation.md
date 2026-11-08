# Zenith phone dictation — local fork notes

> **Local note for this fork.** It is not part of upstream Hex and should not be
> proposed upstream as-is. It records what this checkout was used for, why the
> changes exist, and what a future maintainer/LLM needs to know.
>
> **Last updated:** 2026-11-08
> **Companion docs:**
> `~/zenith-audio/docs/phone-dictation.md` (architecture and decisions),
> `~/zenith-audio/bridge/README.md` (bridge usage and protocol).

---

## 1. Why this fork is involved

The machine has no usable PC microphone. The phone is the microphone:

```text
Poco X3 NFC ──ZAP/UDP 59100──► zenith-server ──► PipeWire "Zenith Wireless Microphone"
                                                          │
                                                          ▼
                                              Hex (local ASR, Vulkan)
                                                          │
                                                          ▼
                                              text back to the phone or PC
```

The phone already sends 48 kHz mono PCM16 in 10 ms frames. The missing piece was
**local speech recognition on the PC** and a way to get the text back. Hex was
chosen because it:

- runs the model locally on the RX 6500 XT through Vulkan;
- keeps a model resident across recordings;
- has a supported local API and a TypeScript SDK (`@kitlangton/hex`);
- is MIT licensed and does not accept pull requests, so it must be consumed as
  a black box (this fork is for local builds/experiments, not upstream work).

The target experience: hold a key on the phone, speak, see provisional words
while speaking, and have the final text inserted into the phone's focused field
(an Android IME). PC dictation through Hex's own hotkey remains a separate path.

---

## 2. What was done in this checkout

1. **Updated the source.**
   `git fetch --all --prune` and a fast-forward merge of `upstream/dev` (46
   commits, ending at `62a40669`, "add orukeet to the speech model catalog").
   Local branch `dev` is clean; `origin/dev` (the fork) is 46 commits behind.

2. **Built Hex from source.**
   Rust 1.94 was too old for the new lockfile (`oxc_syntax` needs 1.97), so the
   stable toolchain was updated to **1.99.1**. Build dependencies installed:
   `alsa-lib-devel` and `openblas-devel` (the earlier OpenBLAS symlink and the
   local ALSA prefix workarounds were removed). Build:

   ```bash
   cd ~/hex
   CARGO_BUILD_JOBS=16 cargo build --locked --release
   ```

   The binary is `~/hex/target/release/hex` (**Hex 3.3.0**). It is not installed
   into `~/.local/bin`; use the full path or install it separately.

3. **Imported the pt-BR model.**
   The `-30s` Parakeet fine-tune was converted from `.nemo` to Q8_0 GGUF with
   `transcribe.cpp`'s converter and imported:

   ```bash
   ~/hex/target/release/hex model import \
     ~/.cache/parakeet-spike/models/parakeet-tdt-v3-ptBR-30s/parakeet-tdt-v3-ptBR-30s-Q8_0.gguf
   # imported  custom-7d13a4fb474b2100  Parakeet TDT 0.6B v3 pt-BR (TAGARELA)

   ~/hex/target/release/hex model check custom-7d13a4fb474b2100 --language pt
   # ready  AMD Radeon RX 6500 XT (RADV NAVI24) (vulkan)
   ```

   The id is a SHA-256 prefix of the imported file, so it is **machine-local**.
   The model lives in `~/.local/share/hex/models/custom/`.

4. **Patched `transcribe.cpp` for the Parakeet v3 frame bug (#252).**
   The pinned dependency is `anomalyco/transcribe.cpp` at rev `2778ae4`. The
   patch is local (not upstream) and lives in the Zenith repo:
   `~/zenith-audio/docs/patches/transcribe-cpp-252-compact-last-mel-frame.patch`.

   The checkout is at:
   `~/.cargo/git/checkouts/transcribe.cpp-0d04a17001a6bbfd/2778ae4/`.

   Apply it there, then rebuild:

   ```bash
   cd ~/.cargo/git/checkouts/transcribe.cpp-0d04a17001a6bbfd/2778ae4
   git apply ~/zenith-audio/docs/patches/transcribe-cpp-252-compact-last-mel-frame.patch
   cd ~/hex
   cargo clean
   CARGO_BUILD_JOBS=16 cargo build --locked --release
   ```

   **Why the patch is needed** is in §3.

5. **Built the bridge** in `~/zenith-audio/bridge/` (Bun + `@kitlangton/hex`).
   It starts `hex service --embedded`, keeps the model resident, accepts TCP
   audio, sends live `PARTIAL` frames by re-transcribing the tail, and sends one
   final `RESULT` for the whole utterance. See §4.

6. **Validated end to end.**
   - The bug clips from transcribe.cpp #252 (5.04 s empty, 10.56 s truncated)
     now return full text.
   - A 14.2 s FLEURS pt-BR sample returns the reference exactly.
   - The bridge's live test produced **17 PARTIAL frames** and a final RESULT
     identical to the reference.
   - `bun test` (11 tests) and `tsc --noEmit` are clean.
   - OpenCodeReview was run twice; the high/medium findings were fixed (§6).

---

## 3. The Parakeet v3 frame bug (#252) and the real fix

This is the most important local change and the easiest to lose.

**Symptom.** The stock Parakeet v3 (and the `-30s` fine-tune, same architecture)
returns **empty** or **truncated** text for roughly one clip length in eight.
Reproduced with Hex 3.3.0 and the stock v3 Q8_0:

```text
clip 5.04 s             → (empty)
clip 5.04 s, −1 sample  → "By no possible supposition could this stranger, Fitzroy Simpson, have caused"
clip 10.56 s            → "His bridle is missing, so that Simpson must have put this on."
clip 10.56 s, −1 sample → "... Then, having left the door open behind him, he was leading the horse away over the moor,"
```

**Cause.** The encoder graph is built with **every** mel frame, including the
frontend's final padding frame. NeMo counts all but the last
(`pcm.len() / HOP`). The extra frame joins the attention and can blank or
truncate the transcript. The failure depends on `mel_frames % 8 == 1`.

**Why the obvious fix is not enough.** `pre_encode_stage_lengths()` in
`src/arch/parakeet/model.cpp` is only consumed when `kestrel_length_masking` is
on, and the v3 family does not set it. Changing `len[0]` to `mel_frames - 1`
there is dead code for v3: the extra frame still enters the encoder. Trimming
the decoder input afterwards (`T_valid`) is also insufficient, because the last
valid frame was already influenced by the extra frame through attention/conv.

**The actual fix (validated).** Keep the last mel frame out of the graph:

1. derive `graph_mel_frames = mel_frames - 1`;
2. compact the row-major mel buffer from `[n_mels, mel_frames]` to
   `[n_mels, graph_mel_frames]`;
3. build the encoder graph with `graph_mel_frames`, upload the compacted buffer,
   and pass `graph_mel_frames` to the length masks.

The patch touches `run_one_shot_inner` in `src/arch/parakeet/model.cpp`. It is
the same idea as wavo's `encoder_frames` (wavo PR #90): the model must process
the frames NeMo counts, not the frontend's padding.

**Measured impact after the fix:**

| Model | FLEURS raw | FLEURS without digit sentences | TAGARELA spontaneous |
|---|---|---|---|
| `-30s` fine-tune Q8_0 | 7.60% | 4.05% | 18.44% |
| stock Parakeet v3 Q8_0 | 5.45% | 4.73% | 21.72% |
| Whisper large-v3-turbo Q8_0 | 5.00% | 4.50% | 22.95% |

The `-30s` and Whisper numbers are unchanged by the patch. Stock v3 moved
slightly (5.36→5.45, 21.31→21.72) while losing the empty/truncated failures.
The purpose is **reliability**, not a WER win.

**Caveat.** The patch is applied to a Cargo git checkout, not to the Cargo
manifest. It survives `cargo clean` (the checkout is outside `target/`), but it
is lost if the pinned revision changes or the checkout is recreated. Reapply it
from the Zenith patch file and rebuild. Upstreaming it to
`handy-computer/transcribe.cpp` #252 is the durable fix.

---

## 4. The bridge and live text

The bridge is in `~/zenith-audio/bridge/`; this Hex checkout only provides the
engine. Summary:

- **Transport:** TCP 59102, 5-byte frame header (`type`, `u32be` length).
- **START/AUDIO/END** from the phone; **PARTIAL/RESULT/ERROR** from the bridge.
- **Resident engine:** `create({ command: [hex, "service", "--embedded"], model,
  language })` from `@kitlangton/hex`; the model stays loaded.
- **Live:** every `liveStrideMs` (default 500 ms) of new audio, re-transcribe the
  last `liveWindowMs` (default 4000 ms), starting after a pause when possible;
  send a `PARTIAL`. The client replaces the preview.
- **Final:** on END, transcribe the **whole utterance** and send `RESULT`.
  Partials are tentative; the final is authoritative. There are no chunk joins,
  so no word is lost or duplicated.
- **Security:** default `127.0.0.1`; a non-loopback bind requires `--token`
  (timing-safe comparison). The phone sends the token in START.
- **Limits:** `--max-seconds` (120), `--idle-timeout-ms` (30 s without audio
  cancels the session), one active dictation (`model-busy` for a second).

The Android IME is **not implemented yet**. It will capture the mic, stream the
frames, show `PARTIAL` as `setComposingText`, and commit `RESULT` with
`commitText`.

---

## 5. Why this model, not another

The chosen model is `alexandreacff/parakeet-tdt-v3-ptBR-30s`, a Parakeet **v3**
fine-tune (multilingual, 8192-token tokenizer) converted to Q8_0.

- **Stock Parakeet v3 is wrong for pt-BR.** NVIDIA's model card says its
  Portuguese training data is **European Portuguese**, and the model accepts no
  language input; it guesses and can drift to English on short spans
  (NVIDIA-NeMo/Speech#14799).
- **`-plus` is not public.** The ONNX/GGUF third-party cards point to
  `alexandreacff/parakeet-tdt-0.6b-v3-ptBR-plus`, but the repo answers 401 and
  the author's public list has only `-30s`, `-20s`, `tagarela-v2` and
  `distil-whisper-tagarela-v2`. Do not build on it.
- **`tagarela-v2` is a different lineage.** Its `model_config.yaml` uses
  Parakeet **v2** (English) with a **1024-token English** tokenizer
  (`en_mm_ytc/tokenizer_spe_bpe_v1024`), while `-30s` uses v3 with 8192
  multilingual tokens. Its vocab has 2 tokens containing `ç`/`ã`/`ção` against
  14 in `-30s`; it is likely worse for pt-BR and was not pursued.
- **Whisper large-v3-turbo** is the fallback. It is slightly better on read
  speech but worse on spontaneous speech and ~6× slower on CPU in these tests
  (4.5 s vs 0.73 s per clip, including process start/model load).

The model spells numbers out ("mil e quatrocentos" for "1.400"). That is a
formatting difference, not a recognition error; inverse text normalization
would be a bridge step if digits are wanted.

---

## 6. OpenCodeReview findings

Two OCR passes were run on the bridge source (25 then 21 findings). The
high-severity and most medium findings were fixed before the commit:

- the `--live-*` flags were ignored; they now become the session defaults;
- the single-dictation slot had a stale-release race; ownership is now tied to
  the socket;
- a client that sent START and stopped sending audio could hold the slot
  forever; an idle timeout now cancels it;
- the bridge bound `0.0.0.0` without a token by default; it now defaults to
  loopback and refuses a non-loopback bind without `--token`;
- `close()` could tear down the native engine during an inference; it now waits
  for the queue with a bound and skips the native close if it did not drain;
- the live pass copied the whole utterance on every pass; a growable buffer
  makes the full PCM a view and copies only the tail window;
- frame/WAV validation, backpressure, argument validation and JSON parsing in
  the test client were hardened.

Remaining debt (mostly low; one medium): a corrupt frame header in the same TCP
read can discard earlier parsed frames before the error; the frame buffer does
not shrink after a large frame; the pause heuristic's thresholds are constants.
See the Zenith repo for the full OCR outputs.

---

## 7. Upstream context

Hex issues/PRs relevant to this work:

| Item | State | Relevance |
|---|---|---|
| [Hex #1136](https://github.com/anomalyco/hex/issues/1136) — transcribe chunks during dictation for non-streaming models (Parakeet Ultra, v3) | open | The exact upstream feature; maintainers acknowledge, no implementation. |
| [Hex #1470](https://github.com/anomalyco/hex/issues/1470) — live text for Parakeet v3/Ultra | open | The bridge implements this idea independently (tail re-transcription). |
| [Hex #878](https://github.com/anomalyco/hex/issues/878) — live text for Whisper | open | Working fork handoff, not merged. |
| [Hex #449](https://github.com/anomalyco/hex/issues/449) — live transcript preview | open | Related UI request. |
| [Hex #957](https://github.com/anomalyco/hex/issues/957) — client/server modes over LAN/Tailscale | open | Aligns with phone→PC; no implementation. |
| [Hex PR #1474](https://github.com/anomalyco/hex/pull/1474) — `/dictations` partials | merged | Live text over SSE only when the engine model streams. |
| [transcribe.cpp #252](https://github.com/handy-computer/transcribe.cpp/issues/252) — Parakeet v3 loses text on some lengths | open | The batch blocker this fork patches. |
| [transcribe.cpp #215](https://github.com/handy-computer/transcribe.cpp/issues/215) — Parakeet streaming commit freeze | open | Streaming blocker; a fix is offered, not merged. |

Hex does not accept pull requests; the transcribe.cpp patch is the one change
that could be upstreamed there (or carried in a fork).

---

## 8. Commands and paths

```bash
# Build Hex (after the patch, if needed)
cd ~/hex
CARGO_BUILD_JOBS=16 cargo build --locked --release

# Import the model (once)
~/hex/target/release/hex model import \
  ~/.cache/parakeet-spike/models/parakeet-tdt-v3-ptBR-30s/parakeet-tdt-v3-ptBR-30s-Q8_0.gguf

# Check the model on the GPU
~/hex/target/release/hex model check custom-7d13a4fb474b2100 --language pt

# Transcribe a file (offline, no service)
~/hex/target/release/hex model transcribe sample.wav \
  --model custom-7d13a4fb474b2100 --language pt

# Bridge: loopback (local test)
cd ~/zenith-audio/bridge
bun install
bun run start -- \
  --hex ~/hex/target/release/hex \
  --model custom-7d13a4fb474b2100 \
  --language pt

# Bridge: phone on the LAN (token required)
TOKEN=$(openssl rand -hex 16)
bun run start -- \
  --host 0.0.0.0 --token "$TOKEN" \
  --hex ~/hex/target/release/hex \
  --model custom-7d13a4fb474b2100 \
  --language pt

# Test through the bridge (48 kHz mono s16le WAV)
ffmpeg -i input.wav -ar 48000 -ac 1 -c:a pcm_s16le /tmp/sample-48k.wav
bun run live-test -- --wav /tmp/sample-48k.wav --token "$TOKEN" --fast
```

Paths:

| Path | Meaning |
|---|---|
| `~/hex/target/release/hex` | Built Hex 3.3.0 (not installed in `$PATH`) |
| `~/.local/share/hex/models/custom/` | Imported GGUF and manifest |
| `~/.cargo/git/checkouts/transcribe.cpp-0d04a17001a6bbfd/2778ae4/` | Pinned transcribe.cpp checkout carrying the #252 patch |
| `~/.cache/parakeet-spike/models/parakeet-tdt-v3-ptBR-30s/` | Q8_0 GGUF (and the `.nemo` source one level up) |
| `~/zenith-audio/docs/patches/transcribe-cpp-252-compact-last-mel-frame.patch` | The #252 patch |
| `~/zenith-audio/docs/phone-dictation.md` | Full architecture/decision record |
| `~/zenith-audio/bridge/` | TCP bridge + live scheduler + tests |

---

## 9. What is local vs upstream

- **Local to this fork/machine:** the built binary, the imported
  `custom-<hash>` model, the patched transcribe.cpp checkout, the bridge, and
  the model id. None of this is upstream Hex.
- **Upstream Hex:** the `dev` branch was fast-forwarded to `62a40669`; no
  source files were modified except the external dependency checkout.
- **The patch** is a local change to a pinned dependency; it is not in this
  repository's git history. The durable copy is the `.patch` file in the Zenith
  repo. If the checkout is updated, reapply it and rebuild.
- **The bridge** is in the Zenith repo, not Hex; it uses Hex through the
  published SDK and the embedded service.

---

## 10. Known gaps and next steps

1. **Android IME** — capture, TCP, `setComposingText` for PARTIAL, `commitText`
   for RESULT, cancel and permission handling. This is the remaining piece for
   the phone experience.
2. **Token pairing** — the bridge accepts `--token`; the phone needs a way to
   store it (manual entry or a small pairing flow).
3. **Pause heuristic** — the live window starts after 100 ms below RMS 0.01 in
   the first 800 ms. Tune it against the phone's noise floor and mic gain.
4. **#252 upstream** — send the compaction patch to transcribe.cpp #252 (or a
   fork) so the fix survives a Hex update. Hex itself does not take PRs.
5. **Idle unload** — the bridge keeps the model resident until it exits; a
   future `--idle-unload` can release VRAM without stopping the bridge.
6. **Number formatting** — the `-30s` model spells numbers out; add inverse text
   normalization in the bridge if digits are preferred.
7. **OCR debt** — parser frame-loss on corrupt same-read input (medium),
   frame-buffer shrinking, and pause threshold configurability (low).

---

## 11. Quick context for a future LLM

If you continue this work:

- Start with `~/zenith-audio/docs/phone-dictation.md` and
  `~/zenith-audio/bridge/README.md`.
- Do not "fix" the model choice by switching to stock Parakeet v3 (pt-PT,
  language drift) or `tagarela-v2` (1024-token English lineage). The chosen
  model is the `-30s` v3 fine-tune, imported as `custom-7d13a4fb474b2100`.
- Do not remove the transcribe.cpp #252 patch; without it, batch dictation
  returns empty/truncated text on ~1/8 of clip lengths. Reapply the patch from
  `~/zenith-audio/docs/patches/` if the checkout changes.
- The bridge's final pass always transcribes the whole utterance; partials are
  tentative and must be replaced, never appended.
- The model id is machine-local; regenerate it with `hex model import` on
  another machine.
- Hex's `dev` branch is currently 46 commits ahead of the fork's `origin/dev`;
  the local `dev` matches `upstream/dev` at `62a40669`.
