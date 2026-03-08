# transcribe

Fully local, real-time, low-latency, high-accuracy speech transcription for lightweight devices (especially laptops).  
Designed for psychotherapy workflows with a delivery path to clinical-grade compliance.

As of March 2, 2026, this README defines an implementation plan from prototype to a clinically deployable v1.

## Python Support

- Runtime and development environments support Python 3.13+.
- Windows standalone packaging with `uv run .\scripts\build_windows_standalone.py --backend nuitka` currently requires Python 3.13 specifically.
- If you are building the Windows standalone package from Python 3.14+, use `--backend pyinstaller` or switch the build environment to Python 3.13 until Nuitka's Python 3.14 support is stable for the NeMo/librosa dependency path.

## Regulatory Baseline

This plan must align with:

- `REGULATORY_CHECKLIST.md` (offline-first minimum engineering controls)
- `REGULATOR_COMPLIANCE_POLICY.md` (mandatory engineering policy)

Key operating assumption:

- Runtime is offline-only: no telemetry, no cloud inference, no remote analytics, no background network traffic.
- No secondary use of health data (no advertising, profiling, or model training on client data by default).

Clinical-use boundary:

- Phases 0-1 are engineering builds and are not approved for clinical use.
- Until compliance gates are complete, only synthetic or de-identified test data should be used.
- Early architecture must preserve a direct path to full compliance (no shortcuts that create rework later).

## Product Intent

- 100% local inference (no cloud dependency, no API keys).
- Real-time streaming transcription from microphone and/or system audio.
- Fast partial updates with stable final text.
- Strong accuracy across noisy real-world speech.
- Runs on common hardware: Apple Silicon laptops, Intel/AMD laptops, and low-power edge devices where possible.
- Speaker diarization by default; can disambiguate therapist and patient voices.
- Configurable audio source modes: `mic`, `speakers`, or `both`.
- Two product phases:
1. Phase 1: high-quality therapy session transcript.
2. Phase 2: local LLM-generated client notes from transcript.
- Client notes should be based on best-practice guidance plus therapist-configurable rules/templates.

## Non-Goals (v1)

- Training custom speech or language models.
- Non-English language support.
- Cloud sync or cloud inference.

## Scope Clarifications

- "HIPAA-aligned" means implementing technical safeguards that make compliance feasible. Formal legal certification is separate from engineering completion.
- v1 target is desktop/laptop first, edge-device support second.
- CLI-first is acceptable for initial releases; GUI can follow once core pipeline is stable.
- Any future networked feature requires a separate compliance expansion plan.

## Success Criteria (v1)

Functional:

- End-to-end transcript latency for partials: under 1.0s median on target laptop hardware.
- End-to-end finalization latency per segment: under 3.0s median.
- Diarization quality: less than 10% speaker-label confusion on evaluation sessions.
- Stability: 60-minute session without crash or memory leak.

Compliance:

- Runtime proven to function with no network connectivity.
- No plaintext PHI in persistent storage, logs, temp files, or caches.
- All sensitive persisted artifacts encrypted at rest.
- Tamper-evident local audit logs with retention controls.

## Compliance Posture by Stage

1. PoC and Harness (Phase 0)
- Goal: verify feasibility and performance.
- Constraint: no architectural decisions that block future compliance controls.

2. MVP Transcription (Phase 1)
- Goal: functional transcript flow and diarization.
- Constraint: remain non-clinical until auth, encryption, and audit controls are active.

3. Clinical Readiness (Phases 1.5-3)
- Goal: complete required controls and evidence.
- Constraint: release only after checklist/policy gates are passed.

## Proposed Technical Architecture

1. Audio Capture Layer
- Inputs: mic, system loopback, or mixed.
- Platform backends:
  - macOS: AVFoundation loopback + mic.
  - Linux: PulseAudio/PipeWire capture.
  - Windows: WASAPI loopback + mic.
- Common output format: mono 16k PCM frames.

2. Streaming Preprocessing
- Optional denoise and AGC.
- Voice activity detection (VAD) for segmentation and compute reduction.
- Sliding window chunker with overlap for context continuity.

3. ASR Engine (Local)
- Runtime abstraction with pluggable backend.
- Start with one default backend (fastest path), keep interface for swap/evaluation.
- Emits partial hypotheses and finalized segments with timestamps.

4. Diarization Layer
- Online speaker embedding + clustering where possible.
- Fallback: post-hoc diarization pass for improved final transcript labels.
- Speaker mapping workflow: therapist can relabel speakers after session.

5. Transcript State Manager
- Maintains partial vs final text, timestamps, and speaker IDs.
- Handles segment revisions without losing history.
- Preserves change metadata to support auditability and rectification workflows.

6. Security and Data Layer
- Local encrypted session store for transcripts, notes, and metadata.
- Data classification at ingestion (`PHI`, `personal`, `operational`).
- Key management via OS secure storage and/or credential-derived keys.
- Temporary files, caches, and backups are protected using the same data handling rules as primary records.
- Tamper-evident local audit log for read/write/delete/export/auth events.
- Integrity verification detects corruption or unauthorized modification.

7. Identity and Access Layer
- Local user authentication required.
- Strong password/passphrase policy, adaptive hashing (Argon2id/bcrypt).
- Inactivity auto-lock and session expiration.
- Local profile separation and RBAC for multi-user configurations.
- MFA support for privileged/clinical accounts where practical in offline mode.
- Biometric unlock is optional convenience only, not sole authentication.

8. Notes Generation (Phase 2)
- Local LLM runtime abstraction (model configurable).
- Prompt pipeline:
  - session transcript
  - therapist template/rules
  - compliance constraints
- Structured output schema for consistent note format.
- Notes are advisory drafts for clinician review, not autonomous decisions.

## Model/Runtime Strategy

Build around interfaces first, then benchmark concrete backends.

- ASR candidates:
  - `faster-whisper` (strong Python integration, good speed/accuracy tradeoff).
  - `whisper.cpp` (excellent local portability and quantization options).
- Diarization candidates:
  - `pyannote.audio` pipeline (higher quality, heavier runtime).
  - embedding + clustering lightweight path (faster/lighter, lower quality ceiling).
- LLM candidates for notes:
  - small local instruct models (for example Qwen-family instruct variants).
  - runtime options: `llama.cpp`-based or equivalent local inference runtime.

Decision rule: choose defaults by measured latency/accuracy on target hardware, not by benchmark claims.

Current production model options (selected from `data/benchmarks/hf_diarized`):

- Quality profile: `nvidia/canary-qwen-2.5b`
- Fast/low-capability profile: `nvidia/parakeet-tdt-0.6b-v3`

## ASR Backend Setup (Bench)

Install optional ASR backends for non-Whisper model families:

```bash
# Qwen models only
uv sync --extra qwen-asr

# NVIDIA NeMo models only
uv sync --extra nemo-asr

# Both families
uv sync --extra asr-backends
```

Notes:

- `nemo-asr` builds native dependencies (for example `kaldialign`) and requires local build tools (`cmake`, `make`, C/C++ compiler).
- If you only need `qwen/*` models, prefer `--extra qwen-asr` to avoid NeMo native build requirements.

Model routing used by `bench` and `init-bench`:

- `whisper-*` and `faster-whisper-*` -> `faster-whisper`
- `nvidia/*` -> `nemo_asr` (`nemo_toolkit[asr]`)
- `qwen/*` -> `qwen-asr`

Examples:

```bash
# Cache model + dataset assets
uv run init-bench --model nvidia/canary-qwen-2.5b
uv run init-bench --model nvidia/parakeet-tdt-0.6b-v3

# Run diarized transcription benchmark
uv run bench --model nvidia/canary-qwen-2.5b
uv run bench --model nvidia/parakeet-tdt-0.6b-v3
```

## Live Session Test Rig (Stage 1)

Use the Stage 1 live session runner for open-mic streaming transcription on Linux.

`session run` now captures all available input devices in the selected mode and
automatically routes each frame from the clearest source (for example, best mic
or best speaker loopback at that moment).

```bash
# Fixture dry-run (no audio hardware needed)
.venv/bin/python3 -m transcribe.cli session run --fixture --duration-sec 15 --chunk-sec 3

# Real microphone run (Ctrl+C to stop, default model is NVIDIA Parakeet)
.venv/bin/python3 -m transcribe.cli session run \
  --duration-sec 0 \
  --mode both \
  --chunk-sec 4 \
  --partial-interval-sec 0

# Use a larger quality-focused model explicitly
.venv/bin/python3 -m transcribe.cli session run \
  --model nvidia/canary-qwen-2.5b \
  --duration-sec 0 \
  --mode both \
  --chunk-sec 4 \
  --partial-interval-sec 0

# Pin a specific mic by index (from `capture devices`)
.venv/bin/python3 -m transcribe.cli session run \
  --duration-sec 0 \
  --mode both \
  --mic-device 2

# Or pass exact device names shown in `capture devices`
.venv/bin/python3 -m transcribe.cli session run \
  --duration-sec 0 \
  --mic-device "USB Audio Device: - (hw:2,0)"

# Restrict to one device per source type and fail if any source is missing
.venv/bin/python3 -m transcribe.cli session run \
  --duration-sec 0 \
  --mode both \
  --single-device-per-source \
  --strict-sources
```

For larger models (for example `nvidia/canary-qwen-2.5b`), keep partials off
(`--partial-interval-sec 0`) unless your machine can sustain real-time inference.
Silent chunks are automatically detected and skipped (no ASR call) so the session
can catch up during quiet periods.
Chunks are trimmed for leading/trailing silence and resampled to the requested
ASR rate (default `16 kHz`) before inference, even when capture hardware runs
at `44.1/48 kHz`.
Final chunks now carry a short audio overlap into the next chunk by default
(`--chunk-overlap-sec 0.75`) to reduce clipped words at phrase boundaries.

Quality tuning tips:

- If logs show `No speaker monitor/loopback device found`, you are effectively
  transcribing microphone audio only. Use `--mode mic` for that workflow.
- Use `--chunk-sec 3` or `--chunk-sec 4` for better phrase boundaries with
  conversational speech.
- Keep `--partial-interval-sec 0` on larger models to preserve final accuracy.

Outputs are written under `data/live_sessions/<session-id>/`:

- `events.jsonl` (partial/final event stream)
- `transcript.json` (session metadata + finalized segments)
- `transcript.txt` (plain finalized transcript)

The Linux capture backend now auto-negotiates a supported input sample rate and reports
requested/effective values at session end.

If you want the larger quality-focused model instead, use:
`uv run transcribe session run --model nvidia/canary-qwen-2.5b --duration-sec 0`.

Add `--debug` to surface backend logs, partial events, and detailed session diagnostics.

If the model is not already cached locally, pre-populate it first (offline policy):

```bash
uv run init-bench --model nvidia/canary-qwen-2.5b
```

`nvidia/canary-qwen-2.5b` depends on additional local tokenizer/model assets
(for example `Qwen/Qwen3-1.7B`); `init-bench` pre-populates those dependencies.

## Delivery Plan

## Phase 0 - Foundation and Compliance-Ready Architecture (Week 1)

Deliverables:

- Project skeleton and configuration system.
- Audio capture abstraction with per-OS adapters (start with one OS fully working).
- Reproducible benchmark harness for latency and quality checks.
- Compliance-first foundations:
  - no-network runtime mode and explicit prohibition on outbound calls.
  - no hard-coded URLs/endpoints or runtime remote dependencies.
  - PHI-safe logging policy (no plaintext health data in logs).
  - data classification fields in core schema.

Exit criteria:

- Can capture and save synchronized audio stream(s) from selected source mode.
- Benchmark command produces consistent timing outputs.
- Architecture review confirms no blockers for encryption, auth, audit, and retention controls.

## Phase 1 - Live Transcription MVP (Weeks 2-4)

Deliverables:

- Streaming ASR pipeline with partial and final segment updates.
- Basic diarization integration.
- CLI session runner: start/stop, transcript output, JSON export.
- Compliance scaffolding:
  - transcript version/change metadata.
  - audit event schema (read/write/delete/export/auth).
  - non-clinical build labeling.

Exit criteria:

- 30-minute live session completes without crash.
- Partial/final transcript stream works end-to-end.
- Speaker labels available on finalized segments.
- Build is explicitly restricted to non-clinical testing unless later compliance gates are enabled.

## Phase 1.5 - Quality and Security Baseline (Weeks 5-6)

Deliverables:

- Model/backend evaluator (accuracy vs latency on curated test set).
- Improved segmentation and punctuation normalization.
- Error handling and recovery for device/input interruptions.
- Local access controls:
  - authentication and inactivity lock.
  - Full session files including all PII are contained in one encrypted and compressed at rest folder.
  - Therapist is prompted to unlock at the start of the session.
  - Unencrypted files exist only in memory and are written to disc encrypted and compressed.
  - password policy aligned with NIST-style guidance.
  - credential hashing with Argon2id.

Exit criteria:

- Meets latency targets on at least one Apple Silicon and one Intel/AMD profile.
- Diarization confusion under target on internal eval set.
- Security tests pass for auth flow, lock behavior, and PHI-safe logging.

## Phase 2 - Notes Generation MVP with AI Controls (Weeks 7-8)

Deliverables:

- Local LLM integration behind a provider interface.
- Therapist-configurable prompt templates and style constraints.
- Structured note generation outputs (for example JSON + rendered text).
- AI governance controls:
  - feature is opt-in.
  - explicit limitations/disclosure in UX and docs.
  - no model training on user data by default.

Exit criteria:

- Given a completed session transcript, notes generate locally with no network calls.
- Output conforms to configured template and section requirements.
- Notes are marked advisory and clinician-reviewed.

## Phase 2.5 - Privacy and Data Lifecycle Controls (Week 9)

Deliverables:

- Encryption at rest for transcripts, notes, exports, and backups.
- Key management using OS secure storage and key rotation policy.
- Retention/deletion workflows including cryptographic erasure where feasible.
- Full-disk encryption support/enforcement documented per platform where possible.
- Export and rights support:
  - structured machine-readable export.
  - encrypted export by default with explicit warning flow.
  - rectification workflow that preserves audit history.
- Tamper-evident local audit log with retention configuration.
- Audit timestamps include UTC and local timezone context.

Exit criteria:

- Security checklist passes for local-only operation.
- Export/delete/rectification actions verified by integration tests.
- `REGULATORY_CHECKLIST.md` controls pass with implementation evidence.

## Phase 3 - Clinical Compliance Readiness (Weeks 10-12)

Deliverables:

- Compliance evidence package mapping:
  - `REGULATORY_CHECKLIST.md` -> implemented controls and tests.
  - `REGULATOR_COMPLIANCE_POLICY.md` -> implemented controls, process, or explicit N/A rationale.
- Secure SDLC controls in CI:
  - dependency vulnerability scanning.
  - release block on high/critical CVEs.
- Security governance artifacts:
  - threat model for PHI data flows and trust boundaries.
  - pre-release security review checklist.
  - incident response runbook (24h triage target, 72h notification readiness).
  - annual security risk assessment and compliance training requirements documented.
  - local security monitoring for failed login and privilege escalation events.
- Audit and retention hardening:
  - tamper-evidence verification.
  - six-year retention support for audit logs.

Exit criteria:

- All checklist items complete.
- Applicable policy controls satisfied and evidenced.
- Build approved for clinical deployment.

## Control Mapping (Checklist + Policy)

- Network isolation (Checklist 1, Policy 1/2): enforced from Phase 0 onward.
- Data classification/minimization (Checklist 2, Policy 1/2): schema and handling in Phase 0, verified by Phase 2.5.
- Encryption/key management (Checklist 3, Policy 4): implemented Phase 2.5, validated Phase 3.
- Auth/access/RBAC/session controls (Checklist 4/5, Policy 3): implemented Phase 1.5-2.5, validated Phase 3.
- Audit logging/retention/integrity (Checklist 6/7, Policy 5): implemented Phase 1-2.5, hardened Phase 3.
- Export/rectification/erasure (Checklist 8/9, Policy 6): implemented Phase 2.5, validated Phase 3.
- Secure SDLC and dependency controls (Checklist 10, Policy 8/12): introduced Phase 1.5, enforced Phase 3.
- AI restrictions (Checklist 11, Policy 10): implemented Phase 2.
- Platform security integration (Checklist 12, Policy 4/8): implemented Phase 2.5.
- Incident readiness (Policy 9): documented and tested in Phase 3.
- Vendor/subprocessor controls (Policy 11): N/A for offline runtime; required before any third-party PHI processor is introduced.
- Data residency/cross-border transfer (Policy 7): satisfied by local-only storage; re-evaluate if sync features are added.
- Data in transit controls (Policy 4.1): N/A for offline-only runtime; TLS 1.2+ and related controls become mandatory if network transport is introduced.

## Testing Plan

- Unit tests:
  - audio chunking, timestamp alignment, transcript merge logic, prompt rendering.
  - auth/session timeout, key handling, and audit event creation.
- Integration tests:
  - simulated audio session -> transcript file -> note generation.
  - export/rectification/deletion workflow with audit preservation.
- Compliance tests:
  - runtime no-network verification.
  - encrypted-at-rest checks for DB/files/backups/exports.
  - no-PHI-in-logs checks.
  - tamper-evident audit log validation.
- Security pipeline checks:
  - dependency vulnerability scanning.
  - secret leakage and unsafe logging detection.
- Performance tests:
  - latency and throughput on representative hardware tiers.
- Regression set:
  - fixed corpus of anonymized speech samples with expected transcript deltas.

## Risks and Mitigations

- Cross-platform audio capture complexity:
  - mitigate with strict adapter interface and per-OS test matrix.
- Diarization quality in overlapping speech:
  - mitigate with confidence scores and post-session correction workflow.
- Latency vs accuracy tradeoff:
  - mitigate with configurable model sizes and dynamic fallback profiles.
- Local LLM hallucination in notes:
  - mitigate with structured outputs and grounding to transcript excerpts.
- Early-phase compliance drift:
  - mitigate with phase exit gates tied to checklist/policy evidence.

## Release Gates

1. Engineering PoC Gate (end of Phase 1)
- Functional MVP only, non-clinical use.
- No architectural blockers to mandatory controls.

2. Security Baseline Gate (end of Phase 2.5)
- Encryption, auth, audit, export, deletion controls implemented.
- Checklist mostly complete with remaining hardening scoped to Phase 3.

3. Clinical Release Gate (end of Phase 3)
- Checklist complete.
- Policy controls implemented and evidenced.
- Approved for clinical deployment.

## Immediate Next Steps (Build Order)

1. Implement audio capture interface and one production-grade OS backend.
2. Add streaming ASR path with partial/final events and transcript state manager.
3. Add audit event schema, data classification fields, and PHI-safe logging guardrails.
4. Continue benchmark regressions and validate `nvidia/canary-qwen-2.5b` + `nvidia/parakeet-tdt-0.6b-v3` on target hardware tiers.
5. Implement local auth/session lock and encrypted persistence before any clinical pilot.



