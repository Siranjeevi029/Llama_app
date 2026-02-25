# AI Tools & Frameworks - 2 Weeks Research Report

**Author:** Siranjeevi P
**Period:** February 2 - February 16, 2026

---

## Table of Contents

1. [MemU - Long-Term Memory Framework](#1-memu---long-term-memory-framework)
2. [LiveKit - Conversational AI with Telephony](#2-livekit---conversational-ai-with-telephony)
3. [Moondream3 - Vision Language Model](#3-moondream3---vision-language-model)
4. [Shannon - AI Penetration Testing Framework](#4-shannon---ai-penetration-testing-framework)
5. [Decart - AI Video Editing](#5-decart---ai-video-editing)
6. [Primer - AI-Ready Repo Documentation](#6-primer---ai-ready-repo-documentation)
7. [Matryoshka Embedding Model](#7-matryoshka-embedding-model)
8. [Sarvam Bulbul v3 - Text-to-Speech](#8-sarvam-bulbul-v3---text-to-speech)

---

## 1. MemU - Long-Term Memory Framework

**Date:** February 2, 2026

### Overview

MemU is a memory framework that gives LLMs long-term memory, enabling them to recall and respond based on past interactions. It requires a paid API or self-hosted open-source deployment.

### Key Features

- **Memorize & Retrieve** architecture for structured memory management
- Supports RAG-based semantic retrieval or LLM-based retrieval
- Configurable storage: in-memory, PostgreSQL, etc.
- Per-user memory isolation via `user_id`
- Modality-based separation to avoid cross-topic memory confusion
- Compatible with OpenRouter API for flexible model selection

### Setup

- Create a MemU service with desired LLM and embedding models
- Configure retrieval method:
  - `retrieve_config = {"method": "rag"}` for semantic embedding retrieval
  - `retrieve_config = {"method": "llm"}` for LLM-based document retrieval
- Use `user_id` to scope memory per user
- Set `modality` to categorize and separate different types of information

### Performance

| Metric | Value |
|--------|-------|
| Avg retrieval latency | 8 - 10 seconds |

### Testing Results

- **Input:** `example_conversation.json`, `memU_sample.txt`
- **Color preference test:** When user changed preference from yellow to pink, MemU retained both entries but failed to prioritize the latest update
- **Conclusion:** MemU doesn't hallucinate, but struggles with temporal tracking — it can't reliably distinguish between outdated and current preferences

### Limitations

- Does not handle preference updates (old vs. new) well
- Retrieval stores collective info rather than latest state
- When integrated with a Claude/Groq agent, it stores both old and new values and responds with combined info (e.g., "user likes both yellow and pink")

### Demo Files

| File | Description |
|------|-------------|
| `mem U gradio.mp4` | Basic memorize and retrieve demo |
| `mem U create.mp4` | Memory creation workflow |
| `mem U update.mp4` | Memory update workflow |
| `groq memU chatting.mp4` | Chat agent with MemU integration |
| `groq memU chatting2.mp4` | Color preference update demo |

### References

- [Claude Agent using MemU](https://github.com/NevaMind-AI/open-personal-agent)
- Code: `memU_groq_chat`

---

## 2. LiveKit - Conversational AI with Telephony

**Date:** February 6, 2026

### Overview

LiveKit is a framework for building real-time conversational AI applications. It follows a **STT -> LLM -> TTS** pipeline for voice agents.

### Architecture

```
Speech-to-Text (STT) → LLM Processing → Text-to-Speech (TTS)
```

### Implementation

1. **Agent Setup:** Create an agent with room options in a LiveKit project. The agent enters a room only when triggered.
2. **Room Creation:** A separate script creates the LiveKit room, connecting the user and the LLM.
3. **Execution:**
   - `uv run agent.py start` — starts the agent (waits for trigger)
   - `uv run make.py` — triggers the LLM to join the room and calls the user

### Twilio Integration

- LiveKit supports telephony via Twilio for outbound calls
- Twilio provides $15 free credits for phone number provisioning
- Outbound call flow: LiveKit triggers Twilio → Twilio calls the user's phone number → User joins the LiveKit room

### Demo Files

| File | Description |
|------|-------------|
| `livekit.mp4` | Voice agent pipeline demo |
| `livekit_twilio.mp4` | Telephonic call via Twilio |
| `twilio_call.png` | Screenshot of triggered call |

---

## 3. Moondream3 - Vision Language Model

**Date:** February 8, 2026

### Overview

**Model:** `moondream/moondream3-preview` (9B parameters)

### GPU Loading Issues

| Setup | Result |
|-------|--------|
| Single GPU | Model loading failure |
| Multiple GPUs | Loads successfully (17.5 GiB VRAM) but fails at inference |

**Inference Error:**
```
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
```

**Root Cause:** Moondream3 uses custom rotary embedding with internal `self.device` tracking. It assumes single-device execution and is not tensor-parallel safe — **cannot be loaded across multiple GPUs**.

### Test Results

| Input | Task | Notes |
|-------|------|-------|
| `twilio_call.png` | Image understanding | - |
| `trafficlight.webp` | Object recognition | - |
| `captcha1.jpeg` | CAPTCHA reading | - |
| `charge.webp` | Image analysis | - |
| `bus_number1.jpg` | Text extraction | - |
| `trafficlight2.jpg` | Scene understanding | - |
| `explain_scenario1.png` | Scenario explanation | - |
| `object_detect1.jpg` | Object detection | - |
| `obj_det2.jpg`, `obj_det3.jpg` | Object detection | - |
| `loginpage1.png` - `loginpage4.jpg` | UI understanding | - |
| `syspage1.jpg` - `syspage4.png` | System page analysis | - |

---

## 4. Shannon - AI Penetration Testing Framework

**Date:** February 9-10, 2026

### Overview

Shannon is an autonomous AI pentester that delivers actual exploits, not just vulnerability alerts. It hunts for attack vectors in code, then uses a built-in browser (Playwright MCP server) to execute real exploits including injection attacks and auth bypasses.

**GitHub:** [KeygraphHQ/shannon](https://github.com/KeygraphHQ/shannon)

### Setup

- Uses OpenAI model: `gpt-4o-mini`
- Target tested: `https://authorization-frontend.netlify.app/`

### How It Works

1. Generates a web-based workflow dashboard URL for monitoring
2. Uses Playwright MCP server to manually browse and interact with the target website
3. Tests for vulnerabilities including injection attacks and authentication bypass
4. Produces a comprehensive security assessment report

### Output

- `comprehensive_security_assessment_report.md` — Full vulnerability and security report

---

## 5. Decart - AI Video Editing

**Date:** February 12, 2026

### Overview

Decart is an AI-powered live video editing platform.

**Credits:** 1,000 free credits per account

### Editing Modes

| Mode | Cost | Quality | Duration | Audio Sync |
|------|------|---------|----------|------------|
| Live video editing | 1 credit/sec | Good | Full | Not always proper |
| Video-to-video | 15 credits/sec | Good accuracy | Clips to first 5 sec | Good |
| Long video restyling | 1 credit/sec | Lower (forces animation) | Full video | Good |

### Test Results

| Prompt | Input | Output | Notes |
|--------|-------|--------|-------|
| Change T-shirt color to yellow | `hugh jakman interview.mp4` | `decart-edit-yellow.mp4` | Works, occasional hallucinations |
| Change hair color to black | `taylor-swift.mp4` | `decart-haircolor-black.mp4` | - |
| Make the woman wear specs | `taylor-swift.mp4` | `decart-specs.mp4` | - |
| Video-to-video conversion | `taylor-swift.mp4` | `decart-5sec.mp4`, `decart-5sec-2ndrun.mp4` | Clips to 5 seconds |
| Long restyle: yellow dress | - | `decart-long1.mp4` | Generates animated output even without asking |
| Long restyle: add specs | - | `decart-long2.mp4` | Same animation issue |

### Known Issues

- Live editing can hallucinate with random artifacts
- Audio sync inconsistencies in live editing mode
- Video-to-video clips output to first 5 seconds
- Long video restyling always produces animated output regardless of prompt

---

## 6. Primer - AI-Ready Repo Documentation

**Date:** February 13, 2026

### Overview

Primer is a CLI tool that analyzes codebases and generates `.github/copilot-instructions.md` files to help AI coding assistants understand projects. Supports single repos, batch processing across organizations, and includes an evaluation framework.

**GitHub:** [pierceboggan/primer](https://github.com/pierceboggan/primer)

### Testing Methodology

Compared AI responses **with** vs **without** Primer-generated instructions across multiple repositories.

### Results

| Repository | Results File | With Instructions |
|------------|-------------|-------------------|
| [microsoft/BitNet](https://github.com/microsoft/BitNet) | `results_bitnet2.json` | Better |
| [KeygraphHQ/shannon](https://github.com/KeygraphHQ/shannon) | `results_shannon2.json` | Better |
| [pierceboggan/primer](https://github.com/pierceboggan/primer) | `results_primer2.json` | Better |

### Conclusion

With `copilot-instructions.md`, the model performs better than usual in most scenarios.

---

## 7. Matryoshka Embedding Model

**Date:** February 13, 2026

### Overview

### How It Works

| Model Type | Training | Truncation Impact |
|------------|----------|-------------------|
| **Normal embedding** | Optimized only for full vector (e.g., 768 dims) | Sharp performance drop when truncated |
| **Matryoshka embedding** | Optimized at multiple sizes (768, 512, 256, 128, 64) | Minimal loss — important info packed into early dimensions |

### Test Setup

- **Embedding dimension:** 64 (same for both models)
- **Input:** `input.txt`

### Models Compared

| | Normal Model | Matryoshka Model |
|--|--------------|------------------|
| **Model** | `tomaarsen/mpnet-base-nli` | `tomaarsen/mpnet-base-nli-matryoshka` |
| **Retrieval time** | Faster | Slightly slower |
| **Storage** | More space | Less space |
| **Top-k accuracy** | Poor — fails to retrieve correct results | Excellent — correctly retrieves answers |
| **Confidence** | Low separation between top-k scores | High separation between top-k scores |

### Accuracy Tests

| Query | Normal Model | Matryoshka Model |
|-------|-------------|------------------|
| Capital of France | Correct | Correct |
| When did WW2 start? (Answer: 1939) | Incorrect (returned 1941) | Correct (returned 1939) |

### Conclusion

At the same low dimension (64), Matryoshka embeddings significantly outperform normal embeddings in retrieval accuracy. The trade-off is slightly higher query latency but reduced storage requirements.

### Inference Docs

- Normal: `mat testing`
- Matryoshka: `mat testing 2`

---

## 8. Sarvam Bulbul v3 - Text-to-Speech

**Date:** February 16, 2026 | **Status:** In Progress

### Overview

**Model:** `sarvam-bulbul-v3` — A multilingual TTS model tested in English and Tamil.

### Performance Metrics

| Metric | Value |
|--------|-------|
| Avg TTFT (Time to First Token) | 1.392 ms |
| Avg Batch Latency | 4.227 s |

### English Test Results

| Input Text | TTFT (s) | Batch Latency (s) | Issues |
|------------|----------|-------------------|--------|
| Hello! This is a quick speech test. | 1.311 | 4.414 | None |
| Wait... what? | 1.455 | 2.409 | None |
| I said, 'Call me at 7:30,' not 7:13. | 1.238 | 4.038 | None |
| This costs $19.99 — not $199.99. | 1.253 | 7.607 | None |
| Stop. Now continue. Don't rush. | 1.375 | 3.371 | None |
| Your OTP is 9 1 3 0 7 6. | 1.232 | 3.726 | None |
| The total is 32,542 rupees. | 1.311 | 5.004 | None |
| The ratio is 3.14159. | 1.279 | 3.992 | Inconsistent transcription of decimal digits |
| Version 2.1.0 is stable. | 1.311 | 3.704 | None |
| The meeting is on 8 January 2026 at 7:30 PM. | 1.216 | 5.320 | None |
| It happened between 7:29:07 and 7:30:45. | 1.165 | 4.899 | Gaps in time notation |
| It's ₹840/month for 300,000 characters. | 1.302 | 5.437 | None |
| That's 0.28 paise per character. | 1.410 | 4.073 | None |
| Email me at chief.dev+test@gmail.com. | 1.458 | 6.133 | `.com` spelled out instead of pronounced as a word |
| Call +91 98765 43210 for support. | 1.395 | 5.988 | None |
| GPU, CPU, and RAM usage are high. | 1.260 | 4.273 | None |
| The API returned HTTP 502 Bad Gateway. | 1.224 | 4.367 | None |
| We use SaaS tools for CI/CD. | 1.205 | 5.061 | "SaaS" spelled; slash pronounced as "forward slash" |
| LiveKit agents integrate with Supabase functions. | 1.402 | 4.795 | None |
| Deploy it on Nixpacks and monitor with Sentry. | 1.215 | 4.319 | None |
| I read it every day. | 1.278 | 2.371 | "read" pronounced as "red" (past tense) instead of "reed" (present) |
| I read it yesterday. | 0.936 | 1.904 | Correct |
| Please record the call. | 1.243 | 2.386 | None |
| Save the record in the folder. | 1.261 | 2.653 | None |
| (Long security prompt) | 1.241 | 7.417 | None |
| (Connection unstable prompt) | 1.702 | 7.015 | None |
| (Reschedule interview prompt) | 1.351 | 3.685 | None |
| Congratulations! You did an amazing job. | 1.004 | 3.500 | None |
| I'm sorry you had to face that. | 1.295 | 2.769 | None |

### Tamil Test Results

| Input Text | TTFT (s) | Batch Latency (s) | Issues |
|------------|----------|-------------------|--------|
| வணக்கம்! இது ஒரு விரைவு பேச்சு சோதனை. | 1.382 | 4.136 | None |
| ஒரு நிமிடம். என்ன? | 2.085 | 3.818 | None |
| நான் சொன்னது, 7:30க்கு அழை, 7:13க்கு அல்ல. | 1.259 | 4.794 | None |
| இது ₹19.99 — ₹199.99 அல்ல. | 2.081 | 7.494 | "Rupees" pronounced in English instead of Tamil "rubai" |
| நிறுத்து. இப்போது தொடருங்கள். | 1.317 | 4.332 | None |
| உங்கள் OTP 9 1 3 0 7 6. | 1.344 | 4.447 | None |
| மொத்தம் 32,542 ரூபாய். | 1.493 | 5.627 | None |
| விகிதம் 3.14159. | 0.993 | 2.893 | "Point" not translated to Tamil "pulli" |
| பதிப்பு 2.1.0 நிலையானது. | 1.009 | 3.681 | Same "point" pronunciation issue |
| சந்திப்பு 8 ஜனவரி 2026 அன்று மாலை 7:30க்கு. | 1.230 | 5.183 | Year number is spelled out digit by digit |
| அது 7:29:07 முதல் 7:30:45 வரை நடந்தது. | 1.405 | 7.318 | Improper gaps; seconds merged with next word |
| மாதத்திற்கு ₹840 — 300,000 எழுத்துகளுக்கு. | 1.328 | 5.032 | Hyphen/dash range not handled correctly |
| ஒவ்வொரு எழுத்துக்கும் 0.28 பைசா. | 1.550 | 4.405 | None |
| எனக்கு chief.dev+test@gmail.com எழுதுங்கள். | 1.438 | 7.101 | `.com` spelled out (same as English) |
| உதவிக்கு +91 98765 43210 அழைக்கவும். | 2.130 | 8.042 | None |
| வாழ்த்துக்கள்! அருமையான வேலை செய்தீர்கள். | 1.229 | 4.583 | None |
| அதை சந்திக்க வேண்டியிருந்தது வருத்தமாக உள்ளது. | 1.311 | 5.444 | None |

### Known Issues Summary

| Category | Issue | Severity |
|----------|-------|----------|
| Decimals & Numbers | Inconsistent handling of decimal values (3.14159) | Medium |
| Punctuation | Colons and hyphens cause gaps or mispronunciation | Medium |
| Homographs | "read" (present vs past tense) not context-aware | Medium |
| Abbreviations | "SaaS" spelled out; slash read as "forward slash" | Low |
| Email/URLs | `.com` spelled letter-by-letter instead of as a word | Low |
| Tamil: Code-switching | English words used for symbols when Tamil alternatives exist (rupees, point) | Medium |
| Tamil: Numbers | Year spelled digit-by-digit; seconds merged with adjacent words | Medium |
| Tamil: Ranges | Dash/hyphen ranges not recognized properly | Medium |
| Streaming | Audio artifacts present during streaming (`streaming_prob.mp4`) | High |

### Overall Assessment

The model performs well with emotions and simple sentences in both English and Tamil. Weaknesses emerge when handling decimal values, colons, hyphens, and special characters. In Tamil, the model tends to fall back to English pronunciation for symbols even when Tamil equivalents exist.

---

## Summary

| Tool | Category | Verdict |
|------|----------|---------|
| **MemU** | LLM Memory | Useful but struggles with temporal updates |
| **LiveKit** | Voice AI + Telephony | Solid framework, good Twilio integration |
| **Moondream3** | Vision LLM | Cannot run on multi-GPU; single GPU only |
| **Shannon** | AI Pentesting | Delivers real exploits, not just alerts |
| **Decart** | Video Editing | Good for short edits; animation bias in long videos |
| **Primer** | Repo Documentation | Improves AI coding assistant performance |
| **Matryoshka** | Embeddings | Superior to normal models at low dimensions |
| **Sarvam Bulbul v3** | TTS | Good overall; weak on numbers, symbols, and Tamil code-switching |
