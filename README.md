# greCAPTCHA: Assessing Understanding as Evidence of Research Authorship Under Generative AI

Authors: Justin Payan*, Bálint Gyevnár*, Atoosa Kasirzadeh, Nihar B. Shah; (* equal contribution)

The `main` branch holds information about the research article, the code for running the experiments in the research article, and the anonymized data from the article.
If you want to see the source code for [www.grecaptcha.com](http://www.grecaptcha.com), you should look in the `public_version` branch.

## Summary

greCAPTCHA is a manuscript-specific assessment tool. Researchers upload a
paper and contribution statement, generate a reusable question set through
OpenRouter, and run timed participant assessments with rubric-based feedback.

It supports fill-in-the-blank, multiple-choice, and free-response questions;
saved sets and templates; paired experiments; participant links; CSV export;
and SQLite backups. Answer keys and researcher-only metadata are never sent to
the participant during an assessment.

This repository provides supplementary material for the [paper](https://www.cs.cmu.edu/~nihars/preprints/greCAPTCHA.pdf) with the same title as the repo.
There are two primary artefacts:

1. **Prototype greCAPTCHA:** A web application to host a prototype greCAPTCHA assessment that elicits evidence about whether claimed authors understand their own paper and could take responsibility for them. See below for setup.
2. **Anonymized Participant Data:** An anonymized dataset of 31 participants' experience interacting with this system, found in the `data_anonymized/` folder.

If you use our work please cite it. A bibtex blurb is available [below](#please-cite).

## Local setup

Requirements: Node.js 20+ and either an [OpenRouter](https://openrouter.ai/) API key or an
OpenAI-compatible self-hosted LLM endpoint (for example, vLLM).

```bash
npm install
cp .env.example .env.local
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The database is created under `data/`, and
checked-in Drizzle migrations run automatically on startup.

Minimum `.env.local` configuration:

```dotenv
# Choose one provider, or configure both.
OPENROUTER_API_KEY=your_key_here
# Keep strict validation for production; false permits recoverable local-model mismatches.
# STRICT_MODE=true
# LOCAL_LLM_BASE_URL=http://localhost:8000/v1
# LOCAL_LLM_API_KEY=
# LOCAL_LLM_FORCE_TEXT=0
# LOCAL_LLM_MAX_DOCUMENT_CHARS=60000
DATABASE_URL=./data/research-captcha.db
```

When `LOCAL_LLM_BASE_URL` is configured, models returned by its
`GET /v1/models` endpoint appear alongside OpenRouter models. The endpoint must implement
`POST /v1/chat/completions` and JSON-schema response formats. Self-hosted model IDs are
prefixed with `self-hosted:` internally so they cannot collide with OpenRouter IDs.

OpenRouter's `openrouter/free` router is also available in the model picker whenever an
OpenRouter API key is configured. It automatically chooses an available free model. No free
model accepts files natively, so pair it with a parser engine (Cloudflare AI, Mistral OCR or local
text extraction); the generation screen warns when "Native model processing" is selected for any
model without file input. Only some free models enforce JSON schemas, so requests ask OpenRouter
to route only to providers supporting every request parameter. When a reply still does not match
the expected format, the error names the model the router picked. Because the underlying model
can change, generated output quality and supported capabilities may vary.

Self-hosted endpoints receive the PDF directly first. If the endpoint rejects that request with an
HTTP error, greCAPTCHA extracts selectable text locally and retries without the PDF; timeouts and
malformed model output are reported as errors rather than retried. Set `LOCAL_LLM_FORCE_TEXT=1`
to skip the first attempt when the endpoint is known to be text-only. Choosing the "Local text
extraction" PDF engine does the same for any model, OpenRouter and direct Gemini included: only
the extracted text is sent. Scanned PDFs still require OCR.

Generation validation is controlled by `STRICT_MODE`, which defaults to `true`. With
`STRICT_MODE=false`, mismatched question counts, option counts, and free-response rubric totals
are logged as warnings and the model's output is used where it remains valid: surplus questions
and options are cut to the configured counts, malformed items are dropped, and a card left with no
usable question still fails generation. This is intended
for testing small local models, not for production study data.

### Local Qwen2.5 3B test server

The repository includes a llama.cpp launcher for Qwen2.5 3B Instruct, selected to provide
better generation quality while remaining practical on an M2 Mac with 8 GB of unified memory:

```bash
python3 -m pip install "llama-cpp-python[server]" huggingface-hub
python3 scripts/test_server.py
```

The launcher downloads the Q4_K_M GGUF model on first start (about 2 GB), stores it under
`models/`, and serves it at `http://127.0.0.1:8000/v1`. It uses the model's full native
32k-token context with an 8-bit (`q8_0`) KV cache and a single server slot. Measured on an 8 GB
M2, that costs about 0.7 GB beyond the ~1.8 GB of weights. Override with `LOCAL_LLM_CONTEXT` and
`LOCAL_LLM_KV_TYPE` (`f16`, `q8_0` or `q4_0`) in the launcher's environment. The single slot
keeps the manuscript cached between cards, so only the first card of a set pays to read it:
about a minute for a 60,000-character manuscript on that machine. Self-hosted requests time out
after `LOCAL_LLM_TIMEOUT_SECONDS` (default 600). The local `.env.local`
configuration points greCAPTCHA at this endpoint. Restart the Next.js app after changing
environment variables.

Qwen2.5 3B Instruct is text-only. The local configuration therefore sets
`LOCAL_LLM_FORCE_TEXT=1`, avoiding a failed native-PDF request. `LOCAL_LLM_MAX_DOCUMENT_CHARS`
limits the manuscript size sent to the model; increase it only if the model's context window
allows it.

For a deployment, also set:

```dotenv
RESEARCHER_PASSWORD=use-a-long-random-password
PUBLIC_BASE_URL=https://your-public-hostname.example
```

`GEMINI_API_KEY` is optional. When set, Gemini models are sent directly to the
Google API; otherwise they use OpenRouter. See `.env.example` for backup
settings.

## Researcher workflow

1. Sign in at `/login`.
2. Upload a PDF and enter the contribution statement.
3. Choose a model and PDF extractor, then configure question cards. Cards can
  be fill-in-the-blank, multiple-choice, or free response, with optional
   prompts, card names, warm-up status, and soft time limits.
4. Generate and review the question set. Saved sets can be renamed, inspected,
  reused, or deleted.
5. Create an attempt from a saved set, or create a paired experiment with an
  own-paper and unfamiliar-paper block.
6. Use the assessment plan to start an in-person session or copy a participant
  link. New participant links are disabled until they are ready to use.
7. Export responses with **Export all attempts as CSV**.

The plan page is researcher-only and includes item descriptions and progress.
The participant sees only the current question. Submitted answers are locked;
questions may be skipped, and skipped or timed-out items remain distinct from
ordinary wrong answers in the review and export.

## Assessment behavior

- A landing page appears before each question set. The first question's clock
starts only when the participant presses **Start**.
- Per-question limits are soft: they record overruns but never cut off an
answer. An optional overall set limit is enforced server-side.
- Warm-up questions appear first and are excluded from the overall score.
- Fill-in-the-blank and multiple-choice items are graded deterministically;
free responses are graded in grouped model calls using their rubrics.
- Questions, options, rubrics, feedback, and answers support LaTeX through
KaTeX. Malformed math is shown as source text.
- Experiments counterbalance paper order and unfamiliar-paper stratum. A
chained experiment link runs both blocks and reveals results together.



## Access and data

The researcher interface requires `RESEARCHER_PASSWORD` in production.
Participant links are capability URLs, so treat them as sensitive. The server
keeps API keys and answer keys private, and timing is recorded authoritatively
on the server.

The app stores research data in SQLite at `DATABASE_URL`. Exports contain
submitted responses, scores, answer keys, and timing data; handle them under
the study's retention and privacy requirements. This MVP has one researcher
password rather than per-user accounts, and does not provide rate limiting or
an audit log.

Enable backups with:

```dotenv
BACKUP_DIR=./backups
BACKUP_INTERVAL_MINUTES=60
BACKUP_KEEP=48
```

Backups include a consistent database snapshot and CSV export. For a quick
manual local backup:

```bash
sqlite3 data/research-captcha.db ".backup 'backup-$(date +%F).db'"
```



## Production and Cloudflare Tunnel

The app can run locally while Cloudflare Tunnel provides the public participant
hostname. Set `PUBLIC_BASE_URL` to that hostname so copied links are usable by
participants.

```bash
npm ci
npm run build
npm run start:tunnel
cloudflared tunnel run research-captcha
```

Use `cloudflared tunnel login`, create a tunnel, route its DNS hostname, and
configure `cloudflared/config.example.yml` as described by Cloudflare. Run the
researcher dashboard directly on `127.0.0.1:3000`; long PDF generation and
grading requests can exceed Cloudflare's proxy timeout. The
`npm run start:tunnel:verbose` command enables request logging without logging
request bodies.

## Development commands

```bash
npm run dev          # development server
npm run build        # production build
npm run start        # run a production build
npm run db:generate  # generate a migration after changing the schema
npm run db:migrate   # apply pending migrations explicitly
npm run db:push      # push the Drizzle schema manually
npx drizzle-kit studio
```

Schema definitions live in `src/db/schema.ts`; checked-in migrations live in
`drizzle/`. To reset local data, stop the server and remove the database plus
its `-wal` and `-shm` files. This permanently deletes saved sets, attempts, and
answers.

## Please cite

If you use our work please use the following citation:

```
@misc{payan2026grecaptcha,
  title         = {{greCAPTCHA}: Assessing Understanding as Evidence of Research Authorship Under Generative {AI}},
  author        = {Payan, Justin and Gyevn{\'a}r, B{\'a}lint and Kasirzadeh, Atoosa and Shah, Nihar B.},
  year          = {2026},
  eprint        = {2609.20481},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DL},
  url           = {https://arxiv.org/abs/2609.20481},
  note          = {Justin Payan and B{\'a}lint Gyevn{\'a}r contributed equally.}
}
```
