"""Generate 150 EN personas for Demo Day 2026 (broad-IT/AI/ML profile).

Образец полей и стиля — `data/personas/personas_mobius_en.json`.
В отличие от Mobius (мобильная разработка), Demo Day шире по доменам:
NLP, LLM, ML, CV, RecSys, Agents, EdTech, Fintech, Industrial ML, MLOps,
ASR, Security, Analytics. Distribution:

  experience:    junior 8%, middle 30%, senior 44%, lead/principal 18%   (12/45/66/27)
  company_size:  smallish-startup 18%, midsize 42%, large 28%, enterprise 12%   (27/63/42/18)

Генерация порционная: каждый LLM-вызов выдаёт 5 персон с явными
заданными role/experience/company_size/preferred_topics, чтобы контроль
распределения был жёсткий и стиль не плавал.

Запуск:
    cd experiments && source .venv/bin/activate
    python scripts/generate_demoday_personas_en.py --concurrency 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_CANDIDATES = [ROOT.parent / ".env", ROOT.parent.parent / "party-of-one" / ".env"]
OUT_PATH = ROOT / "data" / "personas" / "personas_demoday_en.json"
PARTIAL_PATH = ROOT / "data" / "personas" / "personas_demoday_en.partial.jsonl"

PRICING = {
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "openai/gpt-4.1-mini": (0.40, 1.60),
}

# Целевое распределение
N_TOTAL = 150
EXP_TARGETS = [
    ("junior", 12),           # 8%
    ("middle", 45),           # 30%
    ("senior", 66),           # 44%
    ("lead/principal", 27),   # 18%
]
CO_SIZE_TARGETS = [
    ("smallish-startup", 27), # 18%
    ("midsize", 63),          # 42%
    ("large", 42),            # 28%
    ("enterprise", 18),       # 12%
]

# Domain buckets: каждая категория = роль(и) + интересы.
# 150 персон делятся на следующие группы: пропорции основаны на Demo Day
# top-category counts (NLP/ML/LLM/Agents/CV/RecSys и т.д.).
ROLE_BUCKETS = [
    # (key, n, roles_pool, interests_pool, preferred_topics_pool)
    ("nlp_llm", 28,
        ["NLP Engineer", "LLM Engineer", "Applied NLP Researcher",
         "LLM Platform Engineer", "Conversational AI Engineer",
         "Search and Retrieval Engineer"],
        ["RAG", "LLM fine-tuning", "Prompt engineering", "Embeddings",
         "Retrieval", "Information extraction", "Function calling",
         "Multilingual NLP", "Question answering", "Classification",
         "Text generation", "vLLM", "LangChain", "LlamaIndex"],
        ["NLP", "LLM", "RAG", "Search"]),
    ("agents", 22,
        ["AI Agent Engineer", "Autonomous Systems Researcher",
         "LLM Agent Developer", "Multi-Agent Systems Engineer"],
        ["LLM agents", "Multi-agent orchestration", "Tool use",
         "Function calling", "Browser agents", "Code agents",
         "Reasoning loops", "Agent evaluation", "AutoGen",
         "LangGraph", "OpenAI Assistants", "Memory and planning"],
        ["Agents", "Autonomous agents", "LLM"]),
    ("cv", 16,
        ["Computer Vision Engineer", "ML Engineer (CV)",
         "Applied CV Researcher", "Vision Platform Engineer"],
        ["Object detection", "Segmentation", "OCR", "Multimodal vision",
         "Diffusion models", "3D reconstruction", "Pose estimation",
         "Video understanding", "TensorRT", "ONNX", "Edge deployment",
         "Quantisation"],
        ["Computer Vision", "CV", "OCR"]),
    ("recsys", 14,
        ["Recommender Systems Engineer", "Search and Ranking Engineer",
         "Personalization ML Engineer", "Applied ML Engineer (RecSys)"],
        ["Two-tower models", "Candidate generation", "Ranking",
         "Cold start", "Sequence models", "Multi-stage ranking",
         "Implicit feedback", "Embeddings", "Diversity",
         "Catboost", "Reranker training", "Evaluation"],
        ["Recsys", "RecSys", "Personalization", "Search"]),
    ("ml_industrial", 14,
        ["Senior ML Engineer", "Applied ML Engineer", "ML Platform Engineer",
         "Industrial ML Engineer", "Production ML Engineer"],
        ["Tabular ML", "Gradient boosting", "Feature engineering",
         "AutoML", "Model monitoring", "A/B testing",
         "Streaming inference", "Time-series", "Demand forecasting",
         "Anomaly detection"],
        ["Industrial ML", "ML", "Analytics"]),
    ("mlops", 10,
        ["MLOps Engineer", "ML Platform Engineer",
         "ML Infrastructure Engineer", "DevOps for ML"],
        ["Kubeflow", "MLflow", "Feature store", "Model registry",
         "Triton Inference Server", "Kubernetes", "GPU scheduling",
         "Distributed training", "Data versioning", "ETL", "CI/CD for ML"],
        ["MLOps", "Industrial ML"]),
    ("fintech_ml", 12,
        ["Senior ML Engineer (Fintech)", "Quant ML Engineer",
         "Credit Scoring ML Engineer", "Fraud Detection ML Engineer",
         "Risk Modelling Engineer"],
        ["Credit scoring", "Fraud detection", "Anti-money-laundering",
         "Survival analysis", "Uplift modelling",
         "Behavioural scoring", "Transactional sequence models"],
        ["Fintech", "ML", "Industrial ML"]),
    ("asr_speech", 10,
        ["ASR Engineer", "Speech Recognition Researcher",
         "Speech AI Engineer", "Conversational Voice Engineer"],
        ["Whisper", "Conformer", "Streaming ASR", "Speaker diarization",
         "TTS", "Voice cloning", "Speech-to-text", "VAD",
         "Acoustic models", "Endpoint detection"],
        ["ASR", "TTS", "NLP"]),
    ("edtech", 8,
        ["AI for Education Engineer", "EdTech Product ML Engineer",
         "Learning Analytics Engineer", "AI Tutor Engineer"],
        ["Adaptive learning", "Knowledge tracing",
         "Intelligent tutoring", "Curriculum design",
         "Automated grading", "Student modelling"],
        ["EdTech", "NLP", "LLM"]),
    ("security_ai", 6,
        ["AI Security Researcher", "LLM Red Team Engineer",
         "ML Security Engineer", "AI Safety Engineer"],
        ["LLM red teaming", "Adversarial ML",
         "Prompt injection", "Model jailbreaks",
         "Privacy-preserving ML", "Differential privacy"],
        ["Security", "LLM", "AI"]),
    ("analytics_ds", 10,
        ["Data Scientist", "Analytics Engineer",
         "Senior Analyst", "Causal Inference Specialist",
         "Product Data Scientist"],
        ["Causal inference", "A/B testing", "Cohort analysis",
         "Retention modelling", "DBT", "Looker",
         "ClickHouse", "Experimentation platform"],
        ["Analytics", "ML"]),
]
assert sum(b[1] for b in ROLE_BUCKETS) == N_TOTAL, sum(b[1] for b in ROLE_BUCKETS)


SYSTEM_PROMPT = (
    "You generate fully fleshed-out conference attendee personas for a "
    "broad-IT conference (NLP, ML, LLM, CV, Agents, RecSys, EdTech, "
    "Fintech, Industrial ML, MLOps, ASR, AI Security, Analytics). "
    "Personas must be in English, technical and concrete. "
    "Style reference: short technical paragraph in third person, factual, "
    "mentions specific tools/stacks/libraries/use cases the persona owns. "
    "Avoid generic marketing fluff. Each persona must be unique in its "
    "narrative.\n\n"
    "STRICT OUTPUT FORMAT: a JSON array of N objects, no commentary, "
    "no markdown, no code fences. Each object MUST have keys:\n"
    "  id (string, exactly as given in input),\n"
    "  role (string),\n"
    "  experience (string, exactly one of: junior, middle, senior, lead/principal),\n"
    "  company_size (string, exactly one of: smallish-startup, midsize, "
    "large, enterprise),\n"
    "  interests (array of 3-5 short strings, technical tags),\n"
    "  preferred_topics (array of 2-4 short conference-track tags),\n"
    "  background (string, 4-7 sentences in English, third person, "
    "specific tools/stacks/use cases this persona owns or wants to learn).\n\n"
    "RULES:\n"
    "1. Use EXACTLY the id, role, experience, company_size, and "
    "preferred_topics from the input slot. Do not invent fields.\n"
    "2. interests must be 3-5 specific technical concepts/libraries.\n"
    "3. background must be ~5 sentences, third person, concrete; mention "
    "team size or company type when relevant, a current technical project, "
    "and what they want to learn at this conference.\n"
    "4. Avoid markdown formatting and avoid first person.\n"
    "5. Personas across the batch must differ in their narratives "
    "(different companies, different specific projects)."
)


def load_api_key() -> str:
    for env in ENV_CANDIDATES:
        if env.exists():
            cfg = dotenv_values(env)
            k = cfg.get("OPENROUTER_API_KEY")
            if k:
                return k
    raise SystemExit("OPENROUTER_API_KEY not found in .env")


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = [ln for ln in lines if not ln.startswith("```")]
        s = "\n".join(lines).strip()
    return s


def design_slots() -> list[dict]:
    """Полное распределение 150 слотов по: bucket × experience × company_size."""
    rng = random.Random(20260512)
    # 1. Generate ordered experience list (12 / 45 / 66 / 27)
    exp_pool = []
    for exp, n in EXP_TARGETS:
        exp_pool.extend([exp] * n)
    rng.shuffle(exp_pool)
    # 2. Generate ordered company_size list (27 / 63 / 42 / 18)
    co_pool = []
    for co, n in CO_SIZE_TARGETS:
        co_pool.extend([co] * n)
    rng.shuffle(co_pool)
    # 3. Generate ordered bucket list per ROLE_BUCKETS counts
    bucket_pool = []
    for b in ROLE_BUCKETS:
        bucket_pool.extend([b] * b[1])
    rng.shuffle(bucket_pool)

    slots = []
    for i in range(N_TOTAL):
        b = bucket_pool[i]
        # pick a role from bucket randomly (rng deterministic)
        role = rng.choice(b[2])
        # pick 2-3 preferred_topics from bucket pool
        pref = rng.sample(b[4], k=min(len(b[4]), rng.choice([2, 3])))
        # pick 3-4 interests
        n_int = rng.choice([3, 4, 4, 5])
        n_int = min(n_int, len(b[3]))
        interests = rng.sample(b[3], k=n_int)
        slots.append({
            "id": f"p_demoday_{i + 1:03d}",
            "role": role,
            "experience": exp_pool[i],
            "company_size": co_pool[i],
            "interests": interests,
            "preferred_topics": pref,
            "bucket": b[0],  # internal hint to LLM
        })
    return slots


async def gen_batch(client, model, slots_batch: list[dict], sem, pricing) -> tuple[list[dict], float]:
    user_msg = (
        "Generate background descriptions for these "
        f"{len(slots_batch)} personas. Use the supplied id/role/experience/"
        "company_size/interests/preferred_topics EXACTLY; only fill in "
        "the background. Bucket hints (for the kind of stack/projects "
        "to mention) are included but DROP the 'bucket' key in your output. "
        "Return JSON array.\n\n"
        f"INPUT:\n{json.dumps(slots_batch, ensure_ascii=False)}"
    )
    async with sem:
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.6,
                        max_tokens=4000,
                        timeout=120,
                    ),
                    timeout=150.0,
                )
                msg = resp.choices[0].message.content or ""
                u = resp.usage
                cost = 0.0
                if u is not None:
                    p_in, p_out = pricing
                    cost = (u.prompt_tokens / 1e6) * p_in + (u.completion_tokens / 1e6) * p_out
                txt = _strip_fences(msg)
                lo, hi = txt.find("["), txt.rfind("]")
                if lo == -1 or hi == -1:
                    raise ValueError(f"no JSON array in response: {msg[:200]}")
                arr = json.loads(txt[lo:hi + 1])
                if not isinstance(arr, list) or len(arr) != len(slots_batch):
                    raise ValueError(
                        f"expected {len(slots_batch)} personas, got "
                        f"{len(arr) if isinstance(arr, list) else type(arr).__name__}"
                    )
                # Sanity merge: enforce id/role/experience/company_size/interests/preferred_topics from input
                # and only accept background from LLM.
                out = []
                for slot, gen in zip(slots_batch, arr):
                    if not isinstance(gen, dict):
                        raise ValueError(f"item is not dict: {gen!r}")
                    bg = str(gen.get("background", "")).strip()
                    if len(bg) < 80:
                        raise ValueError(f"background too short for {slot['id']}: {bg!r}")
                    out.append({
                        "id": slot["id"],
                        "role": slot["role"],
                        "experience": slot["experience"],
                        "company_size": slot["company_size"],
                        "interests": slot["interests"],
                        "preferred_topics": slot["preferred_topics"],
                        "background": bg,
                    })
                return out, cost
            except Exception as e:
                if attempt == 2:
                    print(f"  FAILED batch ({slots_batch[0]['id']}..): {e}")
                    return [], 0.0
                await asyncio.sleep(2.0 * (attempt + 1))


async def main_async(args):
    slots = design_slots()
    print(f"Designed {len(slots)} persona slots")
    # Print summary
    from collections import Counter
    exp_c = Counter(s["experience"] for s in slots)
    co_c = Counter(s["company_size"] for s in slots)
    bk_c = Counter(s["bucket"] for s in slots)
    rl_c = Counter(s["role"] for s in slots)
    print("  experience:", dict(exp_c))
    print("  company_size:", dict(co_c))
    print("  bucket:", dict(bk_c))
    print("  distinct roles:", len(rl_c))
    print()

    # Resume support
    done: dict[str, dict] = {}
    if PARTIAL_PATH.exists():
        with open(PARTIAL_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["id"]] = rec
                except Exception:
                    continue
        print(f"  resumed: {len(done)}/{len(slots)} already generated")

    # Batch slots into groups of args.batch_size
    todo = [s for s in slots if s["id"] not in done]
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    print(f"  to generate: {len(todo)} personas in {len(batches)} batches")

    api_key = load_api_key()
    pricing = PRICING.get(args.model, (1.0, 5.0))
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    total_cost = 0.0
    t0 = time.perf_counter()

    pbar = tqdm(total=len(todo), desc="personas", smoothing=0.1)
    fpartial = open(PARTIAL_PATH, "a", encoding="utf-8")

    async def worker(batch):
        nonlocal total_cost
        out, cost = await gen_batch(client, args.model, batch, sem, pricing)
        total_cost += cost
        for p in out:
            done[p["id"]] = p
            fpartial.write(json.dumps(p, ensure_ascii=False) + "\n")
            fpartial.flush()
            pbar.update(1)
        pbar.set_postfix({"cost": f"${total_cost:.4f}"})

    await asyncio.gather(*[worker(b) for b in batches])
    pbar.close()
    fpartial.close()
    print(f"  cost: ${total_cost:.4f}, wall: {time.perf_counter() - t0:.1f}s")

    # Write final ordered output
    out_list = []
    missing = 0
    for s in slots:
        if s["id"] in done:
            out_list.append(done[s["id"]])
        else:
            missing += 1
            print(f"  MISSING {s['id']}")
    if missing:
        print(f"  WARNING: {missing} slots missing")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f"WROTE {OUT_PATH}  ({len(out_list)} personas)")
    return total_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=5)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
