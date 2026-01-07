#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_analyze_creatives_v3_final.py
- Corrigido: URL da OpenAI (chat/completions)
- Corrigido: Payload de visão (messages + image_url)
- Corrigido: Match de ad_id (string normalization)
- Corrigido: Inclusão de anúncios ACTIVE na análise
"""

import os
import json
import math
import hashlib
import re
import logging
import base64
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------
# CONFIG
# ---------------------------
CATALOG_PATH = os.getenv("CREATIVES_CATALOG_PATH", "creatives_output/catalog.csv")
SCORE_PATH = os.getenv("SCORE_LATEST_PATH", "creative_score_automation.csv")
CACHE_DIR = Path(os.getenv("AI_CACHE_DIR", "ai_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "analysis_cache.jsonl"
OUT_DIR = Path(os.getenv("AI_OUT_DIR", "creatives_output/analysis"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / "analysis_results.jsonl"
SUMMARY_PATH = OUT_DIR / "summary_ai.txt"
MAX_FRAMES = int(os.getenv("AI_MAX_FRAMES", "12"))
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower()
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_ENDPOINT_URL = os.getenv("AI_ENDPOINT_URL", "").strip()
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o")
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT_SECONDS", "120"))

NAME_FIELDS = [
    "AdCode", "AdName", "Variation", "Month live", "Market", "Asset type",
    "Channel", "Partnership Ad?", "Persona", "Actor/Affiliate/Expert",
    "Resource", "Core Message", "Leave Blank", "Tone", "Anchor",
    "Overarching", "Value Prop", "Key Outcome", "Production", "Style/Setting",
]

def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def money_br(x):
    try:
        x = float(x)
        if math.isnan(x): return "R$ 0,00"
        s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {s}"
    except: return "R$ 0,00"

def pct(x):
    try:
        x = float(x)
        if math.isnan(x): return "0.00%"
        return f"{x*100:.2f}%"
    except: return "0.00%"

def normalize_name(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"_+", "_", name)
    return name

def parse_ad_name(ad_name: str) -> dict:
    ad_name = normalize_name(ad_name)
    parts = ad_name.split("_")
    out = {}
    for i, field in enumerate(NAME_FIELDS):
        if i < len(parts):
            out[field] = parts[i]
        else:
            out[field] = ""
    return out

def load_jsonl(path: Path) -> list:
    if not path.exists(): return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except: continue
    return rows

def append_jsonl(path: Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def pick_frames(frames_dir: Path, max_frames: int) -> list:
    if not frames_dir.exists(): return []
    frames = sorted([p for p in frames_dir.glob("*.jpg") if p.is_file()])
    if not frames: return []
    if len(frames) <= max_frames: return [str(p) for p in frames]
    step = max(1, len(frames) // max_frames)
    idxs = list(range(0, len(frames), step))[:max_frames]
    return [str(frames[i]) for i in idxs]

def openai_request(payload: dict) -> dict:
    url = AI_ENDPOINT_URL or "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI_API_KEY}",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=AI_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def build_openai_payload(prompt: str, frame_paths: list) -> dict:
    content = [{"type": "text", "text": prompt}]
    for p in frame_paths:
        try:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
            })
        except Exception as e:
            logger.error(f"Erro ao ler frame {p}: {e}")
    
    return {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
    }

def extract_json_from_openai(resp_json: dict) -> dict:
    try:
        content = resp_json["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        logger.error(f"Erro ao extrair JSON: {e}")
        raise RuntimeError(f"Falha na resposta da IA: {e}")

def main():
    run_at = now_utc_iso()
    catalog_path = Path(CATALOG_PATH)
    if not catalog_path.exists():
        logger.error(f"Catálogo não encontrado: {CATALOG_PATH}")
        return

    if not Path(SCORE_PATH).exists():
        logger.error(f"Score não encontrado: {SCORE_PATH}")
        return

    catalog = pd.read_csv(catalog_path)
    score = pd.read_csv(SCORE_PATH)
    
    catalog["ad_id"] = catalog["ad_id"].astype(str)
    score["ad_id"] = score["ad_id"].astype(str)
    score_idx = score.set_index("ad_id", drop=False)

    cache_rows = load_jsonl(CACHE_PATH)
    cache_idx = {(r.get("creative_key"), r.get("media_sha256")): r for r in cache_rows}

    analyzed, skipped, failures = 0, 0, 0
    paused_rows, top_rows, active_rows = [], [], []

    total_spend = score["spend_std"].sum() if "spend_std" in score.columns else 0

    for _, row in catalog.iterrows():
        ad_id = str(row["ad_id"])
        local_path = Path(str(row["local_path"]))
        if not local_path.exists(): continue

        ad_name = str(row.get("ad_name", ""))
        creative_key = normalize_name(ad_name)
        media_sha = sha256_file(local_path)

        if (creative_key, media_sha) in cache_idx:
            skipped += 1
            continue

        perf_row = score_idx.loc[ad_id] if ad_id in score_idx.index else None
        label = "ACTIVE"
        if perf_row is not None:
            reason = str(perf_row.get("pause_reason", "")).upper()
            if "HARD" in reason or "SCORE" in reason: label = "PAUSED"
            if str(perf_row.get("is_top5", "")).lower() in ("1", "true"): label = "TOP_PERFORMER"

        perf = {
            "label": label,
            "score": round(float(perf_row["performance_score"]), 2) if perf_row is not None and not pd.isna(perf_row["performance_score"]) else None,
            "spend_br": money_br(perf_row["spend_std"]) if perf_row is not None else "R$ 0,00",
            "cpa_br": money_br(perf_row["cac"]) if perf_row is not None else "N/A",
        }

        frames_dir = Path(str(row.get("frames_dir", "")))
        if not frames_dir or not frames_dir.exists():
            frames_dir = Path("creatives_output/frames") / ad_id

        frame_paths = pick_frames(frames_dir, MAX_FRAMES)
        context = {
            "creative_key": creative_key,
            "ad_id": ad_id,
            "performance": perf,
            "name_fields": parse_ad_name(ad_name)
        }

        try:
            if AI_PROVIDER == "openai":
                payload = build_openai_payload(f"Analise este criativo de Meta Ads. Contexto: {json.dumps(context)}", frame_paths)
                resp = openai_request(payload)
                result = {**context, "analysis": extract_json_from_openai(resp), "media_sha256": media_sha}
                append_jsonl(RESULTS_PATH, result)
                append_jsonl(CACHE_PATH, {**result, "cached_at": run_at})
                analyzed += 1
                if label == "PAUSED": paused_rows.append(result)
                elif label == "TOP_PERFORMER": top_rows.append(result)
                else: active_rows.append(result)
        except Exception as e:
            failures += 1
            logger.error(f"Falha no ad_id {ad_id}: {e}")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(f"IA - Resumo ({run_at})\nNovos: {analyzed} | Cache: {skipped} | Falhas: {failures}\n")
    
    logger.info(f"Finalizado: {analyzed} analisados, {failures} falhas.")

if __name__ == "__main__":
    main()
