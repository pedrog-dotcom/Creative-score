#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_analyze_creatives.py

Objetivo:
- Ler o catálogo gerado pelo download_creatives.py (catalog.csv)
- Ler o score mais recente (creative_score_automation.csv)
- Para cada criativo, gerar uma análise "por IA" do porquê performou bem/mal,
  com foco em:
  - Pausados (HARD STOP e SCORE baixo)
  - Top performers (Top 5 por score)

Importante:
- Este arquivo já organiza os dados (métricas + frames), mas a chamada para IA
  é feita via um "adapter" simples para você conectar no provedor que preferir.

Como configurar:
- AI_PROVIDER = "http" (default) ou "openai_compat" (opcional)
- AI_ENDPOINT_URL  (se AI_PROVIDER=http) -> seu endpoint que recebe prompt + imagens
- AI_API_KEY       (opcional)
- AI_MODEL         (opcional)

Saídas:
- creatives_output/analysis/analysis_results.jsonl  (1 linha por anúncio)
- creatives_output/analysis/summary_ai.txt          (resumo: pausados + top 5)
"""

import os
import json
import csv
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests


OUT_DIR = Path(os.getenv("OUT_DIR", "creatives_output"))
CATALOG_PATH = Path(os.getenv("CATALOG_PATH", str(OUT_DIR / "catalog.csv")))
SCORE_CSV_PATH = Path(os.getenv("SCORE_CSV_PATH", "creative_score_automation.csv"))

AI_PROVIDER = os.getenv("AI_PROVIDER", "http").strip().lower()
AI_ENDPOINT_URL = os.getenv("AI_ENDPOINT_URL", "").strip()
AI_API_KEY = os.getenv("AI_API_KEY", "").strip()
AI_MODEL = os.getenv("AI_MODEL", "").strip()  # opcional

MAX_FRAMES_SEND = int(os.getenv("MAX_FRAMES_SEND", "12"))


def load_csv_dict(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip().replace(".", "").replace(",", ".")
        return float(s)
    except Exception:
        try:
            return float(x)
        except Exception:
            return None


def b64_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def pick_frames(frames_dir: Path, max_n: int) -> List[Path]:
    if not frames_dir.exists():
        return []
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        # fallback: qualquer jpg
        frames = sorted(frames_dir.glob("*.jpg"))
    if len(frames) <= max_n:
        return frames
    # amostragem uniforme
    idxs = [int((i + 1) / (max_n + 1) * len(frames)) for i in range(max_n)]
    idxs = [min(max(i, 0), len(frames) - 1) for i in idxs]
    out = [frames[i] for i in idxs]
    # remove duplicados mantendo ordem
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq


def build_prompt(ad: Dict[str, Any], score_row: Dict[str, Any]) -> str:
    """
    Prompt bem objetivo e consistente, para a IA aprender com o tempo.
    """
    ad_name = ad.get("ad_name", "")
    pause_reason = (ad.get("pause_reason") or "").strip()
    score = score_row.get("performance_score") or ad.get("performance_score") or ""
    spend = score_row.get("spend_std") or ad.get("spend_std") or ""
    share = score_row.get("spend_share") or score_row.get("share") or ""
    cac = score_row.get("cac") or ad.get("cac") or ""
    cac_vs = score_row.get("cpa_vs_avg") or ""

    context = [
        f"Anúncio: {ad_name}",
        f"ad_id: {ad.get('ad_id')}",
        f"Status: {ad.get('effective_status')}",
        f"Motivo pausa (se houver): {pause_reason or 'N/A'}",
        f"Score: {score}",
        f"Investimento (janela): {spend}",
        f"Share de investimento (janela): {share}",
        f"CPA/CAC (janela do score): {cac}",
        f"CPA vs média: {cac_vs}",
    ]

    return (
        "Você é um analista de criativos focado em performance.\n"
        "Analise os frames do criativo e explique, de forma direta e prática:\n"
        "1) Quais elementos visuais e de mensagem explicam performance BOA ou RUIM.\n"
        "2) O que provavelmente aconteceu nos primeiros 2 segundos (hook) e no restante.\n"
        "3) Quais mudanças simples eu faria para melhorar (3 sugestões).\n"
        "4) Classifique este criativo em: TOP / OK / RUIM e justifique.\n\n"
        "Contexto de performance (não invente números além do que está aqui):\n"
        + "\n".join(context)
    )


def call_ai_http(prompt: str, images_b64: List[str]) -> Dict[str, Any]:
    """
    Provider genérico:
    Espera um endpoint seu que aceite:
    {
      "prompt": "...",
      "images": ["<base64 jpg>", ...],
      "model": "..."
    }
    E responda algo como:
    {
      "analysis_text": "...",
      "tags": [...],
      "rating": "TOP|OK|RUIM",
      "key_reasons": [...],
      "improvements": [...]
    }
    """
    if not AI_ENDPOINT_URL:
        # Sem endpoint configurado: devolve análise "pendente"
        return {
            "analysis_text": "AI_ENDPOINT_URL não configurado. Configure para habilitar análise automática.",
            "rating": "N/A",
            "key_reasons": [],
            "improvements": [],
            "tags": [],
        }

    payload = {"prompt": prompt, "images": images_b64}
    if AI_MODEL:
        payload["model"] = AI_MODEL

    headers = {"Content-Type": "application/json"}
    if AI_API_KEY:
        headers["Authorization"] = f"Bearer {AI_API_KEY}"

    r = requests.post(AI_ENDPOINT_URL, headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def main():
    if not CATALOG_PATH.exists():
        raise RuntimeError(f"Catálogo não encontrado: {CATALOG_PATH}. Rode download_creatives.py antes.")

    catalog = load_csv_dict(CATALOG_PATH)
    score_rows = load_csv_dict(SCORE_CSV_PATH)

    # index do score por ad_id (se existir)
    score_by_id: Dict[str, Dict[str, Any]] = {}
    for r in score_rows:
        ad_id = str(r.get("ad_id") or "").strip()
        if ad_id:
            score_by_id[ad_id] = r

    analysis_dir = OUT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = analysis_dir / "analysis_results.jsonl"

    results = []
    for ad in catalog:
        ad_id = str(ad.get("ad_id") or "").strip()
        frames_dir = Path(ad.get("frames_dir") or "")
        frames = pick_frames(frames_dir, MAX_FRAMES_SEND) if frames_dir else []

        images_b64 = []
        for p in frames:
            try:
                images_b64.append(b64_image(p))
            except Exception:
                continue

        score_row = score_by_id.get(ad_id, {})
        prompt = build_prompt(ad, score_row)

        # chama IA (ou gera placeholder)
        ai_out = call_ai_http(prompt, images_b64)

        record = {
            "ad_id": ad_id,
            "ad_name": ad.get("ad_name", ""),
            "effective_status": ad.get("effective_status", ""),
            "pause_reason": ad.get("pause_reason", ""),
            "performance_score": score_row.get("performance_score") or ad.get("performance_score", ""),
            "spend_std": score_row.get("spend_std") or ad.get("spend_std", ""),
            "cac": score_row.get("cac") or ad.get("cac", ""),
            "frames_used": len(images_b64),
            "ai": ai_out,
        }
        results.append(record)

    # escreve jsonl
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # cria um resumo simples
    paused = [r for r in results if str(r.get("pause_reason") or "").strip()]
    # tenta top 5 por score numérico
    def score_num(x):
        try:
            return float(str(x.get("performance_score") or "0").replace(",", "."))
        except Exception:
            return 0.0
    top5 = sorted(results, key=score_num, reverse=True)[:5]

    summary_path = analysis_dir / "summary_ai.txt"
    lines = []
    lines.append("Resumo IA - Criativos")
    lines.append("====================")
    lines.append(f"Total analisados: {len(results)}")
    lines.append(f"Pausados (com reason): {len(paused)}")
    lines.append("")
    lines.append("Pausados (explicação curta):")
    lines.append("----------------------------")
    for r in paused[:50]:
        t = (r.get("ai") or {}).get("analysis_text", "")
        lines.append(f"- {r.get('ad_name')} (score={r.get('performance_score')}, reason={r.get('pause_reason')})")
        if t:
            lines.append(f"  -> {t[:220]}...")
    lines.append("")
    lines.append("Top 5 (explicação curta):")
    lines.append("-------------------------")
    for r in top5:
        t = (r.get("ai") or {}).get("analysis_text", "")
        lines.append(f"- {r.get('ad_name')} (score={r.get('performance_score')})")
        if t:
            lines.append(f"  -> {t[:220]}...")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"✅ Análises salvas em: {out_jsonl}")
    print(f"✅ Resumo salvo em: {summary_path}")
    print("✅ Pronto.")


if __name__ == "__main__":
    main()
