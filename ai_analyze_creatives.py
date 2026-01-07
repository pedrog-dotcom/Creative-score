#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_analyze_creatives_v2.py

V2: análise robusta por IA + saída estruturada (JSON) + deduplicação (não analisar vídeo 2x)
- Lê creatives_output/catalog.csv (gerado pelo download_creatives.py)
- Lê creative_score_automation.csv (gerado pelo score)
- Monta o "contexto de performance" (score, CPA, share spend, pausado/top5 etc.)
- Extrai campos do nome do anúncio (separados por "_")
- Envia para IA com resposta em JSON estruturado
- Cacheia por "creative_key" (nome normalizado) + "media_sha256" para NÃO reanalisar o mesmo vídeo
- Gera:
    creatives_output/analysis/analysis_results.jsonl
    creatives_output/analysis/summary_ai.txt
    ai_cache/analysis_cache.jsonl   (cache persistente recomendado)
"""

import os
import json
import math
import hashlib
import re
import logging
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
# Onde estão os arquivos gerados pelos outros passos
CATALOG_PATH = os.getenv("CREATIVES_CATALOG_PATH", "creatives_output/catalog.csv")
SCORE_PATH = os.getenv("SCORE_LATEST_PATH", "creative_score_automation.csv")

# Cache persistente (para evitar reanalisar vídeo)
CACHE_DIR = Path(os.getenv("AI_CACHE_DIR", "ai_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "analysis_cache.jsonl"

# Saídas desta etapa
OUT_DIR = Path(os.getenv("AI_OUT_DIR", "creatives_output/analysis"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / "analysis_results.jsonl"
SUMMARY_PATH = OUT_DIR / "summary_ai.txt"

# Quantidade máxima de frames que vamos mandar por criativo
MAX_FRAMES = int(os.getenv("AI_MAX_FRAMES", "16"))

# Provider/endpoint
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower()  # openai | http | none
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_ENDPOINT_URL = os.getenv("AI_ENDPOINT_URL", "").strip()
AI_MODEL = os.getenv("AI_MODEL", "gpt-5")  # default para OpenAI (pode trocar)
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT_SECONDS", "120"))

# ---------------------------
# Helpers
# ---------------------------
NAME_FIELDS = [
    "AdCode",
    "AdName",
    "Variation",
    "Month live",
    "Market",
    "Asset type",
    "Channel",
    "Partnership Ad?",
    "Persona",
    "Actor/Affiliate/Expert",
    "Resource",
    "Core Message",
    "Leave Blank",
    "Tone",
    "Anchor",
    "Overarching",
    "Value Prop",
    "Key Outcome",
    "Production",
    "Style/Setting",
]

def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def money_br(x):
    x = safe_float(x, 0.0)
    s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

def pct(x):
    x = safe_float(x, 0.0)
    return f"{x*100:.2f}%"

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    name = name.strip()
    # normaliza espaços e múltiplos underscores
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"_+", "_", name)
    return name

def parse_ad_name(ad_name: str) -> dict:
    """Parseia campos pelo '_' na ordem definida.
    Se vier com menos campos, preenche o resto com vazio.
    Se vier com mais, junta o excedente no último campo.
    """
    ad_name = normalize_name(ad_name)
    parts = ad_name.split("_")
    out = {}
    if len(parts) <= len(NAME_FIELDS):
        for i, field in enumerate(NAME_FIELDS):
            out[field] = parts[i] if i < len(parts) else ""
    else:
        # excedente vai para o último campo (Style/Setting)
        for i, field in enumerate(NAME_FIELDS[:-1]):
            out[field] = parts[i]
        out[NAME_FIELDS[-1]] = "_".join(parts[len(NAME_FIELDS)-1:])
    return out

def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def append_jsonl(path: Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def cache_index(cache_rows: list) -> dict:
    """Indexa cache por (creative_key, media_sha256)."""
    idx = {}
    for r in cache_rows:
        key = (r.get("creative_key"), r.get("media_sha256"))
        if key[0] and key[1]:
            idx[key] = r
    return idx

def pick_frames(frames_dir: Path, max_frames: int) -> list:
    """Seleciona até max_frames, pegando do começo, meio e fim quando possível."""
    if not frames_dir.exists():
        return []
    frames = sorted([p for p in frames_dir.glob("*.jpg") if p.is_file()])
    if not frames:
        return []
    if len(frames) <= max_frames:
        return [str(p) for p in frames]

    # amostra distribuída
    idxs = [0, len(frames)//2, len(frames)-1]
    # completa com passo
    step = max(1, len(frames) // max_frames)
    idxs += list(range(0, len(frames), step))
    idxs = sorted(set(idxs))[:max_frames]
    return [str(frames[i]) for i in idxs]

# ---------------------------
# IA Calls
# ---------------------------
def openai_request(payload: dict) -> dict:
    """Chama OpenAI Responses API via HTTP requests (sem SDK)."""
    url = AI_ENDPOINT_URL or "https://api.openai.com/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI_API_KEY}",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=AI_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def http_request(payload: dict) -> dict:
    """Chama um endpoint genérico HTTP (se você tiver seu próprio serviço)."""
    if not AI_ENDPOINT_URL:
        raise RuntimeError("AI_ENDPOINT_URL não definido (provider=http)")
    headers = {"Content-Type": "application/json"}
    if AI_API_KEY:
        headers["Authorization"] = f"Bearer {AI_API_KEY}"
    resp = requests.post(AI_ENDPOINT_URL, headers=headers, json=payload, timeout=AI_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def build_prompt_structured(context: dict) -> str:
    """
    Prompt noob-friendly e orientado a JSON.
    A IA deve responder SOMENTE com um JSON válido (sem texto fora).
    """
    # Contexto resumido (evita prompt enorme)
    perf = context["performance"]
    parsed = context["name_fields"]

    return f"""
Você é um(a) especialista em criativos de performance (Meta Ads) para e-commerce. 
Analise o criativo e explique seu potencial de performance ou o motivo de sua performance atual, com base em sinais visuais e métricas.

IMPORTANTE:
- Responda SOMENTE com um JSON válido (nada de texto fora do JSON).
- Seja objetivo e didático.
- Use linguagem simples (sem jargões).
- Use evidências visuais (ex.: "texto pequeno", "oferta aparece tarde", "hook fraco").

CONTEXTO DO ANÚNCIO (nome quebrado em campos):
{json.dumps(parsed, ensure_ascii=False)}

PERFORMANCE (últimos 21 dias):
- score: {perf.get("score")}
- motivo_status: {perf.get("label")}
- spend: {perf.get("spend_br")}
- share_spend: {perf.get("share_spend_pct")}
- CPA (1d click): {perf.get("cpa_br")}
- CPA vs média: {perf.get("cpa_vs_avg")}
- CTR: {perf.get("ctr")}
- connect_rate: {perf.get("connect_rate")}
- bounce_rate: {perf.get("bounce_rate")}
- cost_per_checkout: {perf.get("cost_per_checkout_br")}

TAREFA:
1) Diagnóstico visual: o que ajuda / o que atrapalha (hook, clareza, oferta, prova, ritmo, legibilidade, CTA).
2) Diagnóstico de Performance: Por que este criativo é TOP, PAUSADO ou por que ainda está ATIVO (conecte o diagnóstico com métricas).
3) Sugestões: 3 melhorias e 3 testes rápidos (A/B) para próxima variação.
4) Gere tags estruturadas (0-10) para formar um dataset e permitir previsões no futuro.

FORMATO DE SAÍDA (JSON):
{{
  "creative_key": "{context["creative_key"]}",
  "ad_id": "{context.get("ad_id","")}",
  "creative_id": "{context.get("creative_id","")}",
  "classification": {{
    "status_label": "{perf.get("label","")}",
    "why_top_or_bad": ["..."]
  }},
  "visual_diagnosis": {{
    "hook": {{"summary":"...", "strengths":["..."], "issues":["..."]}},
    "offer": {{"summary":"...", "strengths":["..."], "issues":["..."]}},
    "proof": {{"summary":"...", "strengths":["..."], "issues":["..."]}},
    "clarity_and_legibility": {{"summary":"...", "issues":["..."]}},
    "pacing": {{"summary":"...", "issues":["..."]}},
    "cta": {{"summary":"...", "issues":["..."]}},
    "overall_summary": "..."
  }},
  "action_plan": {{
    "improvements": ["...","...","..."],
    "ab_tests": ["...","...","..."]
  }},
  "structured_tags": {{
    "hook_strength_0_10": 0,
    "offer_clarity_0_10": 0,
    "proof_strength_0_10": 0,
    "text_legibility_0_10": 0,
    "pacing_0_10": 0,
    "brand_trust_0_10": 0,
    "creative_fatigue_risk_0_10": 0,
    "overall_quality_0_10": 0,
    "primary_hook_type": "texto|vsl|ugc|before_after|problem_solution|outro",
    "primary_message_type": "beneficio|dor|prova|oferta|outro"
  }},
  "predictions": {{
    "expected_score_0_10": 0,
    "expected_cpa_vs_avg_x": 1.0,
    "confidence_0_100": 50,
    "notes": "Explique em 1-2 frases o porquê da estimativa."
  }}
}}
""".strip()

def build_openai_payload(prompt: str, frame_paths: list) -> dict:
    """
    Payload para OpenAI Responses API.
    Envia texto + imagens (frames). (Vídeo inteiro é evitado; usamos frames).
    """
    content = [{"type": "input_text", "text": prompt}]
    # adiciona frames como input_image
    for p in frame_paths:
        # Usaremos data URL base64 para simplificar (sem upload). Atenção ao tamanho; por isso limitamos MAX_FRAMES.
        import base64
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})

    return {
        "model": AI_MODEL,
        "input": [
            {"role": "user", "content": content}
        ],
        # Ajuda a manter JSON “limpo”
        "text": {"format": {"type": "json_object"}},
    }

def extract_json_from_openai(resp_json: dict) -> dict:
    """Extrai o JSON do output_text. Com text.format=json_object, normalmente vem pronto."""
    # Tentativa 1: output_text direto
    out_text = resp_json.get("output_text")
    if out_text:
        try:
            return json.loads(out_text)
        except Exception:
            pass

    # Tentativa 2: varrer outputs
    for item in resp_json.get("output", []):
        for c in item.get("content", []):
            if c.get("type") in ("output_text", "text") and c.get("text"):
                try:
                    return json.loads(c["text"])
                except Exception:
                    continue
    raise RuntimeError("Não consegui extrair JSON da resposta da IA")

# ---------------------------
# Main
# ---------------------------
def main():
    run_at = now_utc_iso()

    # Catálogo pode ter nomes diferentes em versões anteriores. Tentamos alguns fallbacks.
    catalog_candidates = [
        Path(CATALOG_PATH),
        Path("creatives_output/catalog_ads.csv"),
        Path("creatives_output/catalogo.csv"),
    ]
    catalog_path = next((p for p in catalog_candidates if p.exists()), None)
    if not catalog_path:
        # Em vez de quebrar o pipeline inteiro, geramos um resumo vazio e saímos com sucesso.
        # Assim você ainda recebe o email/artefatos do score.
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write(
                "IA - Resumo da Execução\n"
                "======================\n"
                f"run_at (UTC): {run_at}\n"
                "\n"
                "⚠️ Catálogo de criativos não encontrado nesta execução.\n"
                "Verifique o step 'Download creatives' e o caminho 'creatives_output/catalog.csv'.\n"
            )
        print(f"⚠️ Catálogo não encontrado. Resumo vazio gerado em: {SUMMARY_PATH}")
        return
    if not Path(SCORE_PATH).exists():
        raise RuntimeError(f"Score latest não encontrado: {SCORE_PATH}")

    catalog = pd.read_csv(catalog_path)
    score = pd.read_csv(SCORE_PATH)

    # Normaliza colunas esperadas
    # Esperamos pelo menos: ad_id, ad_name_std, creative_id, media_type, local_path, frames_dir
    for col in ["ad_id", "creative_id", "media_type", "local_path"]:
        if col not in catalog.columns:
            raise RuntimeError(f"catalog.csv precisa ter a coluna: {col}")

    # score: colunas comuns do seu pipeline
    if "ad_id" not in score.columns:
        raise RuntimeError("creative_score_automation.csv precisa ter coluna ad_id")

    # Indexa score por ad_id
    score_idx = score.set_index("ad_id", drop=False)

    # Cache
    cache_rows = load_jsonl(CACHE_PATH)
    cache_idx = cache_index(cache_rows)

    analyzed = 0
    skipped = 0
    failures = 0

    # Para resumo
    paused_rows = []
    top_rows = []
    active_rows = []

    # métricas globais p/ CPA vs média
    avg_cpa = None
    if "cac" in score.columns:
        cpa_valid = score[(score["cac"].notna()) & (score["cac"] > 0)]
        if not cpa_valid.empty:
            avg_cpa = float(cpa_valid["cac"].mean())

    total_spend = float(score["spend_std"].sum()) if "spend_std" in score.columns else 0.0

    for _, row in catalog.iterrows():
        ad_id = str(row.get("ad_id", ""))
        creative_id = str(row.get("creative_id", ""))
        media_type = str(row.get("media_type", "")).lower()
        local_path = Path(str(row.get("local_path", "")))

        # Pega nome do anúncio (preferência: do score, senão do catalog)
        ad_name = None
        if ad_id in score_idx.index:
            ad_name = str(score_idx.loc[ad_id].get("ad_name_std", ""))
        if not ad_name:
            ad_name = str(row.get("ad_name", row.get("ad_name_std", "")))

        ad_name = normalize_name(ad_name)
        creative_key = ad_name  # dedupe “pela nomenclatura” (como você pediu)

        if not local_path.exists():
            # sem mídia => não analisa
            continue

        media_sha = sha256_file(local_path)
        cache_key = (creative_key, media_sha)

        if cache_key in cache_idx:
            # já analisado exatamente esse criativo (mesmo nome + mesmo arquivo)
            skipped += 1
            # opcional: registrar no results “reuso”
            reuse = dict(cache_idx[cache_key])
            reuse["run_at"] = run_at
            reuse["reused_from_cache"] = True
            append_jsonl(RESULTS_PATH, reuse)
            continue

        # contexto de performance
        perf_row = score_idx.loc[ad_id] if ad_id in score_idx.index else None
        spend = float(perf_row.get("spend_std", 0.0)) if perf_row is not None and "spend_std" in perf_row else 0.0
        share_spend = (spend / total_spend) if total_spend > 0 else 0.0
        cpa = float(perf_row.get("cac", 0.0)) if perf_row is not None and "cac" in perf_row else 0.0
        score_val = float(perf_row.get("performance_score", float("nan"))) if perf_row is not None and "performance_score" in perf_row else float("nan")

        # label do status (para IA entender o papel do criativo)
        label = "ACTIVE"
        if perf_row is not None:
            reason = str(perf_row.get("pause_reason", "")) if "pause_reason" in perf_row else ""
            # Você pode adaptar conforme como seu script salva reasons
            if "HARD" in reason.upper():
                label = "PAUSED_HARD_STOP"
            elif "SCORE" in reason.upper():
                label = "PAUSED_LOW_SCORE"
            # top marker (se você salvar no score)
            if str(perf_row.get("is_top5", "")).lower() in ("1", "true", "yes"):
                label = "TOP_PERFORMER"

        perf = {
            "label": label,
            "score": None if math.isnan(score_val) else round(score_val, 2),
            "spend_br": money_br(spend),
            "share_spend_pct": pct(share_spend),
            "cpa_br": money_br(cpa) if cpa > 0 else "sem purchase (1d)",
            "cpa_vs_avg": ("—" if not avg_cpa or avg_cpa == 0 or cpa <= 0 else f"{(cpa/avg_cpa):.2f}x"),
            "ctr": None if perf_row is None else perf_row.get("ctr", None),
            "connect_rate": None if perf_row is None else perf_row.get("connect_rate", None),
            "bounce_rate": None if perf_row is None else perf_row.get("bounce_rate", None),
            "cost_per_checkout_br": money_br(perf_row.get("cost_per_checkout", 0.0)) if perf_row is not None and "cost_per_checkout" in perf_row else money_br(0.0),
        }

        # frames
        frames_dir = None
        if "frames_dir" in row and isinstance(row.get("frames_dir"), str) and row.get("frames_dir"):
            frames_dir = Path(row.get("frames_dir"))
        else:
            # fallback: padrão do downloader
            frames_dir = Path("creatives_output/frames") / ad_id

        frame_paths = pick_frames(frames_dir, MAX_FRAMES)

        context = {
            "run_at": run_at,
            "creative_key": creative_key,
            "ad_id": ad_id,
            "creative_id": creative_id,
            "media_type": media_type,
            "media_path": str(local_path),
            "media_sha256": media_sha,
            "name_fields": parse_ad_name(ad_name),
            "performance": perf,
        }

        prompt = build_prompt_structured(context)

        try:
            if AI_PROVIDER == "none":
                # modo “sem IA” (debug)
                result = {
                    **context,
                    "analysis": {"status": "SKIPPED_NO_PROVIDER"}
                }
            elif AI_PROVIDER == "openai":
                if not AI_API_KEY:
                    raise RuntimeError("AI_API_KEY não definido (provider=openai)")
                payload = build_openai_payload(prompt, frame_paths)
                resp = openai_request(payload)
                parsed_json = extract_json_from_openai(resp)
                result = {
                    **context,
                    "ai_provider": "openai",
                    "ai_model": AI_MODEL,
                    "analysis": parsed_json,
                }
            elif AI_PROVIDER == "http":
                payload = {"prompt": prompt, "frames": frame_paths, "context": context, "model": AI_MODEL}
                resp = http_request(payload)
                result = {
                    **context,
                    "ai_provider": "http",
                    "analysis": resp,
                }
            else:
                raise RuntimeError(f"AI_PROVIDER inválido: {AI_PROVIDER}")

            # salva results + cache
            append_jsonl(RESULTS_PATH, result)

            cache_obj = dict(result)
            cache_obj["cached_at"] = run_at
            cache_obj["reused_from_cache"] = False
            append_jsonl(CACHE_PATH, cache_obj)

            analyzed += 1

            if label in ("PAUSED_LOW_SCORE", "PAUSED_HARD_STOP"):
                paused_rows.append(result)
            elif label == "TOP_PERFORMER":
                top_rows.append(result)
            else:
                active_rows.append(result)

        except Exception as e:
            failures += 1
            err = {
                **context,
                "error": str(e),
            }
            append_jsonl(RESULTS_PATH, err)

    # ---------------------------
    # Resumo TXT (para Slack/email)
    # ---------------------------
    def line_for(r):
        perf = r.get("performance", {})
        analysis = r.get("analysis", {})
        tags = None
        # tenta localizar structured_tags no JSON
        if isinstance(analysis, dict):
            tags = analysis.get("structured_tags") or analysis.get("analysis", {}).get("structured_tags")
        tag_str = ""
        if isinstance(tags, dict):
            tag_str = f" | quality={tags.get('overall_quality_0_10','—')} hook={tags.get('hook_strength_0_10','—')} offer={tags.get('offer_clarity_0_10','—')}"

        return (
            f"- {r.get('creative_key','')} | spend={perf.get('spend_br')} share={perf.get('share_spend_pct')} "
            f"| cpa={perf.get('cpa_br')} vs_avg={perf.get('cpa_vs_avg')} | score={perf.get('score')}{tag_str}"
        )

    lines = []
    lines.append("IA - Resumo da Execução")
    lines.append("======================")
    lines.append(f"run_at (UTC): {run_at}")
    lines.append(f"analisados_novos: {analyzed}")
    lines.append(f"reutilizados_cache: {skipped}")
    lines.append(f"falhas: {failures}")
    lines.append("")

    lines.append("Pausados (LOW_SCORE ou HARD_STOP) - com análise:")
    lines.append("-----------------------------------------------")
    if paused_rows:
        for r in paused_rows[:50]:
            lines.append(line_for(r))
    else:
        lines.append("Nenhum.")
    lines.append("")

    lines.append("Top performers (marcados como TOP_PERFORMER) - com análise:")
    lines.append("-----------------------------------------------------------")
    if top_rows:
        for r in top_rows[:50]:
            lines.append(line_for(r))
    else:
        lines.append("Nenhum (você pode marcar is_top5 no score para habilitar).")
    lines.append("")

    lines.append("Ativos (ACTIVE) - com análise:")
    lines.append("-----------------------------------------------------------")
    if active_rows:
        for r in active_rows[:50]:
            lines.append(line_for(r))
    else:
        lines.append("Nenhum.")
    lines.append("")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ V2 finalizada | novos={analyzed} | cache={skipped} | falhas={failures}")
    print(f"✅ Results: {RESULTS_PATH}")
    print(f"✅ Cache: {CACHE_PATH}")
    print(f"✅ Summary: {SUMMARY_PATH}")

if __name__ == "__main__":
    main()
