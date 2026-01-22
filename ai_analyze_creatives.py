#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_analyze_creatives.py - V3 (Reestruturado)

Analisa TODOS os criativos usando IA (OpenAI GPT-4o), não apenas os TOP5.

Novidades:
- Analisa TODOS os criativos (ativos, pausados, top performers)
- Envia métricas completas: CPM, CPC, Hook Rate, Placement, etc.
- Gera descrição detalhada de vídeo/áudio
- Backlog inteligente para não reprocessar criativos já analisados
- Suporta tanto vídeos quanto imagens estáticas
"""

import os
import json
import base64
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

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

# Configurações
CONSOLIDATED_PATH = Path("creatives_output/consolidated_data.json")
BACKLOG_PATH = Path("creatives_output/backlog.json")
OUTPUT_DIR = Path("creatives_output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUTPUT_DIR / "analysis_results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary_ai.txt"

# Configurações de IA
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_ENDPOINT_URL = os.getenv("AI_ENDPOINT_URL", "").strip()
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o")
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT_SECONDS", "180"))
MAX_FRAMES = int(os.getenv("AI_MAX_FRAMES", "8"))

# URL padrão da OpenAI
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"


def now_utc_iso() -> str:
    """Retorna timestamp UTC no formato ISO."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_number(value) -> str:
    """Formata número com separador de milhares, tratando valores inválidos."""
    if value is None or value == 'N/A' or value == '':
        return 'N/A'
    try:
        num = float(value)
        if num == int(num):
            return f"{int(num):,}".replace(',', '.')
        return f"{num:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except (ValueError, TypeError):
        return str(value)


def load_json(path: Path) -> Optional[Dict]:
    """Carrega arquivo JSON."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar {path}: {e}")
        return None


def save_json(path: Path, data: Dict) -> None:
    """Salva arquivo JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj: Dict) -> None:
    """Adiciona objeto ao arquivo JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def pick_frames(frames_dir: str, max_frames: int) -> List[str]:
    """Seleciona frames para enviar à IA."""
    frames_path = Path(frames_dir)
    if not frames_path.exists():
        return []
    
    frames = sorted([p for p in frames_path.glob("*.jpg") if p.is_file()])
    if not frames:
        # Tentar PNG também
        frames = sorted([p for p in frames_path.glob("*.png") if p.is_file()])
    
    if not frames:
        return []
    
    if len(frames) <= max_frames:
        return [str(p) for p in frames]
    
    # Amostra distribuída
    step = max(1, len(frames) // max_frames)
    idxs = list(range(0, len(frames), step))[:max_frames]
    return [str(frames[i]) for i in idxs]


def encode_image_base64(path: str) -> Optional[str]:
    """Codifica imagem em base64."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Erro ao codificar imagem {path}: {e}")
        return None


def build_analysis_prompt(creative: Dict) -> str:
    """Constrói o prompt para análise do criativo com TODAS as métricas."""
    kpis = creative.get("kpis", {})
    name_fields = creative.get("name_fields", {})
    status = creative.get("status", "UNKNOWN")
    media_type = creative.get("media_type", "unknown")
    
    # Formatar métricas de vídeo se aplicável
    video_metrics_section = ""
    if media_type == "video":
        video_metrics_section = f"""
**Métricas de Vídeo:**
- Hook Rate: {kpis.get('hook_rate_pct', 'N/A')} (vs média: {kpis.get('hook_rate_vs_avg', 'N/A')})
- Retenção 25%: {kpis.get('video_p25', 'N/A')} views
- Retenção 50%: {kpis.get('video_p50', 'N/A')} views
- Retenção 75%: {kpis.get('video_p75', 'N/A')} views
- Retenção 100%: {kpis.get('video_p100', 'N/A')} views
- Taxa 25%→50%: {kpis.get('retention_25_to_50', 'N/A')}
- Taxa 50%→75%: {kpis.get('retention_50_to_75', 'N/A')}
- Taxa 75%→100%: {kpis.get('retention_75_to_100', 'N/A')}
"""
    
    return f"""
Você é um especialista em criativos de performance para Meta Ads (Facebook/Instagram).
Analise este criativo de forma detalhada e estruturada.

## CONTEXTO DO ANÚNCIO

**Identificação:**
- Ad ID: {creative.get('ad_id', 'N/A')}
- Nome: {creative.get('ad_name', 'N/A')}
- Tipo de Mídia: {media_type.upper()} ({'Vídeo' if media_type == 'video' else 'Imagem Estática'})
- Status: {status}
- Campanha: {kpis.get('campaign_name', 'N/A')}
- Conjunto: {kpis.get('adset_name', 'N/A')}

**Campos da Nomenclatura:**
{json.dumps(name_fields, ensure_ascii=False, indent=2)}

**KPIs de Performance (últimos 21 dias):**
- Score de Performance: {kpis.get('performance_score', 'N/A')}/10
- Gasto: {kpis.get('spend_br', 'N/A')} ({kpis.get('share_spend', 'N/A')} do total)
- CAC: {kpis.get('cac_br', 'N/A')} (vs média: {kpis.get('cac_vs_avg', 'N/A')})
- CPM: {kpis.get('cpm_br', 'N/A')} (vs média: {kpis.get('cpm_vs_avg', 'N/A')})
- CPC: {kpis.get('cpc_br', 'N/A')} (vs média: {kpis.get('cpc_vs_avg', 'N/A')})
- CTR: {kpis.get('ctr_pct', 'N/A')}
- Connect Rate: {kpis.get('connect_rate_pct', 'N/A')}
- Bounce Rate: {kpis.get('bounce_rate_pct', 'N/A')}
- Impressões: {format_number(kpis.get('impressions', 0))}
- Cliques: {format_number(kpis.get('clicks', 0))}
- Compras: {kpis.get('purchases', 'N/A')}
- Checkouts: {kpis.get('checkouts', 'N/A')}
{video_metrics_section}
**Distribuição de Placement:**
- Principal: {kpis.get('top_placement', 'N/A')} / {kpis.get('top_position', 'N/A')}
- Quantidade de placements: {kpis.get('placement_count', 'N/A')}

## TAREFA

Analise {'os frames do vídeo' if media_type == 'video' else 'a imagem'} e forneça:

1. **Descrição Detalhada {'do Vídeo' if media_type == 'video' else 'da Imagem'}:**
   - O que é mostrado {'em cada parte do vídeo' if media_type == 'video' else 'na imagem'}
   - Elementos visuais principais (pessoas, produtos, texto, cores)
   {'- Transições e ritmo' if media_type == 'video' else ''}
   - Qualidade de produção

{'2. **Análise de Áudio (estimativa baseada no visual):**' if media_type == 'video' else ''}
{'   - Tipo provável de narração' if media_type == 'video' else ''}
{'   - Música de fundo provável' if media_type == 'video' else ''}
{'   - Tom geral do áudio' if media_type == 'video' else ''}

{'3' if media_type == 'video' else '2'}. **Diagnóstico Visual:**
   - Hook (gancho inicial): força de 0-10 e justificativa
   - Clareza da oferta: força de 0-10 e justificativa
   - Prova social: força de 0-10 e justificativa
   - Legibilidade do texto: força de 0-10 e justificativa
   - Call to Action: força de 0-10 e justificativa

{'4' if media_type == 'video' else '3'}. **Diagnóstico de Performance:**
   - Por que este criativo está {status}?
   - Correlação entre elementos visuais e métricas
   - O que explica o {'Hook Rate' if media_type == 'video' else 'CTR'}?
   - O que explica o CAC?

{'5' if media_type == 'video' else '4'}. **Classificação:**
   - Tema principal
   - Tipo de narrativa (UGC, Profissional, Animação, etc.)
   - Tom de voz (Urgente, Informativo, Emocional, etc.)
   - Mensagem principal
   - Tipo de produção

{'6' if media_type == 'video' else '5'}. **Sugestões:**
   - 3 melhorias específicas
   - 3 testes A/B recomendados

## FORMATO DE SAÍDA (JSON)

Responda SOMENTE com um JSON válido no seguinte formato:

{{
  "creative_description": {{
    "summary": "Descrição geral do criativo em 2-3 frases",
    "scenes": [
      {{"time": "0-3s", "description": "..."}},
      {{"time": "3-6s", "description": "..."}}
    ],
    "visual_elements": ["elemento1", "elemento2"],
    "text_on_screen": ["texto1", "texto2"],
    "colors_dominant": ["cor1", "cor2"],
    "people_shown": "descrição das pessoas mostradas",
    "product_shown": "descrição do produto mostrado",
    "production_quality": "alta|média|baixa",
    "estimated_duration": "Xs"
  }},
  "audio_description": {{
    "narration_type": "feminina|masculina|sem narração|não aplicável",
    "narration_tone": "urgente|calmo|informativo|emocional",
    "background_music": "sim|não|provável",
    "music_style": "descrição do estilo",
    "spoken_message_summary": "Resumo do que provavelmente é dito"
  }},
  "visual_diagnosis": {{
    "hook": {{"score": 0, "justification": "..."}},
    "offer_clarity": {{"score": 0, "justification": "..."}},
    "social_proof": {{"score": 0, "justification": "..."}},
    "text_legibility": {{"score": 0, "justification": "..."}},
    "cta": {{"score": 0, "justification": "..."}}
  }},
  "performance_diagnosis": {{
    "status_explanation": "Por que está {status}",
    "visual_metric_correlation": "Correlação entre visual e métricas",
    "hook_rate_explanation": "O que explica o hook rate/CTR",
    "cac_explanation": "O que explica o CAC"
  }},
  "classification": {{
    "main_theme": "...",
    "narrative_type": "UGC|Profissional|Animação|Estático|Outro",
    "tone_of_voice": "Urgente|Informativo|Emocional|Aspiracional|Outro",
    "main_message": "...",
    "production_type": "In-house|Creator|Agência|Outro",
    "primary_hook_type": "texto|vsl|ugc|before_after|problem_solution|outro",
    "format_type": "{'video' if media_type == 'video' else 'static'}"
  }},
  "suggestions": {{
    "improvements": ["melhoria1", "melhoria2", "melhoria3"],
    "ab_tests": ["teste1", "teste2", "teste3"]
  }},
  "structured_tags": {{
    "hook_strength": 0,
    "offer_clarity": 0,
    "social_proof_strength": 0,
    "text_legibility": 0,
    "cta_strength": 0,
    "overall_quality": 0,
    "creative_fatigue_risk": 0
  }}
}}
""".strip()


def call_openai_api(prompt: str, frame_paths: List[str]) -> Dict:
    """Chama a API da OpenAI com o prompt e imagens."""
    # Determinar URL
    if AI_ENDPOINT_URL and AI_ENDPOINT_URL.startswith("https://") and "/chat/completions" in AI_ENDPOINT_URL:
        url = AI_ENDPOINT_URL
    else:
        url = DEFAULT_API_URL
    
    logger.info(f"Usando URL da API: {url}")
    logger.info(f"Modelo: {AI_MODEL}")
    logger.info(f"Frames a enviar: {len(frame_paths)}")
    
    # Monta o conteúdo com texto e imagens
    content = [{"type": "text", "text": prompt}]
    
    for frame_path in frame_paths:
        b64 = encode_image_base64(frame_path)
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
            })
    
    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
        "max_tokens": 4096,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI_API_KEY}",
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=AI_TIMEOUT)
    response.raise_for_status()
    
    result = response.json()
    content_str = result["choices"][0]["message"]["content"]
    return json.loads(content_str)


def analyze_creative(creative: Dict) -> Optional[Dict]:
    """Analisa um criativo individual."""
    ad_id = creative.get("ad_id", "")
    frames_dir = creative.get("frames_dir", "")
    media_type = creative.get("media_type", "unknown")
    
    # Seleciona frames
    frame_paths = pick_frames(frames_dir, MAX_FRAMES)
    if not frame_paths:
        logger.warning(f"Nenhum frame encontrado para ad_id {ad_id}")
        return None
    
    # Constrói prompt
    prompt = build_analysis_prompt(creative)
    
    # Chama a IA
    try:
        analysis = call_openai_api(prompt, frame_paths)
        return {
            "ad_id": ad_id,
            "creative_id": creative.get("creative_id", ""),
            "ad_name": creative.get("ad_name", ""),
            "status": creative.get("status", ""),
            "media_type": media_type,
            "kpis": creative.get("kpis", {}),
            "analysis": analysis,
            "analyzed_at": now_utc_iso(),
            "frames_analyzed": len(frame_paths),
        }
    except Exception as e:
        logger.error(f"Erro ao analisar ad_id {ad_id}: {e}")
        return None


def main():
    """Função principal."""
    run_at = now_utc_iso()
    logger.info("=" * 60)
    logger.info(f"Iniciando análise de IA em {run_at}")
    logger.info("=" * 60)
    
    # Verifica API Key
    if not AI_API_KEY:
        logger.error("AI_API_KEY não configurada. Abortando.")
        return
    
    # Carrega dados consolidados
    consolidated = load_json(CONSOLIDATED_PATH)
    if not consolidated:
        logger.error(f"Arquivo consolidado não encontrado: {CONSOLIDATED_PATH}")
        return
    
    # Carrega backlog
    backlog = load_json(BACKLOG_PATH) or {"processed_creatives": {}, "last_updated": None}
    
    creatives = consolidated.get("creatives", [])
    logger.info(f"Total de criativos: {len(creatives)}")
    
    # Contadores
    analyzed = 0
    failed = 0
    skipped = 0
    
    # Listas para o resumo
    results_by_status = {
        "PAUSED": [],
        "PAUSED_HARD_STOP": [],
        "PAUSED_LOW_SCORE": [],
        "TOP_PERFORMER": [],
        "ACTIVE": [],
    }
    
    for creative in creatives:
        ad_id = creative.get("ad_id", "")
        sha256 = creative.get("sha256", "")
        creative_key = f"{ad_id}_{sha256}"
        status = creative.get("status", "UNKNOWN")
        
        # Verifica se já foi processado (mas ainda assim analisa se quiser forçar)
        if creative_key in backlog["processed_creatives"]:
            skipped += 1
            # Recupera resultado anterior se existir
            continue
        
        # Analisa o criativo
        logger.info(f"Analisando: {ad_id} ({status})")
        result = analyze_creative(creative)
        
        if result:
            # Salva resultado
            append_jsonl(RESULTS_PATH, result)
            
            # Atualiza backlog
            backlog["processed_creatives"][creative_key] = {
                "ad_id": ad_id,
                "analyzed_at": result["analyzed_at"],
                "status": status,
            }
            
            # Categoriza para o resumo
            if status in results_by_status:
                results_by_status[status].append(result)
            else:
                results_by_status["ACTIVE"].append(result)
            
            analyzed += 1
            logger.info(f"✅ Analisado: {ad_id} ({analyzed}/{len(creatives)})")
        else:
            failed += 1
            logger.warning(f"❌ Falha: {ad_id}")
    
    # Salva backlog atualizado
    backlog["last_updated"] = run_at
    save_json(BACKLOG_PATH, backlog)
    
    # Gera resumo
    total_paused = len(results_by_status["PAUSED"]) + len(results_by_status["PAUSED_HARD_STOP"]) + len(results_by_status["PAUSED_LOW_SCORE"])
    
    summary_lines = [
        "=" * 60,
        "IA - Resumo da Execução",
        "=" * 60,
        f"Data/Hora (UTC): {run_at}",
        f"Total de criativos: {len(creatives)}",
        f"Analisados nesta execução: {analyzed}",
        f"Reutilizados do cache: {skipped}",
        f"Falhas: {failed}",
        "",
        "-" * 60,
        f"PAUSADOS ({total_paused} criativos):",
        "-" * 60,
    ]
    
    for status_key in ["PAUSED", "PAUSED_HARD_STOP", "PAUSED_LOW_SCORE"]:
        for r in results_by_status[status_key][:5]:
            summary_lines.append(f"  [{status_key}] {r.get('ad_name', '')[:50]}...")
    
    summary_lines.extend([
        "",
        "-" * 60,
        f"TOP PERFORMERS ({len(results_by_status['TOP_PERFORMER'])} criativos):",
        "-" * 60,
    ])
    
    for r in results_by_status["TOP_PERFORMER"][:5]:
        summary_lines.append(f"  - {r.get('ad_name', '')[:50]}...")
    
    summary_lines.extend([
        "",
        "-" * 60,
        f"ATIVOS ({len(results_by_status['ACTIVE'])} criativos):",
        "-" * 60,
    ])
    
    for r in results_by_status["ACTIVE"][:10]:
        summary_lines.append(f"  - {r.get('ad_name', '')[:50]}...")
    
    summary_lines.append("")
    summary_lines.append("=" * 60)
    
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    
    logger.info("=" * 60)
    logger.info(f"Análise concluída: {analyzed} novos, {skipped} cache, {failed} falhas")
    logger.info(f"Resumo salvo em: {SUMMARY_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
