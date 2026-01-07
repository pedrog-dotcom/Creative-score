#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_analyze_creatives.py - V4 (Reestruturado)

Este script realiza a análise de criativos usando IA (OpenAI GPT-4o).

Fluxo:
1. Lê o arquivo consolidado (consolidated_data.json)
2. Verifica o backlog para identificar criativos já processados
3. Envia apenas criativos novos para a IA
4. Gera descrição detalhada de vídeo/áudio
5. Salva resultados e atualiza o backlog

Saídas:
- creatives_output/analysis/analysis_results.jsonl
- creatives_output/backlog.json (atualizado)
- creatives_output/analysis/summary_ai.txt
"""

import os
import json
import base64
import logging
import hashlib
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
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower()
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_ENDPOINT_URL = os.getenv("AI_ENDPOINT_URL", "").strip()
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o")
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT_SECONDS", "180"))
MAX_FRAMES = int(os.getenv("AI_MAX_FRAMES", "8"))


def now_utc_iso() -> str:
    """Retorna timestamp UTC no formato ISO."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        return []
    
    if len(frames) <= max_frames:
        return [str(p) for p in frames]
    
    # Amostra distribuída (início, meio, fim)
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
    """Constrói o prompt para análise do criativo."""
    kpis = creative.get("kpis", {})
    name_fields = creative.get("name_fields", {})
    status = creative.get("status", "UNKNOWN")
    
    return f"""
Você é um especialista em criativos de performance para Meta Ads (Facebook/Instagram).
Analise este criativo de forma detalhada e estruturada.

## CONTEXTO DO ANÚNCIO

**Identificação:**
- Ad ID: {creative.get('ad_id', 'N/A')}
- Nome: {creative.get('ad_name', 'N/A')}
- Tipo de Mídia: {creative.get('media_type', 'N/A')}
- Status: {status}

**Campos da Nomenclatura:**
{json.dumps(name_fields, ensure_ascii=False, indent=2)}

**KPIs de Performance (últimos 21 dias):**
- Score de Performance: {kpis.get('performance_score', 'N/A')}
- Gasto: {kpis.get('spend_br', 'N/A')}
- Share do Gasto: {kpis.get('share_spend', 'N/A')}
- CAC: {kpis.get('cac_br', 'N/A')}
- CAC vs Média: {kpis.get('cac_vs_avg', 'N/A')}
- CTR: {kpis.get('ctr', 'N/A')}
- Connect Rate: {kpis.get('connect_rate', 'N/A')}
- Bounce Rate: {kpis.get('bounce_rate', 'N/A')}
- Impressões: {kpis.get('impressions', 'N/A')}
- Cliques: {kpis.get('clicks', 'N/A')}
- Compras: {kpis.get('purchases', 'N/A')}

## TAREFA

Analise os frames do criativo e forneça:

1. **Descrição Detalhada do Vídeo/Imagem:**
   - O que é mostrado em cada parte do vídeo
   - Elementos visuais principais (pessoas, produtos, texto, cores)
   - Transições e ritmo (se for vídeo)
   - Qualidade de produção

2. **Análise de Áudio (se aplicável):**
   - Tipo de narração (voz feminina/masculina, tom)
   - Música de fundo
   - Efeitos sonoros
   - Clareza da mensagem falada

3. **Diagnóstico Visual:**
   - Hook (gancho inicial): força de 0-10 e justificativa
   - Clareza da oferta: força de 0-10 e justificativa
   - Prova social: força de 0-10 e justificativa
   - Legibilidade do texto: força de 0-10 e justificativa
   - Call to Action: força de 0-10 e justificativa

4. **Diagnóstico de Performance:**
   - Por que este criativo está {status}?
   - Correlação entre elementos visuais e métricas

5. **Classificação:**
   - Tema principal
   - Tipo de narrativa (UGC, Profissional, Animação, etc.)
   - Tom de voz (Urgente, Informativo, Emocional, etc.)
   - Mensagem principal
   - Tipo de produção

6. **Sugestões:**
   - 3 melhorias específicas
   - 3 testes A/B recomendados

## FORMATO DE SAÍDA (JSON)

Responda SOMENTE com um JSON válido no seguinte formato:

{{
  "video_description": {{
    "summary": "Descrição geral do criativo em 2-3 frases",
    "scenes": [
      {{"time": "0-3s", "description": "..."}},
      {{"time": "3-6s", "description": "..."}}
    ],
    "visual_elements": ["elemento1", "elemento2"],
    "production_quality": "alta|média|baixa",
    "estimated_duration": "Xs"
  }},
  "audio_description": {{
    "narration_type": "feminina|masculina|sem narração",
    "narration_tone": "urgente|calmo|informativo|emocional",
    "background_music": "sim|não",
    "music_style": "descrição do estilo",
    "sound_effects": ["efeito1", "efeito2"],
    "spoken_message_summary": "Resumo do que é dito"
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
    "visual_metric_correlation": "Correlação entre visual e métricas"
  }},
  "classification": {{
    "main_theme": "...",
    "narrative_type": "UGC|Profissional|Animação|Estático|Outro",
    "tone_of_voice": "Urgente|Informativo|Emocional|Aspiracional|Outro",
    "main_message": "...",
    "production_type": "In-house|Creator|Agência|Outro",
    "primary_hook_type": "texto|vsl|ugc|before_after|problem_solution|outro"
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
    # URL padrão da OpenAI - SEMPRE usar esta a menos que AI_ENDPOINT_URL seja uma URL completa válida
    DEFAULT_URL = "https://api.openai.com/v1/chat/completions"
    
    # Só usa AI_ENDPOINT_URL se for uma URL completa e válida (começa com https://)
    if AI_ENDPOINT_URL and AI_ENDPOINT_URL.startswith("https://") and "/chat/completions" in AI_ENDPOINT_URL:
        url = AI_ENDPOINT_URL
    else:
        url = DEFAULT_URL
    
    logger.info(f"Usando URL da API: {url}")
    logger.info(f"Modelo: {AI_MODEL}")
    
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
    logger.info(f"Iniciando análise de IA em {run_at}")
    
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
    
    # Filtra criativos que precisam de análise
    pending = [c for c in creatives if c.get("needs_ai_analysis", True)]
    logger.info(f"Criativos pendentes de análise: {len(pending)}")
    
    # Contadores
    analyzed = 0
    failed = 0
    skipped = 0
    
    # Listas para o resumo
    paused_results = []
    top_results = []
    active_results = []
    
    for creative in pending:
        ad_id = creative.get("ad_id", "")
        sha256 = creative.get("sha256", "")
        creative_key = f"{ad_id}_{sha256}"
        
        # Verifica se já foi processado
        if creative_key in backlog["processed_creatives"]:
            skipped += 1
            continue
        
        # Analisa o criativo
        result = analyze_creative(creative)
        
        if result:
            # Salva resultado
            append_jsonl(RESULTS_PATH, result)
            
            # Atualiza backlog
            backlog["processed_creatives"][creative_key] = {
                "ad_id": ad_id,
                "analyzed_at": result["analyzed_at"],
                "status": result["status"],
            }
            
            # Categoriza para o resumo
            status = result.get("status", "")
            if "PAUSED" in status:
                paused_results.append(result)
            elif status == "TOP_PERFORMER":
                top_results.append(result)
            else:
                active_results.append(result)
            
            analyzed += 1
            logger.info(f"Analisado: {ad_id} ({analyzed}/{len(pending)})")
        else:
            failed += 1
    
    # Salva backlog atualizado
    backlog["last_updated"] = run_at
    save_json(BACKLOG_PATH, backlog)
    
    # Gera resumo
    summary_lines = [
        "=" * 60,
        "IA - Resumo da Execução",
        "=" * 60,
        f"Data/Hora (UTC): {run_at}",
        f"Total de criativos: {len(creatives)}",
        f"Pendentes de análise: {len(pending)}",
        f"Analisados nesta execução: {analyzed}",
        f"Reutilizados do cache: {skipped}",
        f"Falhas: {failed}",
        "",
        "-" * 60,
        f"PAUSADOS ({len(paused_results)} criativos):",
        "-" * 60,
    ]
    
    for r in paused_results[:10]:
        summary_lines.append(f"  - {r.get('ad_name', '')[:50]}...")
    
    summary_lines.extend([
        "",
        "-" * 60,
        f"TOP PERFORMERS ({len(top_results)} criativos):",
        "-" * 60,
    ])
    
    for r in top_results[:10]:
        summary_lines.append(f"  - {r.get('ad_name', '')[:50]}...")
    
    summary_lines.extend([
        "",
        "-" * 60,
        f"ATIVOS ({len(active_results)} criativos):",
        "-" * 60,
    ])
    
    for r in active_results[:10]:
        summary_lines.append(f"  - {r.get('ad_name', '')[:50]}...")
    
    summary_lines.append("")
    summary_lines.append("=" * 60)
    
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    
    logger.info(f"Análise concluída: {analyzed} novos, {skipped} cache, {failed} falhas")
    logger.info(f"Resumo salvo em: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
