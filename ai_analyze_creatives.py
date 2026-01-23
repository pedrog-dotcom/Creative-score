#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_analyze_creatives.py - V4 (Reestruturado Completo)

Analisa TODOS os criativos usando IA (OpenAI GPT-4o).

Funcionalidades:
- Analisa TODOS os criativos (ativos, pausados, top performers)
- Busca mídia de múltiplas fontes (frames_dir, local_path, images/)
- Envia imagens/frames reais para a IA
- Correlaciona visual com score de performance
- Gera insights sobre o que funciona e o que evitar
- Backlog inteligente para não reprocessar criativos já analisados
"""

import os
import json
import base64
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from glob import glob

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

# Configurações de Caminhos
CONSOLIDATED_PATH = Path("creatives_output/consolidated_data.json")
BACKLOG_PATH = Path("creatives_output/backlog.json")
OUTPUT_DIR = Path("creatives_output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUTPUT_DIR / "analysis_results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary_ai.txt"

# Diretórios de mídia
FRAMES_BASE_DIR = Path("creatives_output/frames")
IMAGES_BASE_DIR = Path("creatives_output/images")

# Configurações de IA
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_ENDPOINT_URL = os.getenv("AI_ENDPOINT_URL", "").strip()
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o")
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT_SECONDS", "180"))
MAX_FRAMES = int(os.getenv("AI_MAX_FRAMES", "6"))

# URL padrão da OpenAI
DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"


def now_utc_iso() -> str:
    """Retorna timestamp UTC no formato ISO."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_number(value) -> str:
    """Formata número com separador de milhares."""
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


def find_media_files(creative: Dict) -> List[str]:
    """
    Busca arquivos de mídia (imagens/frames) para um criativo.
    Tenta múltiplas fontes em ordem de prioridade.
    """
    ad_id = str(creative.get("ad_id", ""))
    creative_id = str(creative.get("creative_id", ""))
    frames_dir = creative.get("frames_dir", "")
    local_path = creative.get("local_path", "")
    media_type = creative.get("media_type", "").lower()
    
    found_files = []
    
    # 1. Tentar frames_dir se especificado
    if frames_dir:
        frames_path = Path(frames_dir)
        if frames_path.exists():
            files = list(frames_path.glob("*.jpg")) + list(frames_path.glob("*.png"))
            if files:
                logger.info(f"  Encontrados {len(files)} frames em {frames_dir}")
                found_files.extend([str(f) for f in sorted(files)])
    
    # 2. Tentar pasta de frames por ad_id
    if not found_files:
        ad_frames_dir = FRAMES_BASE_DIR / ad_id
        if ad_frames_dir.exists():
            files = list(ad_frames_dir.glob("*.jpg")) + list(ad_frames_dir.glob("*.png"))
            if files:
                logger.info(f"  Encontrados {len(files)} frames em {ad_frames_dir}")
                found_files.extend([str(f) for f in sorted(files)])
    
    # 3. Tentar pasta de frames por creative_id
    if not found_files and creative_id:
        creative_frames_dir = FRAMES_BASE_DIR / creative_id
        if creative_frames_dir.exists():
            files = list(creative_frames_dir.glob("*.jpg")) + list(creative_frames_dir.glob("*.png"))
            if files:
                logger.info(f"  Encontrados {len(files)} frames em {creative_frames_dir}")
                found_files.extend([str(f) for f in sorted(files)])
    
    # 4. Tentar local_path diretamente (para imagens estáticas)
    if not found_files and local_path:
        local_file = Path(local_path)
        if local_file.exists() and local_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            logger.info(f"  Usando imagem local: {local_path}")
            found_files.append(str(local_file))
    
    # 5. Tentar pasta de imagens por ad_id
    if not found_files:
        # Busca por padrão: images/*ad_id*
        pattern = str(IMAGES_BASE_DIR / f"*{ad_id}*")
        matches = glob(pattern)
        for match in matches:
            if Path(match).suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                logger.info(f"  Encontrada imagem: {match}")
                found_files.append(match)
    
    # 6. Busca genérica por ad_id em todo creatives_output
    if not found_files:
        for ext in ['jpg', 'jpeg', 'png', 'webp']:
            pattern = f"creatives_output/**/*{ad_id}*.{ext}"
            matches = glob(pattern, recursive=True)
            if matches:
                logger.info(f"  Encontradas {len(matches)} imagens via busca genérica")
                found_files.extend(matches)
                break
    
    return found_files


def select_frames(files: List[str], max_frames: int) -> List[str]:
    """Seleciona frames para enviar à IA (amostra distribuída)."""
    if not files:
        return []
    
    if len(files) <= max_frames:
        return files
    
    # Amostra distribuída
    step = max(1, len(files) // max_frames)
    idxs = list(range(0, len(files), step))[:max_frames]
    return [files[i] for i in idxs]


def encode_image_base64(path: str) -> Optional[str]:
    """Codifica imagem em base64."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Erro ao codificar imagem {path}: {e}")
        return None


def get_image_mime_type(path: str) -> str:
    """Retorna o MIME type da imagem."""
    ext = Path(path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
    }
    return mime_types.get(ext, 'image/jpeg')


def build_analysis_prompt(creative: Dict, num_images: int) -> str:
    """Constrói o prompt para análise do criativo."""
    kpis = creative.get("kpis", {})
    name_fields = creative.get("name_fields", {})
    status = creative.get("status", "UNKNOWN")
    media_type = creative.get("media_type", "unknown")
    
    # Determina classificação de performance
    score = kpis.get("performance_score", 0)
    try:
        score_num = float(score) if score else 0
    except:
        score_num = 0
    
    if score_num >= 7:
        perf_class = "TOP PERFORMER (score >= 7)"
    elif score_num >= 4:
        perf_class = "MÉDIO (score 4-7)"
    else:
        perf_class = "BAIXO DESEMPENHO (score < 4)"
    
    # Métricas de vídeo
    video_section = ""
    if media_type == "video":
        video_section = f"""
**Métricas de Vídeo:**
- Hook Rate: {kpis.get('hook_rate_pct', 'N/A')} (vs média: {kpis.get('hook_rate_vs_avg', 'N/A')})
- Retenção 25%: {kpis.get('video_p25', 'N/A')}
- Retenção 50%: {kpis.get('video_p50', 'N/A')}
- Retenção 75%: {kpis.get('video_p75', 'N/A')}
- Retenção 100%: {kpis.get('video_p100', 'N/A')}
"""
    
    return f"""
Você é um especialista em criativos de performance para Meta Ads (Facebook/Instagram).
Analise este criativo de forma detalhada, correlacionando os elementos visuais com o score de performance.

## CONTEXTO DO ANÚNCIO

**Identificação:**
- Ad ID: {creative.get('ad_id', 'N/A')}
- Nome: {creative.get('ad_name', 'N/A')}
- Tipo de Mídia: {media_type.upper()}
- Status: {status}
- Campanha: {kpis.get('campaign_name', 'N/A')}

**Nomenclatura Parseada:**
{json.dumps(name_fields, ensure_ascii=False, indent=2) if name_fields else 'N/A'}

## PERFORMANCE (Score: {score}/10 - {perf_class})

**KPIs Principais:**
- Performance Score: {score}/10
- CAC: {kpis.get('cac_br', 'N/A')} (vs média: {kpis.get('cac_vs_avg', 'N/A')})
- CPM: {kpis.get('cpm_br', 'N/A')} (vs média: {kpis.get('cpm_vs_avg', 'N/A')})
- CPC: {kpis.get('cpc_br', 'N/A')} (vs média: {kpis.get('cpc_vs_avg', 'N/A')})
- CTR: {kpis.get('ctr_pct', 'N/A')}
- Connect Rate: {kpis.get('connect_rate_pct', 'N/A')}
- Bounce Rate: {kpis.get('bounce_rate_pct', 'N/A')}

**Volume:**
- Gasto: {kpis.get('spend_br', 'N/A')} ({kpis.get('share_spend', 'N/A')} do total)
- Impressões: {format_number(kpis.get('impressions', 0))}
- Cliques: {format_number(kpis.get('clicks', 0))}
- Compras: {kpis.get('purchases', 'N/A')}
{video_section}
**Placement:**
- Principal: {kpis.get('top_placement', 'N/A')} / {kpis.get('top_position', 'N/A')}

## TAREFA

Você está recebendo {num_images} {'frames do vídeo' if media_type == 'video' else 'imagem(ns)'} deste criativo.

Analise e responda:

1. **Descrição Visual Detalhada:**
   - O que aparece na(s) imagem(ns)?
   - Pessoas, produtos, textos, cores dominantes
   - Qualidade de produção (alta/média/baixa)
   - Tipo de conteúdo (UGC, profissional, animação, etc.)

2. **Correlação Visual x Performance:**
   - Por que este criativo tem score {score}/10?
   - Quais elementos visuais CONTRIBUEM para a performance?
   - Quais elementos visuais PREJUDICAM a performance?
   - O que explica o CAC de {kpis.get('cac_br', 'N/A')}?

3. **Diagnóstico de Elementos (0-10 cada):**
   - Hook/Gancho inicial
   - Clareza da oferta
   - Prova social
   - Legibilidade do texto
   - Call to Action

4. **Classificação:**
   - Tema principal
   - Tipo de narrativa
   - Tom de voz
   - Mensagem principal
   - Tipo de produção

5. **Recomendações:**
   - 3 melhorias específicas para este criativo
   - O que MANTER se for criar variações
   - O que MUDAR para melhorar o score

## FORMATO DE SAÍDA (JSON)

Responda APENAS com um JSON válido:

{{
  "visual_description": {{
    "summary": "Descrição geral em 2-3 frases",
    "elements": ["elemento1", "elemento2"],
    "text_on_screen": ["texto1", "texto2"],
    "colors": ["cor1", "cor2"],
    "people": "descrição das pessoas",
    "product": "descrição do produto",
    "production_quality": "alta|média|baixa",
    "content_type": "UGC|Profissional|Animação|Estático"
  }},
  "performance_correlation": {{
    "score_explanation": "Por que tem score {score}",
    "positive_elements": ["elemento que ajuda1", "elemento que ajuda2"],
    "negative_elements": ["elemento que prejudica1", "elemento que prejudica2"],
    "cac_explanation": "O que explica o CAC"
  }},
  "element_scores": {{
    "hook": {{"score": 0, "reason": "..."}},
    "offer_clarity": {{"score": 0, "reason": "..."}},
    "social_proof": {{"score": 0, "reason": "..."}},
    "text_legibility": {{"score": 0, "reason": "..."}},
    "cta": {{"score": 0, "reason": "..."}}
  }},
  "classification": {{
    "main_theme": "...",
    "narrative_type": "...",
    "tone_of_voice": "...",
    "main_message": "...",
    "production_type": "..."
  }},
  "recommendations": {{
    "improvements": ["melhoria1", "melhoria2", "melhoria3"],
    "keep": ["manter1", "manter2"],
    "change": ["mudar1", "mudar2"]
  }},
  "overall_assessment": {{
    "strengths": ["ponto forte1", "ponto forte2"],
    "weaknesses": ["ponto fraco1", "ponto fraco2"],
    "verdict": "Resumo final em 1 frase"
  }}
}}
""".strip()


def call_openai_api(prompt: str, image_paths: List[str]) -> Dict:
    """Chama a API da OpenAI com o prompt e imagens."""
    # Determinar URL
    if AI_ENDPOINT_URL and AI_ENDPOINT_URL.startswith("https://") and "/chat/completions" in AI_ENDPOINT_URL:
        url = AI_ENDPOINT_URL
    else:
        url = DEFAULT_API_URL
    
    logger.info(f"  Usando URL da API: {url}")
    logger.info(f"  Modelo: {AI_MODEL}")
    logger.info(f"  Imagens a enviar: {len(image_paths)}")
    
    # Monta o conteúdo com texto e imagens
    content = [{"type": "text", "text": prompt}]
    
    images_added = 0
    for img_path in image_paths:
        b64 = encode_image_base64(img_path)
        if b64:
            mime_type = get_image_mime_type(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64}",
                    "detail": "low"
                }
            })
            images_added += 1
            logger.info(f"    + Imagem adicionada: {Path(img_path).name}")
    
    if images_added == 0:
        raise ValueError("Nenhuma imagem foi codificada com sucesso")
    
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
    ad_name = creative.get("ad_name", "")
    media_type = creative.get("media_type", "unknown")
    
    logger.info(f"Buscando mídia para ad_id {ad_id}...")
    
    # Busca arquivos de mídia
    all_files = find_media_files(creative)
    
    if not all_files:
        logger.warning(f"  Nenhuma mídia encontrada para ad_id {ad_id}")
        return None
    
    # Seleciona frames para enviar
    selected_files = select_frames(all_files, MAX_FRAMES)
    logger.info(f"  Selecionados {len(selected_files)} de {len(all_files)} arquivos")
    
    # Constrói prompt
    prompt = build_analysis_prompt(creative, len(selected_files))
    
    # Chama a IA
    try:
        analysis = call_openai_api(prompt, selected_files)
        return {
            "ad_id": ad_id,
            "creative_id": creative.get("creative_id", ""),
            "ad_name": ad_name,
            "status": creative.get("status", ""),
            "media_type": media_type,
            "kpis": creative.get("kpis", {}),
            "analysis": analysis,
            "analyzed_at": now_utc_iso(),
            "images_analyzed": len(selected_files),
            "image_paths": selected_files,
        }
    except Exception as e:
        logger.error(f"  Erro ao analisar ad_id {ad_id}: {e}")
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
    
    # Carrega backlog (para evitar reprocessar)
    backlog = load_json(BACKLOG_PATH) or {"processed_creatives": {}, "last_updated": None}
    
    creatives = consolidated.get("creatives", [])
    logger.info(f"Total de criativos no consolidado: {len(creatives)}")
    
    # Limpa arquivo de resultados anterior para esta execução
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
    
    # Contadores
    analyzed = 0
    failed = 0
    skipped = 0
    no_media = 0
    
    # Resultados por categoria
    results_all = []
    
    for i, creative in enumerate(creatives):
        ad_id = creative.get("ad_id", "")
        sha256 = creative.get("sha256", "")
        creative_key = f"{ad_id}_{sha256}"
        status = creative.get("status", "UNKNOWN")
        score = creative.get("kpis", {}).get("performance_score", 0)
        
        logger.info(f"\n[{i+1}/{len(creatives)}] Processando: {ad_id} (status={status}, score={score})")
        
        # Verifica se já foi processado
        if creative_key in backlog["processed_creatives"]:
            logger.info(f"  Já processado anteriormente, pulando...")
            skipped += 1
            continue
        
        # Analisa o criativo
        result = analyze_creative(creative)
        
        if result:
            # Salva resultado
            append_jsonl(RESULTS_PATH, result)
            results_all.append(result)
            
            # Atualiza backlog
            backlog["processed_creatives"][creative_key] = {
                "ad_id": ad_id,
                "analyzed_at": result["analyzed_at"],
                "status": status,
                "score": score,
            }
            
            analyzed += 1
            logger.info(f"  ✅ Analisado com sucesso!")
        else:
            # Verifica se foi por falta de mídia
            all_files = find_media_files(creative)
            if not all_files:
                no_media += 1
                logger.warning(f"  ⚠️ Sem mídia disponível")
            else:
                failed += 1
                logger.warning(f"  ❌ Falha na análise")
    
    # Salva backlog atualizado
    backlog["last_updated"] = run_at
    save_json(BACKLOG_PATH, backlog)
    
    # Gera resumo
    summary_lines = [
        "=" * 60,
        "RESUMO DA ANÁLISE DE IA",
        "=" * 60,
        f"Data/Hora (UTC): {run_at}",
        f"Total de criativos: {len(creatives)}",
        "",
        f"✅ Analisados nesta execução: {analyzed}",
        f"⏭️ Pulados (já processados): {skipped}",
        f"⚠️ Sem mídia disponível: {no_media}",
        f"❌ Falhas: {failed}",
        "",
        "=" * 60,
        "CRIATIVOS ANALISADOS:",
        "=" * 60,
    ]
    
    # Agrupa por score
    top_performers = [r for r in results_all if float(r.get("kpis", {}).get("performance_score", 0) or 0) >= 7]
    medium = [r for r in results_all if 4 <= float(r.get("kpis", {}).get("performance_score", 0) or 0) < 7]
    low = [r for r in results_all if float(r.get("kpis", {}).get("performance_score", 0) or 0) < 4]
    
    summary_lines.append(f"\n🏆 TOP PERFORMERS (score >= 7): {len(top_performers)}")
    for r in top_performers[:5]:
        summary_lines.append(f"  - {r.get('ad_name', '')[:50]}... (score: {r.get('kpis', {}).get('performance_score', 'N/A')})")
    
    summary_lines.append(f"\n📊 MÉDIOS (score 4-7): {len(medium)}")
    for r in medium[:5]:
        summary_lines.append(f"  - {r.get('ad_name', '')[:50]}... (score: {r.get('kpis', {}).get('performance_score', 'N/A')})")
    
    summary_lines.append(f"\n⚠️ BAIXO DESEMPENHO (score < 4): {len(low)}")
    for r in low[:5]:
        summary_lines.append(f"  - {r.get('ad_name', '')[:50]}... (score: {r.get('kpis', {}).get('performance_score', 'N/A')})")
    
    summary_lines.append("")
    summary_lines.append("=" * 60)
    
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    
    logger.info("\n" + "\n".join(summary_lines))
    logger.info(f"\nResultados salvos em: {RESULTS_PATH}")
    logger.info(f"Resumo salvo em: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
