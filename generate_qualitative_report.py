#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_qualitative_report.py

Este script gera um relatório qualitativo detalhado de cada criativo analisado,
explicando o motivo da nota e fornecendo insights acionáveis.

Saídas:
- reports/qualitative_report.md (Relatório em Markdown)
- reports/qualitative_report.json (Dados estruturados)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generate_report.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurações
ANALYSIS_RESULTS_PATH = Path("creatives_output/analysis/analysis_results.jsonl")
CONSOLIDATED_PATH = Path("creatives_output/consolidated_data.json")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_MD_PATH = REPORTS_DIR / "qualitative_report.md"
REPORT_JSON_PATH = REPORTS_DIR / "qualitative_report.json"


def now_utc_iso() -> str:
    """Retorna timestamp UTC no formato ISO."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_jsonl(path: Path) -> List[Dict]:
    """Carrega arquivo JSONL."""
    if not path.exists():
        return []
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except:
                    continue
    return results


def load_json(path: Path) -> Optional[Dict]:
    """Carrega arquivo JSON."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def get_score_interpretation(score: float) -> str:
    """Interpreta o score de 0-10."""
    if score >= 8:
        return "Excelente"
    elif score >= 6:
        return "Bom"
    elif score >= 4:
        return "Regular"
    elif score >= 2:
        return "Fraco"
    else:
        return "Crítico"


def generate_creative_analysis_md(result: Dict) -> str:
    """Gera a análise em Markdown para um criativo."""
    ad_name = result.get("ad_name", "N/A")
    ad_id = result.get("ad_id", "N/A")
    status = result.get("status", "N/A")
    kpis = result.get("kpis", {})
    analysis = result.get("analysis", {})
    
    # Extrai dados da análise
    video_desc = analysis.get("video_description", {})
    audio_desc = analysis.get("audio_description", {})
    visual_diag = analysis.get("visual_diagnosis", {})
    perf_diag = analysis.get("performance_diagnosis", {})
    classification = analysis.get("classification", {})
    suggestions = analysis.get("suggestions", {})
    tags = analysis.get("structured_tags", {})
    
    # Calcula média dos scores
    scores = [
        tags.get("hook_strength", 0),
        tags.get("offer_clarity", 0),
        tags.get("social_proof_strength", 0),
        tags.get("text_legibility", 0),
        tags.get("cta_strength", 0),
    ]
    avg_score = sum(scores) / len(scores) if scores else 0
    overall = tags.get("overall_quality", avg_score)
    
    md = f"""
---

## 📊 {ad_name[:60]}{'...' if len(ad_name) > 60 else ''}

**ID:** `{ad_id}`  
**Status:** `{status}`  
**Score Geral:** **{overall:.1f}/10** ({get_score_interpretation(overall)})

### 📈 KPIs de Performance

| Métrica | Valor |
|---------|-------|
| Performance Score | {kpis.get('performance_score', 'N/A')} |
| Gasto | {kpis.get('spend_br', 'N/A')} |
| CAC | {kpis.get('cac_br', 'N/A')} |
| CAC vs Média | {kpis.get('cac_vs_avg', 'N/A')} |
| CTR | {kpis.get('ctr', 'N/A'):.4f} |
| Impressões | {kpis.get('impressions', 'N/A'):,} |
| Compras | {kpis.get('purchases', 'N/A')} |

### 🎬 Descrição do Criativo

**Resumo:** {video_desc.get('summary', 'N/A')}

**Qualidade de Produção:** {video_desc.get('production_quality', 'N/A')}  
**Duração Estimada:** {video_desc.get('estimated_duration', 'N/A')}

**Elementos Visuais:**
"""
    
    for elem in video_desc.get("visual_elements", []):
        md += f"- {elem}\n"
    
    md += f"""
### 🔊 Descrição do Áudio

- **Narração:** {audio_desc.get('narration_type', 'N/A')} ({audio_desc.get('narration_tone', 'N/A')})
- **Música de Fundo:** {audio_desc.get('background_music', 'N/A')}
- **Estilo Musical:** {audio_desc.get('music_style', 'N/A')}
- **Mensagem Falada:** {audio_desc.get('spoken_message_summary', 'N/A')}

### 📊 Diagnóstico Visual

| Aspecto | Score | Interpretação | Justificativa |
|---------|-------|---------------|---------------|
| Hook (Gancho) | {visual_diag.get('hook', {}).get('score', 0)}/10 | {get_score_interpretation(visual_diag.get('hook', {}).get('score', 0))} | {visual_diag.get('hook', {}).get('justification', 'N/A')[:100]}... |
| Clareza da Oferta | {visual_diag.get('offer_clarity', {}).get('score', 0)}/10 | {get_score_interpretation(visual_diag.get('offer_clarity', {}).get('score', 0))} | {visual_diag.get('offer_clarity', {}).get('justification', 'N/A')[:100]}... |
| Prova Social | {visual_diag.get('social_proof', {}).get('score', 0)}/10 | {get_score_interpretation(visual_diag.get('social_proof', {}).get('score', 0))} | {visual_diag.get('social_proof', {}).get('justification', 'N/A')[:100]}... |
| Legibilidade | {visual_diag.get('text_legibility', {}).get('score', 0)}/10 | {get_score_interpretation(visual_diag.get('text_legibility', {}).get('score', 0))} | {visual_diag.get('text_legibility', {}).get('justification', 'N/A')[:100]}... |
| CTA | {visual_diag.get('cta', {}).get('score', 0)}/10 | {get_score_interpretation(visual_diag.get('cta', {}).get('score', 0))} | {visual_diag.get('cta', {}).get('justification', 'N/A')[:100]}... |

### 🎯 Classificação

| Atributo | Valor |
|----------|-------|
| Tema Principal | {classification.get('main_theme', 'N/A')} |
| Tipo de Narrativa | {classification.get('narrative_type', 'N/A')} |
| Tom de Voz | {classification.get('tone_of_voice', 'N/A')} |
| Mensagem Principal | {classification.get('main_message', 'N/A')} |
| Tipo de Produção | {classification.get('production_type', 'N/A')} |
| Tipo de Hook | {classification.get('primary_hook_type', 'N/A')} |

### 💡 Por que este criativo está {status}?

{perf_diag.get('status_explanation', 'N/A')}

**Correlação Visual-Métricas:** {perf_diag.get('visual_metric_correlation', 'N/A')}

### ✅ Sugestões de Melhoria

**Melhorias Recomendadas:**
"""
    
    for i, improvement in enumerate(suggestions.get("improvements", []), 1):
        md += f"{i}. {improvement}\n"
    
    md += "\n**Testes A/B Sugeridos:**\n"
    
    for i, test in enumerate(suggestions.get("ab_tests", []), 1):
        md += f"{i}. {test}\n"
    
    return md


def main():
    """Função principal."""
    run_at = now_utc_iso()
    logger.info(f"Gerando relatório qualitativo em {run_at}")
    
    # Carrega resultados da análise
    results = load_jsonl(ANALYSIS_RESULTS_PATH)
    if not results:
        logger.warning("Nenhum resultado de análise encontrado")
        return
    
    logger.info(f"Encontrados {len(results)} criativos analisados")
    
    # Separa por status
    paused = [r for r in results if "PAUSED" in r.get("status", "")]
    top_performers = [r for r in results if r.get("status") == "TOP_PERFORMER"]
    active = [r for r in results if r.get("status") == "ACTIVE"]
    
    # Gera relatório em Markdown
    md_content = f"""# 📊 Relatório Qualitativo de Criativos

**Gerado em:** {run_at}  
**Total de Criativos Analisados:** {len(results)}

## 📋 Sumário Executivo

| Categoria | Quantidade |
|-----------|------------|
| 🔴 Pausados | {len(paused)} |
| 🏆 Top Performers | {len(top_performers)} |
| 🟢 Ativos | {len(active)} |

---

# 🔴 Criativos Pausados

Estes criativos foram pausados por baixo desempenho ou hard stop.
A análise abaixo explica os motivos e sugere melhorias.
"""
    
    for result in paused:
        md_content += generate_creative_analysis_md(result)
    
    md_content += """

---

# 🏆 Top Performers

Estes são os criativos com melhor desempenho.
Entenda o que os torna eficazes para replicar em novos criativos.
"""
    
    for result in top_performers:
        md_content += generate_creative_analysis_md(result)
    
    md_content += """

---

# 🟢 Criativos Ativos

Criativos atualmente em veiculação.
Análise para otimização contínua.
"""
    
    for result in active[:20]:  # Limita a 20 para não ficar muito grande
        md_content += generate_creative_analysis_md(result)
    
    if len(active) > 20:
        md_content += f"\n\n*... e mais {len(active) - 20} criativos ativos não listados.*\n"
    
    # Salva relatório MD
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    # Gera relatório JSON estruturado
    report_json = {
        "generated_at": run_at,
        "total_analyzed": len(results),
        "summary": {
            "paused": len(paused),
            "top_performers": len(top_performers),
            "active": len(active),
        },
        "creatives": results,
    }
    
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Relatório MD salvo em: {REPORT_MD_PATH}")
    logger.info(f"Relatório JSON salvo em: {REPORT_JSON_PATH}")


if __name__ == "__main__":
    main()
