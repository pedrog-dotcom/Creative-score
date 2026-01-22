#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_playbook.py - V2 (Reestruturado)

Gera e atualiza o Playbook de Boas Práticas Criativas com base em:
- Análises de IA de TODOS os criativos
- Histórico acumulado de execuções anteriores
- Padrões identificados entre top performers e criativos pausados

O playbook fica cada vez melhor a cada execução, aprendendo com os dados.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from collections import defaultdict

import requests

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("playbook_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurações
ANALYSIS_RESULTS_PATH = Path("creatives_output/analysis/analysis_results.jsonl")
PLAYBOOK_HISTORY_PATH = Path("creatives_output/playbook_history.json")
PLAYBOOK_OUTPUT_PATH = Path("creatives_output/creative_playbook.md")
CONSOLIDATED_PATH = Path("creatives_output/consolidated_data.json")

# Configurações de IA
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_ENDPOINT_URL = os.getenv("AI_ENDPOINT_URL", "").strip()
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o")
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT_SECONDS", "180"))

DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar {path}: {e}")
        return None


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    results = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except Exception as e:
        logger.error(f"Erro ao carregar {path}: {e}")
    return results


def aggregate_patterns(analyses: List[Dict]) -> Dict:
    patterns = {
        "top_performers": {
            "count": 0,
            "hook_types": defaultdict(int),
            "narrative_types": defaultdict(int),
            "tones": defaultdict(int),
            "production_types": defaultdict(int),
            "themes": defaultdict(int),
            "avg_scores": {"hook": [], "offer_clarity": [], "social_proof": [], "text_legibility": [], "cta": []},
            "common_improvements": defaultdict(int),
            "examples": [],
        },
        "paused": {
            "count": 0,
            "hook_types": defaultdict(int),
            "narrative_types": defaultdict(int),
            "tones": defaultdict(int),
            "production_types": defaultdict(int),
            "themes": defaultdict(int),
            "avg_scores": {"hook": [], "offer_clarity": [], "social_proof": [], "text_legibility": [], "cta": []},
            "common_problems": defaultdict(int),
            "examples": [],
        },
        "active": {
            "count": 0,
            "hook_types": defaultdict(int),
            "narrative_types": defaultdict(int),
            "tones": defaultdict(int),
            "production_types": defaultdict(int),
            "themes": defaultdict(int),
            "avg_scores": {"hook": [], "offer_clarity": [], "social_proof": [], "text_legibility": [], "cta": []},
            "examples": [],
        },
    }
    
    for analysis in analyses:
        status = analysis.get("status", "ACTIVE")
        ai_analysis = analysis.get("analysis", {})
        
        if "TOP" in status:
            category = "top_performers"
        elif "PAUSED" in status:
            category = "paused"
        else:
            category = "active"
        
        patterns[category]["count"] += 1
        
        classification = ai_analysis.get("classification", {})
        if classification:
            hook_type = classification.get("primary_hook_type", "")
            if hook_type:
                patterns[category]["hook_types"][hook_type] += 1
            
            narrative = classification.get("narrative_type", "")
            if narrative:
                patterns[category]["narrative_types"][narrative] += 1
            
            tone = classification.get("tone_of_voice", "")
            if tone:
                patterns[category]["tones"][tone] += 1
            
            production = classification.get("production_type", "")
            if production:
                patterns[category]["production_types"][production] += 1
            
            theme = classification.get("main_theme", "")
            if theme:
                patterns[category]["themes"][theme] += 1
        
        visual_diagnosis = ai_analysis.get("visual_diagnosis", {})
        for metric in ["hook", "offer_clarity", "social_proof", "text_legibility", "cta"]:
            score_data = visual_diagnosis.get(metric, {})
            score = score_data.get("score", 0) if isinstance(score_data, dict) else 0
            if score > 0:
                patterns[category]["avg_scores"][metric].append(score)
        
        suggestions = ai_analysis.get("suggestions", {})
        if category == "top_performers":
            for improvement in suggestions.get("improvements", []):
                patterns[category]["common_improvements"][improvement] += 1
        elif category == "paused":
            perf_diagnosis = ai_analysis.get("performance_diagnosis", {})
            problem = perf_diagnosis.get("status_explanation", "")
            if problem:
                patterns[category]["common_problems"][problem] += 1
        
        if len(patterns[category]["examples"]) < 5:
            patterns[category]["examples"].append({
                "ad_id": analysis.get("ad_id", ""),
                "ad_name": analysis.get("ad_name", "")[:50],
                "score": analysis.get("kpis", {}).get("performance_score", 0),
                "classification": classification,
            })
    
    for category in patterns:
        for metric in patterns[category]["avg_scores"]:
            scores = patterns[category]["avg_scores"][metric]
            if scores:
                patterns[category]["avg_scores"][metric] = round(sum(scores) / len(scores), 2)
            else:
                patterns[category]["avg_scores"][metric] = 0
    
    return patterns


def call_openai_for_playbook(patterns: Dict, history: Dict) -> str:
    if AI_ENDPOINT_URL and AI_ENDPOINT_URL.startswith("https://") and "/chat/completions" in AI_ENDPOINT_URL:
        url = AI_ENDPOINT_URL
    else:
        url = DEFAULT_API_URL
    
    historical_insights = history.get("accumulated_insights", [])
    historical_context = ""
    if historical_insights:
        historical_context = f"\n## INSIGHTS HISTORICOS\n{json.dumps(historical_insights[-10:], ensure_ascii=False, indent=2)}\n"
    
    prompt = f"""
Voce e um especialista em criativos de performance para Meta Ads.
Gere um Playbook de Boas Praticas Criativas baseado nos padroes abaixo.

## TOP PERFORMERS ({patterns['top_performers']['count']} criativos)
Hooks: {json.dumps(dict(patterns['top_performers']['hook_types']), ensure_ascii=False)}
Narrativas: {json.dumps(dict(patterns['top_performers']['narrative_types']), ensure_ascii=False)}
Tons: {json.dumps(dict(patterns['top_performers']['tones']), ensure_ascii=False)}
Scores Medios: {json.dumps(patterns['top_performers']['avg_scores'], ensure_ascii=False)}

## PAUSADOS ({patterns['paused']['count']} criativos)
Hooks: {json.dumps(dict(patterns['paused']['hook_types']), ensure_ascii=False)}
Scores Medios: {json.dumps(patterns['paused']['avg_scores'], ensure_ascii=False)}
Problemas: {json.dumps(dict(patterns['paused']['common_problems']), ensure_ascii=False)}

## ATIVOS ({patterns['active']['count']} criativos)
Hooks: {json.dumps(dict(patterns['active']['hook_types']), ensure_ascii=False)}
Scores Medios: {json.dumps(patterns['active']['avg_scores'], ensure_ascii=False)}
{historical_context}

Gere um Playbook em Markdown com:
1. Resumo Executivo
2. O que Funciona (Top Performers)
3. O que Evitar (Pausados)
4. Checklist de Criacao
5. Recomendacoes de Hook
6. Recomendacoes de Narrativa
7. Testes A/B Sugeridos
8. Metricas de Referencia
9. Proximos Passos
"""

    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI_API_KEY}",
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=AI_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Erro ao gerar playbook via IA: {e}")
        return None


def generate_fallback_playbook(patterns: Dict) -> str:
    top = patterns["top_performers"]
    paused = patterns["paused"]
    
    top_hooks = sorted(top["hook_types"].items(), key=lambda x: x[1], reverse=True)[:3]
    top_narratives = sorted(top["narrative_types"].items(), key=lambda x: x[1], reverse=True)[:3]
    top_tones = sorted(top["tones"].items(), key=lambda x: x[1], reverse=True)[:3]
    paused_hooks = sorted(paused["hook_types"].items(), key=lambda x: x[1], reverse=True)[:3]
    
    hooks_str = "\n".join([f"- **{h[0]}**: {h[1]} ocorrencias" for h in top_hooks]) if top_hooks else "- Dados insuficientes"
    narratives_str = "\n".join([f"- **{n[0]}**: {n[1]} ocorrencias" for n in top_narratives]) if top_narratives else "- Dados insuficientes"
    tones_str = "\n".join([f"- **{t[0]}**: {t[1]} ocorrencias" for t in top_tones]) if top_tones else "- Dados insuficientes"
    paused_hooks_str = "\n".join([f"- **{h[0]}**: {h[1]} ocorrencias em pausados" for h in paused_hooks]) if paused_hooks else "- Dados insuficientes"
    
    playbook = f"""# Playbook de Boas Praticas Criativas

*Atualizado em: {now_utc_iso()}*

---

## Resumo Executivo

- **Top Performers analisados:** {top['count']}
- **Criativos Pausados analisados:** {paused['count']}
- **Criativos Ativos analisados:** {patterns['active']['count']}

---

## O que Funciona (Top Performers)

### Tipos de Hook Mais Efetivos
{hooks_str}

### Narrativas que Performam
{narratives_str}

### Tons de Voz Efetivos
{tones_str}

### Scores Medios dos Top Performers
- Hook: **{top['avg_scores']['hook']}/10**
- Clareza da Oferta: **{top['avg_scores']['offer_clarity']}/10**
- Prova Social: **{top['avg_scores']['social_proof']}/10**
- Legibilidade: **{top['avg_scores']['text_legibility']}/10**
- CTA: **{top['avg_scores']['cta']}/10**

---

## O que Evitar (Criativos Pausados)

### Tipos de Hook a Evitar
{paused_hooks_str}

### Scores Medios dos Pausados
- Hook: **{paused['avg_scores']['hook']}/10**
- Clareza da Oferta: **{paused['avg_scores']['offer_clarity']}/10**
- Prova Social: **{paused['avg_scores']['social_proof']}/10**
- Legibilidade: **{paused['avg_scores']['text_legibility']}/10**
- CTA: **{paused['avg_scores']['cta']}/10**

---

## Checklist de Criacao

- [ ] Hook forte nos primeiros 3 segundos
- [ ] Oferta clara e direta
- [ ] Prova social visivel
- [ ] Texto legivel em mobile
- [ ] CTA claro e urgente
- [ ] Qualidade de producao adequada

---

## Metricas de Referencia

| Metrica | Top Performers | Pausados | Diferenca |
|---------|---------------|----------|-----------|
| Hook Score | {top['avg_scores']['hook']} | {paused['avg_scores']['hook']} | {round(top['avg_scores']['hook'] - paused['avg_scores']['hook'], 2)} |
| Oferta | {top['avg_scores']['offer_clarity']} | {paused['avg_scores']['offer_clarity']} | {round(top['avg_scores']['offer_clarity'] - paused['avg_scores']['offer_clarity'], 2)} |
| Prova Social | {top['avg_scores']['social_proof']} | {paused['avg_scores']['social_proof']} | {round(top['avg_scores']['social_proof'] - paused['avg_scores']['social_proof'], 2)} |
| Legibilidade | {top['avg_scores']['text_legibility']} | {paused['avg_scores']['text_legibility']} | {round(top['avg_scores']['text_legibility'] - paused['avg_scores']['text_legibility'], 2)} |
| CTA | {top['avg_scores']['cta']} | {paused['avg_scores']['cta']} | {round(top['avg_scores']['cta'] - paused['avg_scores']['cta'], 2)} |

---

*Este playbook e atualizado automaticamente a cada execucao do pipeline.*
"""
    return playbook


def main():
    run_at = now_utc_iso()
    logger.info("=" * 60)
    logger.info(f"Atualizando Playbook em {run_at}")
    logger.info("=" * 60)
    
    analyses = load_jsonl(ANALYSIS_RESULTS_PATH)
    if not analyses:
        logger.warning("Nenhuma analise encontrada. Abortando.")
        return
    
    logger.info(f"Analises carregadas: {len(analyses)}")
    
    history = load_json(PLAYBOOK_HISTORY_PATH) or {"runs": [], "accumulated_insights": []}
    
    patterns = aggregate_patterns(analyses)
    
    logger.info(f"Padroes agregados:")
    logger.info(f"  - Top Performers: {patterns['top_performers']['count']}")
    logger.info(f"  - Pausados: {patterns['paused']['count']}")
    logger.info(f"  - Ativos: {patterns['active']['count']}")
    
    playbook_content = None
    
    if AI_API_KEY:
        logger.info("Gerando playbook via IA...")
        playbook_content = call_openai_for_playbook(patterns, history)
    
    if not playbook_content:
        logger.info("Gerando playbook fallback...")
        playbook_content = generate_fallback_playbook(patterns)
    
    PLAYBOOK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PLAYBOOK_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(playbook_content)
    
    logger.info(f"Playbook salvo em: {PLAYBOOK_OUTPUT_PATH}")
    
    run_summary = {
        "run_at": run_at,
        "total_analyses": len(analyses),
        "top_performers": patterns["top_performers"]["count"],
        "paused": patterns["paused"]["count"],
        "active": patterns["active"]["count"],
        "top_hooks": dict(patterns["top_performers"]["hook_types"]),
        "avg_scores_top": patterns["top_performers"]["avg_scores"],
        "avg_scores_paused": patterns["paused"]["avg_scores"],
    }
    
    history["runs"].append(run_summary)
    
    new_insight = {
        "date": run_at,
        "insight": f"Top hooks: {list(patterns['top_performers']['hook_types'].keys())[:3]}",
        "top_count": patterns["top_performers"]["count"],
        "paused_count": patterns["paused"]["count"],
    }
    history["accumulated_insights"].append(new_insight)
    history["accumulated_insights"] = history["accumulated_insights"][-50:]
    
    save_json(PLAYBOOK_HISTORY_PATH, history)
    
    logger.info("=" * 60)
    logger.info("Playbook atualizado com sucesso!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
