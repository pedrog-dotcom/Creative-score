#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_playbook.py

Este script gera e atualiza um playbook de boas práticas criativas,
baseado no histórico de análises e nos padrões identificados nos
criativos de melhor e pior desempenho.

Saídas:
- playbook/creative_playbook.md (Playbook em Markdown)
- playbook/playbook_data.json (Dados estruturados)
- playbook/insights_history.jsonl (Histórico de insights)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("update_playbook.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurações
ANALYSIS_RESULTS_PATH = Path("creatives_output/analysis/analysis_results.jsonl")
PLAYBOOK_DIR = Path("playbook")
PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)
PLAYBOOK_MD_PATH = PLAYBOOK_DIR / "creative_playbook.md"
PLAYBOOK_JSON_PATH = PLAYBOOK_DIR / "playbook_data.json"
INSIGHTS_HISTORY_PATH = PLAYBOOK_DIR / "insights_history.jsonl"


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


def append_jsonl(path: Path, obj: Dict) -> None:
    """Adiciona objeto ao arquivo JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_patterns(results: List[Dict], status_filter: str) -> Dict:
    """Extrai padrões dos criativos filtrados por status."""
    filtered = [r for r in results if status_filter in r.get("status", "")]
    
    if not filtered:
        return {}
    
    # Contadores
    narrative_types = Counter()
    tones = Counter()
    hook_types = Counter()
    production_types = Counter()
    themes = Counter()
    
    # Scores médios
    hook_scores = []
    offer_scores = []
    proof_scores = []
    legibility_scores = []
    cta_scores = []
    overall_scores = []
    
    # Sugestões mais comuns
    all_improvements = []
    all_tests = []
    
    for r in filtered:
        analysis = r.get("analysis", {})
        classification = analysis.get("classification", {})
        tags = analysis.get("structured_tags", {})
        suggestions = analysis.get("suggestions", {})
        
        # Classificações
        if classification.get("narrative_type"):
            narrative_types[classification["narrative_type"]] += 1
        if classification.get("tone_of_voice"):
            tones[classification["tone_of_voice"]] += 1
        if classification.get("primary_hook_type"):
            hook_types[classification["primary_hook_type"]] += 1
        if classification.get("production_type"):
            production_types[classification["production_type"]] += 1
        if classification.get("main_theme"):
            themes[classification["main_theme"]] += 1
        
        # Scores
        if tags.get("hook_strength"):
            hook_scores.append(tags["hook_strength"])
        if tags.get("offer_clarity"):
            offer_scores.append(tags["offer_clarity"])
        if tags.get("social_proof_strength"):
            proof_scores.append(tags["social_proof_strength"])
        if tags.get("text_legibility"):
            legibility_scores.append(tags["text_legibility"])
        if tags.get("cta_strength"):
            cta_scores.append(tags["cta_strength"])
        if tags.get("overall_quality"):
            overall_scores.append(tags["overall_quality"])
        
        # Sugestões
        all_improvements.extend(suggestions.get("improvements", []))
        all_tests.extend(suggestions.get("ab_tests", []))
    
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    return {
        "count": len(filtered),
        "narrative_types": dict(narrative_types.most_common(5)),
        "tones": dict(tones.most_common(5)),
        "hook_types": dict(hook_types.most_common(5)),
        "production_types": dict(production_types.most_common(5)),
        "themes": dict(themes.most_common(10)),
        "avg_scores": {
            "hook": round(avg(hook_scores), 2),
            "offer_clarity": round(avg(offer_scores), 2),
            "social_proof": round(avg(proof_scores), 2),
            "legibility": round(avg(legibility_scores), 2),
            "cta": round(avg(cta_scores), 2),
            "overall": round(avg(overall_scores), 2),
        },
        "top_improvements": Counter(all_improvements).most_common(10),
        "top_tests": Counter(all_tests).most_common(10),
    }


def generate_playbook_md(top_patterns: Dict, paused_patterns: Dict, run_at: str) -> str:
    """Gera o playbook em Markdown."""
    
    md = f"""# 📚 Playbook de Boas Práticas Criativas

**Última Atualização:** {run_at}

Este playbook é gerado automaticamente com base na análise de IA dos criativos
de melhor e pior desempenho. Use-o como guia para criar novos criativos e
otimizar os existentes.

---

## 📊 Visão Geral

| Métrica | Top Performers | Pausados |
|---------|----------------|----------|
| Total Analisado | {top_patterns.get('count', 0)} | {paused_patterns.get('count', 0)} |
| Score Médio | {top_patterns.get('avg_scores', {}).get('overall', 0)}/10 | {paused_patterns.get('avg_scores', {}).get('overall', 0)}/10 |

---

## ✅ O Que Funciona (Baseado nos Top Performers)

### 🎯 Tipos de Narrativa que Performam

"""
    
    for narrative, count in top_patterns.get("narrative_types", {}).items():
        md += f"- **{narrative}**: {count} criativos\n"
    
    md += """
### 🎤 Tons de Voz Eficazes

"""
    
    for tone, count in top_patterns.get("tones", {}).items():
        md += f"- **{tone}**: {count} criativos\n"
    
    md += """
### 🪝 Tipos de Hook que Convertem

"""
    
    for hook, count in top_patterns.get("hook_types", {}).items():
        md += f"- **{hook}**: {count} criativos\n"
    
    md += """
### 🎬 Tipos de Produção

"""
    
    for prod, count in top_patterns.get("production_types", {}).items():
        md += f"- **{prod}**: {count} criativos\n"
    
    md += f"""
### 📈 Scores Médios dos Top Performers

| Aspecto | Score Médio |
|---------|-------------|
| Hook | {top_patterns.get('avg_scores', {}).get('hook', 0)}/10 |
| Clareza da Oferta | {top_patterns.get('avg_scores', {}).get('offer_clarity', 0)}/10 |
| Prova Social | {top_patterns.get('avg_scores', {}).get('social_proof', 0)}/10 |
| Legibilidade | {top_patterns.get('avg_scores', {}).get('legibility', 0)}/10 |
| CTA | {top_patterns.get('avg_scores', {}).get('cta', 0)}/10 |

---

## ❌ O Que Evitar (Baseado nos Criativos Pausados)

### 🚫 Padrões Problemáticos

"""
    
    for narrative, count in paused_patterns.get("narrative_types", {}).items():
        md += f"- **{narrative}**: {count} criativos pausados\n"
    
    md += f"""
### 📉 Scores Médios dos Pausados

| Aspecto | Score Médio |
|---------|-------------|
| Hook | {paused_patterns.get('avg_scores', {}).get('hook', 0)}/10 |
| Clareza da Oferta | {paused_patterns.get('avg_scores', {}).get('offer_clarity', 0)}/10 |
| Prova Social | {paused_patterns.get('avg_scores', {}).get('social_proof', 0)}/10 |
| Legibilidade | {paused_patterns.get('avg_scores', {}).get('legibility', 0)}/10 |
| CTA | {paused_patterns.get('avg_scores', {}).get('cta', 0)}/10 |

---

## 💡 Recomendações de Melhoria (Mais Frequentes)

"""
    
    for improvement, count in top_patterns.get("top_improvements", []):
        md += f"1. **{improvement}** (mencionado {count}x)\n"
    
    md += """
---

## 🧪 Testes A/B Recomendados (Mais Frequentes)

"""
    
    for test, count in top_patterns.get("top_tests", []):
        md += f"1. **{test}** (sugerido {count}x)\n"
    
    md += """
---

## 📋 Checklist para Novos Criativos

Antes de lançar um novo criativo, verifique:

### Hook (Primeiros 3 segundos)
- [ ] O hook captura atenção imediatamente?
- [ ] Há um elemento visual ou sonoro impactante?
- [ ] O problema/benefício é claro desde o início?

### Oferta
- [ ] A oferta está clara e visível?
- [ ] O preço/desconto está destacado (se aplicável)?
- [ ] Há urgência ou escassez?

### Prova Social
- [ ] Há depoimentos ou resultados reais?
- [ ] Os números são específicos e críveis?
- [ ] Há elementos de autoridade?

### Produção
- [ ] A qualidade de áudio está boa?
- [ ] Os textos são legíveis em mobile?
- [ ] O ritmo mantém a atenção?

### CTA
- [ ] O CTA está claro e visível?
- [ ] Há um motivo para agir agora?
- [ ] O próximo passo é óbvio?

---

## 📊 Temas que Mais Performam

"""
    
    for theme, count in list(top_patterns.get("themes", {}).items())[:10]:
        md += f"- **{theme}**: {count} criativos\n"
    
    md += """
---

## 📝 Notas

Este playbook é atualizado automaticamente a cada execução do pipeline.
Os insights são baseados em análise de IA e dados históricos de performance.

Para melhores resultados:
1. Combine múltiplos elementos que funcionam
2. Teste variações dos padrões de sucesso
3. Evite os padrões identificados nos criativos pausados
4. Monitore os scores e ajuste conforme necessário

---

*Gerado automaticamente pelo Creative Score Pipeline*
"""
    
    return md


def main():
    """Função principal."""
    run_at = now_utc_iso()
    logger.info(f"Atualizando playbook em {run_at}")
    
    # Carrega resultados da análise
    results = load_jsonl(ANALYSIS_RESULTS_PATH)
    if not results:
        logger.warning("Nenhum resultado de análise encontrado")
        return
    
    logger.info(f"Encontrados {len(results)} criativos analisados")
    
    # Extrai padrões
    top_patterns = extract_patterns(results, "TOP_PERFORMER")
    paused_patterns = extract_patterns(results, "PAUSED")
    
    # Gera playbook MD
    playbook_md = generate_playbook_md(top_patterns, paused_patterns, run_at)
    
    with open(PLAYBOOK_MD_PATH, "w", encoding="utf-8") as f:
        f.write(playbook_md)
    
    # Salva dados estruturados
    playbook_data = {
        "generated_at": run_at,
        "total_analyzed": len(results),
        "top_performers": top_patterns,
        "paused": paused_patterns,
    }
    
    with open(PLAYBOOK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(playbook_data, f, ensure_ascii=False, indent=2)
    
    # Adiciona ao histórico de insights
    insight = {
        "run_at": run_at,
        "total_analyzed": len(results),
        "top_count": top_patterns.get("count", 0),
        "paused_count": paused_patterns.get("count", 0),
        "avg_top_score": top_patterns.get("avg_scores", {}).get("overall", 0),
        "avg_paused_score": paused_patterns.get("avg_scores", {}).get("overall", 0),
    }
    append_jsonl(INSIGHTS_HISTORY_PATH, insight)
    
    logger.info(f"Playbook MD salvo em: {PLAYBOOK_MD_PATH}")
    logger.info(f"Playbook JSON salvo em: {PLAYBOOK_JSON_PATH}")
    logger.info(f"Histórico atualizado em: {INSIGHTS_HISTORY_PATH}")


if __name__ == "__main__":
    main()
