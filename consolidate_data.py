#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consolidate_data.py

Este script consolida os dados de criativos (catalog.csv) com os scores de performance
(creative_score_automation.csv) em um único arquivo JSON estruturado.

Fluxo:
1. Lê o catálogo de criativos (com caminhos de vídeos/imagens e frames)
2. Lê os scores de performance (KPIs calculados)
3. Faz o match por ad_id
4. Gera um JSON consolidado com todos os dados necessários para a IA

Saída:
- creatives_output/consolidated_data.json
"""

import os
import json
import logging
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import pandas as pd

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("consolidate_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurações
CATALOG_PATH = os.getenv("CREATIVES_CATALOG_PATH", "creatives_output/catalog.csv")
SCORE_PATH = os.getenv("SCORE_LATEST_PATH", "creative_score_automation.csv")
OUTPUT_DIR = Path("creatives_output")
OUTPUT_JSON = OUTPUT_DIR / "consolidated_data.json"
BACKLOG_PATH = OUTPUT_DIR / "backlog.json"

# Campos do nome do anúncio (nomenclatura padrão)
NAME_FIELDS = [
    "AdCode", "AdName", "Variation", "Month_live", "Market", "Asset_type",
    "Channel", "Partnership_Ad", "Persona", "Actor_Affiliate_Expert",
    "Resource", "Core_Message", "Leave_Blank", "Tone", "Anchor",
    "Overarching", "Value_Prop", "Key_Outcome", "Production", "Style_Setting",
]


def now_utc_iso() -> str:
    """Retorna timestamp UTC no formato ISO."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_name(name: str) -> str:
    """Normaliza o nome do anúncio."""
    name = str(name or "").strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"_+", "_", name)
    return name


def parse_ad_name(ad_name: str) -> Dict[str, str]:
    """Parseia o nome do anúncio em campos estruturados."""
    ad_name = normalize_name(ad_name)
    parts = ad_name.split("_")
    out = {}
    for i, field in enumerate(NAME_FIELDS):
        if i < len(parts):
            out[field] = parts[i]
        else:
            out[field] = ""
    return out


def safe_float(value: Any, default: float = 0.0) -> float:
    """Converte valor para float de forma segura."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    """Converte valor para string de forma segura."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value).strip()


def money_br(value: float) -> str:
    """Formata valor como moeda brasileira."""
    try:
        s = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {s}"
    except:
        return "R$ 0,00"


def pct(value: float) -> str:
    """Formata valor como porcentagem."""
    try:
        return f"{value * 100:.2f}%"
    except:
        return "0.00%"


def load_backlog() -> Dict[str, Any]:
    """Carrega o backlog existente ou cria um novo."""
    if BACKLOG_PATH.exists():
        try:
            with open(BACKLOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"processed_creatives": {}, "last_updated": None}


def save_backlog(backlog: Dict[str, Any]) -> None:
    """Salva o backlog atualizado."""
    backlog["last_updated"] = now_utc_iso()
    with open(BACKLOG_PATH, "w", encoding="utf-8") as f:
        json.dump(backlog, f, ensure_ascii=False, indent=2)


def determine_status(score_row: Optional[pd.Series]) -> str:
    """Determina o status do criativo baseado nos dados de score."""
    if score_row is None:
        return "UNKNOWN"
    
    # Verifica se está pausado
    effective_status = safe_str(score_row.get("effective_status", "")).upper()
    if effective_status == "PAUSED":
        pause_reason = safe_str(score_row.get("pause_reason", "")).upper()
        if "HARD" in pause_reason:
            return "PAUSED_HARD_STOP"
        elif "SCORE" in pause_reason:
            return "PAUSED_LOW_SCORE"
        return "PAUSED"
    
    # Verifica se é top performer
    is_top5 = safe_str(score_row.get("is_top5", "")).lower()
    if is_top5 in ("1", "true", "yes"):
        return "TOP_PERFORMER"
    
    return "ACTIVE"


def consolidate() -> None:
    """Função principal de consolidação."""
    run_at = now_utc_iso()
    logger.info(f"Iniciando consolidação de dados em {run_at}")
    
    # Verifica se os arquivos existem
    if not Path(CATALOG_PATH).exists():
        logger.error(f"Catálogo não encontrado: {CATALOG_PATH}")
        return
    
    if not Path(SCORE_PATH).exists():
        logger.error(f"Score não encontrado: {SCORE_PATH}")
        return
    
    # Carrega os dados
    catalog = pd.read_csv(CATALOG_PATH)
    score = pd.read_csv(SCORE_PATH)
    
    logger.info(f"Catálogo carregado: {len(catalog)} criativos")
    logger.info(f"Score carregado: {len(score)} registros")
    
    # Normaliza IDs para string
    catalog["ad_id"] = catalog["ad_id"].astype(str)
    score["ad_id"] = score["ad_id"].astype(str)
    
    # Indexa score por ad_id
    score_idx = score.set_index("ad_id", drop=False)
    
    # Carrega backlog
    backlog = load_backlog()
    
    # Calcula métricas globais
    total_spend = safe_float(score["spend_std"].sum()) if "spend_std" in score.columns else 0.0
    avg_cpa = 0.0
    if "cac" in score.columns:
        cpa_valid = score[(score["cac"].notna()) & (score["cac"] > 0)]
        if not cpa_valid.empty:
            avg_cpa = float(cpa_valid["cac"].mean())
    
    # Consolida os dados
    consolidated = {
        "run_at": run_at,
        "metadata": {
            "total_creatives": len(catalog),
            "total_spend": money_br(total_spend),
            "avg_cpa": money_br(avg_cpa),
        },
        "creatives": []
    }
    
    matched = 0
    unmatched = 0
    
    for _, row in catalog.iterrows():
        ad_id = str(row["ad_id"])
        ad_name = safe_str(row.get("ad_name", ""))
        creative_id = safe_str(row.get("creative_id", ""))
        media_type = safe_str(row.get("media_type", "")).lower()
        local_path = safe_str(row.get("local_path", ""))
        frames_dir = safe_str(row.get("frames_dir", ""))
        sha256 = safe_str(row.get("sha256", ""))
        
        # Busca dados de score
        score_row = None
        if ad_id in score_idx.index:
            score_row = score_idx.loc[ad_id]
            if isinstance(score_row, pd.DataFrame):
                score_row = score_row.iloc[0]
            matched += 1
        else:
            unmatched += 1
        
        # Determina status
        status = determine_status(score_row)
        
        # Extrai KPIs
        kpis = {}
        if score_row is not None:
            spend = safe_float(score_row.get("spend_std", 0))
            cac = safe_float(score_row.get("cac", 0))
            performance_score = safe_float(score_row.get("performance_score", 0))
            
            kpis = {
                "performance_score": round(performance_score, 2) if performance_score > 0 else None,
                "spend": spend,
                "spend_br": money_br(spend),
                "share_spend": pct(spend / total_spend) if total_spend > 0 else "0.00%",
                "cac": cac,
                "cac_br": money_br(cac) if cac > 0 else "N/A",
                "cac_vs_avg": f"{cac / avg_cpa:.2f}x" if avg_cpa > 0 and cac > 0 else "N/A",
                "ctr": safe_float(score_row.get("ctr", 0)),
                "connect_rate": safe_float(score_row.get("connect_rate", 0)),
                "bounce_rate": safe_float(score_row.get("bounce_rate", 0)),
                "cost_per_checkout": safe_float(score_row.get("cost_per_checkout", 0)),
                "impressions": int(safe_float(score_row.get("impressions_std", 0))),
                "clicks": int(safe_float(score_row.get("clicks_std", 0))),
                "purchases": int(safe_float(score_row.get("purchase_std", 0))),
            }
        
        # Parseia nome do anúncio
        name_fields = parse_ad_name(ad_name)
        
        # Verifica se já foi processado pela IA
        creative_key = f"{ad_id}_{sha256}"
        already_processed = creative_key in backlog.get("processed_creatives", {})
        
        # Monta objeto consolidado
        creative_data = {
            "ad_id": ad_id,
            "creative_id": creative_id,
            "ad_name": ad_name,
            "media_type": media_type,
            "local_path": local_path,
            "frames_dir": frames_dir,
            "sha256": sha256,
            "status": status,
            "kpis": kpis,
            "name_fields": name_fields,
            "already_processed_by_ai": already_processed,
            "needs_ai_analysis": not already_processed,
        }
        
        consolidated["creatives"].append(creative_data)
    
    # Estatísticas
    consolidated["metadata"]["matched_with_score"] = matched
    consolidated["metadata"]["unmatched"] = unmatched
    consolidated["metadata"]["pending_ai_analysis"] = sum(
        1 for c in consolidated["creatives"] if c["needs_ai_analysis"]
    )
    
    # Salva JSON consolidado
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Dados consolidados salvos em: {OUTPUT_JSON}")
    logger.info(f"Total: {len(consolidated['creatives'])} criativos")
    logger.info(f"Match com score: {matched} | Sem match: {unmatched}")
    logger.info(f"Pendentes de análise IA: {consolidated['metadata']['pending_ai_analysis']}")


if __name__ == "__main__":
    consolidate()
