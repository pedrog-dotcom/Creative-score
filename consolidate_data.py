#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consolidate_data.py - V2 (Reestruturado)

Consolida os dados de criativos (catalog.csv) com os scores de performance
(creative_score_automation.csv) em um único arquivo JSON estruturado.

Novidades:
- Inclui TODAS as métricas: CPM, CPC, Hook Rate, Placement, etc.
- Identifica tipo de mídia (vídeo vs estático)
- Prepara dados completos para análise de IA
- Suporta criativos ativos e pausados
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
CATALOG_PATH = Path(os.getenv("CREATIVES_CATALOG_PATH", "creatives_output/catalog.csv"))
SCORE_PATH = Path(os.getenv("SCORE_LATEST_PATH", "creative_score_automation.csv"))
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


def determine_status(score_row: Optional[pd.Series], catalog_status: str) -> str:
    """Determina o status do criativo baseado nos dados."""
    # Primeiro, verificar o status do catálogo
    if catalog_status and catalog_status.upper() == "PAUSED":
        if score_row is not None:
            pause_reason = safe_str(score_row.get("pause_reason", "")).upper()
            if "HARD" in pause_reason:
                return "PAUSED_HARD_STOP"
            elif "SCORE" in pause_reason:
                return "PAUSED_LOW_SCORE"
        return "PAUSED"
    
    if score_row is not None:
        # Verifica se é top performer
        is_top5 = score_row.get("is_top5", False)
        if is_top5 == True or str(is_top5).lower() in ("1", "true", "yes"):
            return "TOP_PERFORMER"
    
    return "ACTIVE"


def consolidate() -> None:
    """Função principal de consolidação."""
    run_at = now_utc_iso()
    logger.info(f"Iniciando consolidação de dados em {run_at}")
    
    # Verifica se os arquivos existem
    if not CATALOG_PATH.exists():
        logger.error(f"Catálogo não encontrado: {CATALOG_PATH}")
        return
    
    if not SCORE_PATH.exists():
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
    
    avg_cpm = 0.0
    if "cpm_calc" in score.columns:
        cpm_valid = score[(score["cpm_calc"].notna()) & (score["cpm_calc"] > 0)]
        if not cpm_valid.empty:
            avg_cpm = float(cpm_valid["cpm_calc"].mean())
    
    avg_cpc = 0.0
    if "cpc_calc" in score.columns:
        cpc_valid = score[(score["cpc_calc"].notna()) & (score["cpc_calc"] > 0)]
        if not cpc_valid.empty:
            avg_cpc = float(cpc_valid["cpc_calc"].mean())
    
    avg_hook_rate = 0.0
    if "hook_rate" in score.columns:
        hr_valid = score[(score["hook_rate"].notna()) & (score["hook_rate"] > 0)]
        if not hr_valid.empty:
            avg_hook_rate = float(hr_valid["hook_rate"].mean())
    
    # Consolida os dados
    consolidated = {
        "run_at": run_at,
        "metadata": {
            "total_creatives": len(catalog),
            "total_spend": money_br(total_spend),
            "total_spend_raw": total_spend,
            "avg_cpa": money_br(avg_cpa),
            "avg_cpa_raw": avg_cpa,
            "avg_cpm": money_br(avg_cpm),
            "avg_cpm_raw": avg_cpm,
            "avg_cpc": money_br(avg_cpc),
            "avg_cpc_raw": avg_cpc,
            "avg_hook_rate": pct(avg_hook_rate),
            "avg_hook_rate_raw": avg_hook_rate,
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
        object_type = safe_str(row.get("object_type", ""))
        local_path = safe_str(row.get("local_path", ""))
        frames_dir = safe_str(row.get("frames_dir", ""))
        sha256 = safe_str(row.get("sha256", ""))
        catalog_status = safe_str(row.get("effective_status", ""))
        campaign_id = safe_str(row.get("campaign_id", ""))
        
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
        status = determine_status(score_row, catalog_status)
        
        # Extrai KPIs completos
        kpis = {}
        if score_row is not None:
            spend = safe_float(score_row.get("spend_std", 0))
            cac = safe_float(score_row.get("cac", 0))
            performance_score = safe_float(score_row.get("performance_score", 0))
            cpm = safe_float(score_row.get("cpm_calc", 0))
            cpc = safe_float(score_row.get("cpc_calc", 0))
            hook_rate = safe_float(score_row.get("hook_rate", 0))
            ctr = safe_float(score_row.get("ctr", 0))
            connect_rate = safe_float(score_row.get("connect_rate", 0))
            bounce_rate = safe_float(score_row.get("bounce_rate", 0))
            impressions = int(safe_float(score_row.get("impressions_std", 0)))
            clicks = int(safe_float(score_row.get("clicks_std", 0)))
            purchases = int(safe_float(score_row.get("inc_purchase", 0)))
            checkouts = int(safe_float(score_row.get("inc_initiate_checkout", 0)))
            
            # Video metrics
            video_p25 = safe_float(score_row.get("video_p25", 0))
            video_p50 = safe_float(score_row.get("video_p50", 0))
            video_p75 = safe_float(score_row.get("video_p75", 0))
            video_p100 = safe_float(score_row.get("video_p100", 0))
            retention_25_50 = safe_float(score_row.get("retention_25_to_50", 0))
            retention_50_75 = safe_float(score_row.get("retention_50_to_75", 0))
            retention_75_100 = safe_float(score_row.get("retention_75_to_100", 0))
            
            # Placement
            top_placement = safe_str(score_row.get("top_placement", ""))
            top_position = safe_str(score_row.get("top_position", ""))
            placement_count = int(safe_float(score_row.get("placement_count", 0)))
            
            # Campaign info
            campaign_name = safe_str(score_row.get("campaign_name_std", ""))
            adset_name = safe_str(score_row.get("adset_name_std", ""))
            
            kpis = {
                # Score
                "performance_score": round(performance_score, 2) if performance_score > 0 else None,
                
                # Custos
                "spend": spend,
                "spend_br": money_br(spend),
                "share_spend": pct(spend / total_spend) if total_spend > 0 else "0.00%",
                
                # CAC
                "cac": cac,
                "cac_br": money_br(cac) if cac > 0 else "N/A",
                "cac_vs_avg": f"{cac / avg_cpa:.2f}x" if avg_cpa > 0 and cac > 0 else "N/A",
                
                # CPM
                "cpm": cpm,
                "cpm_br": money_br(cpm) if cpm > 0 else "N/A",
                "cpm_vs_avg": f"{cpm / avg_cpm:.2f}x" if avg_cpm > 0 and cpm > 0 else "N/A",
                
                # CPC
                "cpc": cpc,
                "cpc_br": money_br(cpc) if cpc > 0 else "N/A",
                "cpc_vs_avg": f"{cpc / avg_cpc:.2f}x" if avg_cpc > 0 and cpc > 0 else "N/A",
                
                # Taxas
                "ctr": ctr,
                "ctr_pct": pct(ctr),
                "hook_rate": hook_rate,
                "hook_rate_pct": pct(hook_rate),
                "hook_rate_vs_avg": f"{hook_rate / avg_hook_rate:.2f}x" if avg_hook_rate > 0 and hook_rate > 0 else "N/A",
                "connect_rate": connect_rate,
                "connect_rate_pct": pct(connect_rate),
                "bounce_rate": bounce_rate,
                "bounce_rate_pct": pct(bounce_rate),
                
                # Volume
                "impressions": impressions,
                "clicks": clicks,
                "purchases": purchases,
                "checkouts": checkouts,
                
                # Video metrics (se aplicável)
                "video_p25": int(video_p25),
                "video_p50": int(video_p50),
                "video_p75": int(video_p75),
                "video_p100": int(video_p100),
                "retention_25_to_50": pct(retention_25_50),
                "retention_50_to_75": pct(retention_50_75),
                "retention_75_to_100": pct(retention_75_100),
                
                # Placement
                "top_placement": top_placement,
                "top_position": top_position,
                "placement_count": placement_count,
                
                # Context
                "campaign_name": campaign_name,
                "adset_name": adset_name,
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
            "campaign_id": campaign_id,
            "media_type": media_type,
            "object_type": object_type,
            "local_path": local_path,
            "frames_dir": frames_dir,
            "sha256": sha256,
            "status": status,
            "kpis": kpis,
            "name_fields": name_fields,
            "already_processed_by_ai": already_processed,
            "needs_ai_analysis": True,  # Agora TODOS precisam de análise
        }
        
        consolidated["creatives"].append(creative_data)
    
    # Estatísticas
    consolidated["metadata"]["matched_with_score"] = matched
    consolidated["metadata"]["unmatched"] = unmatched
    consolidated["metadata"]["total_videos"] = sum(1 for c in consolidated["creatives"] if c["media_type"] == "video")
    consolidated["metadata"]["total_images"] = sum(1 for c in consolidated["creatives"] if c["media_type"] == "image")
    consolidated["metadata"]["total_active"] = sum(1 for c in consolidated["creatives"] if c["status"] == "ACTIVE")
    consolidated["metadata"]["total_paused"] = sum(1 for c in consolidated["creatives"] if "PAUSED" in c["status"])
    consolidated["metadata"]["total_top_performers"] = sum(1 for c in consolidated["creatives"] if c["status"] == "TOP_PERFORMER")
    
    # Salva JSON consolidado
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Dados consolidados salvos em: {OUTPUT_JSON}")
    logger.info(f"Total: {len(consolidated['creatives'])} criativos")
    logger.info(f"Match com score: {matched} | Sem match: {unmatched}")
    logger.info(f"Vídeos: {consolidated['metadata']['total_videos']} | Imagens: {consolidated['metadata']['total_images']}")
    logger.info(f"Ativos: {consolidated['metadata']['total_active']} | Pausados: {consolidated['metadata']['total_paused']}")


if __name__ == "__main__":
    consolidate()
