#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Score.py - V2 (Reestruturado)

Calcula o score de performance para TODOS os criativos de TODAS as campanhas
de vendas da conta, incluindo ativos e pausados.

Novidades:
- Busca todas as campanhas com objetivo de CONVERSIONS/OUTCOME_SALES
- Inclui criativos pausados no cálculo
- Hook Rate incluído na fórmula do score
- Métricas adicionais: CPM, CPC, Formato, Placement breakdown
- Identificação do tipo de mídia (vídeo vs estático)
"""

import math
import logging
import os
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("performance_score.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")
AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID", "")
API_VERSION = os.getenv("META_GRAPH_VERSION", "v24.0")

# Datas
END_DATE_DT = datetime.now(timezone.utc).date()
START_DATE_DT = END_DATE_DT - timedelta(days=21)

DATE_START = START_DATE_DT.strftime("%Y-%m-%d")
DATE_END = END_DATE_DT.strftime("%Y-%m-%d")

# Janelas de atribuição
WINDOW_STD = None  # padrão da conta
WINDOW_INC = ['1d_click']  # incremental

# Regras do negócio
HARD_SPEND_LIMIT = 1500.0
HARD_CPP_LIMIT = 1500.0
SCORE_CUTOFF = 3.0
MIN_ACTIVE_AFTER = 30
MIN_IMPRESSIONS_SCORE = 10000
MIN_AGE_DAYS_SCORE = 6

# Objetivos de campanha de vendas
SALES_OBJECTIVES = [
    "OUTCOME_SALES",
    "CONVERSIONS",
    "PRODUCT_CATALOG_SALES",
]

run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# ==============================================================================
# HELPERS
# ==============================================================================
def safe_div(n: float, d: float) -> float:
    """Divisão segura que retorna 0 se denominador for 0."""
    return n / d if d and not math.isclose(float(d), 0.0) else 0.0


def clamp(x: float, lo: float, hi: float) -> float:
    """Limita valor entre lo e hi."""
    return max(lo, min(hi, x))


def actions_to_dict(actions_list: list) -> Dict[str, float]:
    """Converte lista de actions em dict somando valores por action_type."""
    out = {}
    if not isinstance(actions_list, list):
        return out
    for a in actions_list:
        try:
            k = a.get("action_type")
            v = float(a.get("value", 0) or 0)
            if k:
                out[k] = out.get(k, 0.0) + v
        except Exception:
            continue
    return out


def get_action_exact(actions_list: list, action_type: str) -> float:
    """Obtém valor exato de uma action específica."""
    d = actions_to_dict(actions_list)
    return float(d.get(action_type, 0.0))


def minmax_norm_pos(x: float, xmin: float, xmax: float) -> float:
    """Normalização min-max para métricas positivas (maior é melhor)."""
    denom = (xmax - xmin)
    return safe_div((x - xmin), denom) if denom and not math.isclose(denom, 0.0) else 0.0


def minmax_norm_neg(x: float, xmin: float, xmax: float) -> float:
    """Normalização min-max para métricas negativas (menor é melhor)."""
    denom = (xmax - xmin)
    return safe_div((xmax - x), denom) if denom and not math.isclose(denom, 0.0) else 0.0


def parse_meta_time(x: str) -> Optional[datetime]:
    """Parseia timestamp do Meta."""
    try:
        if isinstance(x, str) and x.endswith("+0000"):
            x = x[:-5] + "+00:00"
        return datetime.fromisoformat(x)
    except Exception:
        return None


def graph_request(endpoint: str, params: dict, timeout: int = 60) -> dict:
    """Faz requisição à Graph API."""
    url = f"https://graph.facebook.com/{API_VERSION}/{endpoint}"
    params["access_token"] = ACCESS_TOKEN
    
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            
            if r.status_code == 429 or "User request limit reached" in r.text:
                wait = min(300, 60 * (attempt + 1))
                logger.warning(f"Rate limit. Esperando {wait}s...")
                import time
                time.sleep(wait)
                continue
            
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Erro na requisição (tentativa {attempt + 1}): {e}")
            import time
            time.sleep(5)
    
    return {}


# ==============================================================================
# 1) BUSCAR TODAS AS CAMPANHAS DE VENDAS
# ==============================================================================
def fetch_sales_campaigns() -> List[Dict]:
    """Busca todas as campanhas de vendas da conta."""
    logger.info("Buscando campanhas de vendas...")
    
    campaigns = []
    endpoint = f"{AD_ACCOUNT_ID}/campaigns"
    params = {
        "fields": "id,name,objective,status,effective_status",
        "limit": 500,
    }
    
    data = graph_request(endpoint, params)
    
    for campaign in data.get("data", []):
        objective = campaign.get("objective", "")
        if objective in SALES_OBJECTIVES:
            campaigns.append({
                "campaign_id": campaign.get("id"),
                "campaign_name": campaign.get("name"),
                "objective": objective,
                "status": campaign.get("status"),
                "effective_status": campaign.get("effective_status"),
            })
    
    logger.info(f"Encontradas {len(campaigns)} campanhas de vendas")
    return campaigns


# ==============================================================================
# 2) BUSCAR INSIGHTS DOS ADS (TODAS AS CAMPANHAS)
# ==============================================================================
def fetch_insights_all_campaigns(
    campaign_ids: List[str],
    suffix_label: str,
    attribution_windows: list = None
) -> pd.DataFrame:
    """Busca insights de todas as campanhas de vendas."""
    
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    
    FacebookAdsApi.init(access_token=ACCESS_TOKEN)
    account = AdAccount(AD_ACCOUNT_ID)
    
    fields = [
        "campaign_id", "campaign_name",
        "adset_id", "adset_name",
        "ad_id", "ad_name",
        "impressions", "clicks", "spend",
        "inline_link_clicks",
        "actions",
        "video_p25_watched_actions",
        "video_p50_watched_actions",
        "video_p75_watched_actions",
        "video_p100_watched_actions",
        "video_play_actions",
        "cpm", "cpc", "cpp",
    ]
    
    all_rows = []
    
    for campaign_id in campaign_ids:
        params = {
            "level": "ad",
            "time_range": {"since": DATE_START, "until": DATE_END},
            "filtering": [{"field": "campaign.id", "operator": "EQUAL", "value": campaign_id}],
            "limit": 5000,
        }
        
        if attribution_windows:
            params['action_attribution_windows'] = attribution_windows
        
        try:
            insights = account.get_insights(fields=fields, params=params)
            for r in insights:
                all_rows.append(dict(r))
        except Exception as e:
            logger.error(f"Erro ao buscar insights da campanha {campaign_id}: {e}")
    
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    
    # Converter colunas numéricas
    numeric_cols = ["impressions", "clicks", "spend", "inline_link_clicks", "cpm", "cpc", "cpp"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    
    # Renomear colunas com sufixo
    base_cols = [
        "ad_id", "ad_name", "adset_name", "campaign_id", "campaign_name",
        "impressions", "clicks", "spend", "inline_link_clicks", "actions",
        "video_p25_watched_actions", "video_p50_watched_actions",
        "video_p75_watched_actions", "video_p100_watched_actions",
        "video_play_actions", "cpm", "cpc", "cpp",
    ]
    available_cols = [c for c in base_cols if c in df.columns]
    df = df[available_cols].copy()
    
    new_col_names = []
    for c in df.columns:
        if c in ["ad_id", "campaign_id"]:
            new_col_names.append(c)
        else:
            new_col_names.append(f"{c}_{suffix_label}")
    df.columns = new_col_names
    
    return df


# ==============================================================================
# 3) BUSCAR PLACEMENT BREAKDOWN
# ==============================================================================
def fetch_placement_breakdown(campaign_ids: List[str]) -> pd.DataFrame:
    """Busca breakdown por placement para cada ad."""
    
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    
    FacebookAdsApi.init(access_token=ACCESS_TOKEN)
    account = AdAccount(AD_ACCOUNT_ID)
    
    all_rows = []
    
    for campaign_id in campaign_ids:
        params = {
            "level": "ad",
            "time_range": {"since": DATE_START, "until": DATE_END},
            "filtering": [{"field": "campaign.id", "operator": "EQUAL", "value": campaign_id}],
            "breakdowns": ["publisher_platform", "platform_position"],
            "limit": 5000,
        }
        
        try:
            insights = account.get_insights(
                fields=["ad_id", "impressions", "spend", "publisher_platform", "platform_position"],
                params=params
            )
            for r in insights:
                all_rows.append(dict(r))
        except Exception as e:
            logger.error(f"Erro ao buscar placement da campanha {campaign_id}: {e}")
    
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0)
    df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0)
    
    # Agregar por ad_id
    placement_agg = df.groupby("ad_id").apply(
        lambda x: {
            "placement_breakdown": x[["publisher_platform", "platform_position", "impressions", "spend"]].to_dict("records"),
            "top_placement": x.loc[x["impressions"].idxmax(), "publisher_platform"] if not x.empty else "",
            "top_position": x.loc[x["impressions"].idxmax(), "platform_position"] if not x.empty else "",
            "placement_count": len(x),
        }
    ).reset_index(name="placement_data")
    
    # Expandir
    placement_agg["placement_breakdown"] = placement_agg["placement_data"].apply(lambda x: x.get("placement_breakdown", []))
    placement_agg["top_placement"] = placement_agg["placement_data"].apply(lambda x: x.get("top_placement", ""))
    placement_agg["top_position"] = placement_agg["placement_data"].apply(lambda x: x.get("top_position", ""))
    placement_agg["placement_count"] = placement_agg["placement_data"].apply(lambda x: x.get("placement_count", 0))
    
    return placement_agg[["ad_id", "placement_breakdown", "top_placement", "top_position", "placement_count"]]


# ==============================================================================
# 4) METADADOS DOS ADS (STATUS, CREATIVE, TIPO DE MÍDIA)
# ==============================================================================
def fetch_ad_metadata(ad_ids: List[str]) -> pd.DataFrame:
    """Busca metadados dos ads incluindo tipo de mídia."""
    
    meta_rows = []
    url = f"https://graph.facebook.com/{API_VERSION}"
    
    # Processa em lotes de 50
    for i in range(0, len(ad_ids), 50):
        chunk = ad_ids[i:i+50]
        batch = [
            {
                "method": "GET",
                "relative_url": f"{ad_id}?fields=id,effective_status,created_time,creative{{id,object_type,video_id,image_url,thumbnail_url}}"
            }
            for ad_id in chunk
        ]
        
        payload = {
            "access_token": ACCESS_TOKEN,
            "batch": json.dumps(batch)
        }
        
        try:
            r = requests.post(url, data=payload, timeout=60)
            r.raise_for_status()
            results = r.json()
            
            for res in results:
                if res.get("code") == 200:
                    data = json.loads(res.get("body", "{}"))
                    creative = data.get("creative", {})
                    object_type = creative.get("object_type", "")
                    
                    # Determinar tipo de mídia
                    if creative.get("video_id") or object_type == "VIDEO":
                        media_type = "video"
                    elif object_type in ["SHARE", "PHOTO"]:
                        media_type = "image"
                    else:
                        media_type = "unknown"
                    
                    meta_rows.append({
                        "ad_id": str(data.get("id")),
                        "effective_status": data.get("effective_status"),
                        "created_time": data.get("created_time"),
                        "creative_id": creative.get("id"),
                        "object_type": object_type,
                        "media_type": media_type,
                        "video_id": creative.get("video_id"),
                        "image_url": creative.get("image_url"),
                        "thumbnail_url": creative.get("thumbnail_url"),
                    })
        except Exception as e:
            logger.error(f"Erro no batch metadata: {e}")
    
    dfm = pd.DataFrame(meta_rows)
    if dfm.empty:
        return dfm
    
    dfm["created_dt"] = dfm["created_time"].apply(parse_meta_time)
    return dfm


# ==============================================================================
# 5) CALCULAR KPIs (COM HOOK RATE)
# ==============================================================================
def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos os KPIs incluindo Hook Rate."""
    if df.empty:
        return df
    
    # CTR
    df["ctr"] = df.apply(
        lambda r: safe_div(r.get("clicks_std", 0.0), r.get("impressions_std", 0.0)),
        axis=1
    )
    
    # CPM e CPC (já vêm da API, mas recalculamos para garantir)
    df["cpm_calc"] = df.apply(
        lambda r: safe_div(r.get("spend_std", 0.0) * 1000, r.get("impressions_std", 0.0)),
        axis=1
    )
    df["cpc_calc"] = df.apply(
        lambda r: safe_div(r.get("spend_std", 0.0), r.get("clicks_std", 0.0)),
        axis=1
    )
    
    # Link clicks
    def link_clicks_std(row):
        v = float(row.get("inline_link_clicks_std", 0.0) or 0.0)
        if v > 0:
            return v
        return get_action_exact(row.get("actions_std", []), "link_click")
    
    df["link_clicks_std"] = df.apply(link_clicks_std, axis=1)
    
    # Hook Rate (video_play / impressions) - só para vídeos
    def calc_hook_rate(row):
        if row.get("media_type") != "video":
            return 0.0
        video_plays = 0.0
        video_play_actions = row.get("video_play_actions_std", [])
        if isinstance(video_play_actions, list):
            for a in video_play_actions:
                if a.get("action_type") == "video_view":
                    video_plays = float(a.get("value", 0) or 0)
                    break
        impressions = float(row.get("impressions_std", 0) or 0)
        return safe_div(video_plays, impressions)
    
    df["hook_rate"] = df.apply(calc_hook_rate, axis=1)
    
    # Video retention rates (25%, 50%, 75%, 100%)
    def get_video_retention(row, pct):
        col_name = f"video_p{pct}_watched_actions_std"
        actions = row.get(col_name, [])
        if isinstance(actions, list):
            for a in actions:
                if a.get("action_type") == f"video_p{pct}_watched":
                    return float(a.get("value", 0) or 0)
        return 0.0
    
    df["video_p25"] = df.apply(lambda r: get_video_retention(r, 25), axis=1)
    df["video_p50"] = df.apply(lambda r: get_video_retention(r, 50), axis=1)
    df["video_p75"] = df.apply(lambda r: get_video_retention(r, 75), axis=1)
    df["video_p100"] = df.apply(lambda r: get_video_retention(r, 100), axis=1)
    
    # Retention rates (% de quem viu 25% que viu 50%, etc.)
    df["retention_25_to_50"] = df.apply(lambda r: safe_div(r["video_p50"], r["video_p25"]), axis=1)
    df["retention_50_to_75"] = df.apply(lambda r: safe_div(r["video_p75"], r["video_p50"]), axis=1)
    df["retention_75_to_100"] = df.apply(lambda r: safe_div(r["video_p100"], r["video_p75"]), axis=1)
    
    # Incremental metrics (1d click)
    df["lpv_inc"] = df.apply(
        lambda r: get_action_exact(r.get("actions_inc", []), "landing_page_view"),
        axis=1
    )
    df["custom_event_inc"] = df.apply(
        lambda r: get_action_exact(r.get("actions_inc", []), "offsite_conversion.fb_pixel_custom"),
        axis=1
    )
    df["inc_initiate_checkout"] = df.apply(
        lambda r: get_action_exact(r.get("actions_inc", []), "initiate_checkout"),
        axis=1
    )
    df["inc_purchase"] = df.apply(
        lambda r: get_action_exact(r.get("actions_inc", []), "purchase"),
        axis=1
    )
    
    # Purchase padrão (std)
    df["purchase_std"] = df.apply(
        lambda r: get_action_exact(r.get("actions_std", []), "purchase"),
        axis=1
    )
    
    # KPIs de funil
    df["connect_rate"] = df.apply(
        lambda r: safe_div(r["lpv_inc"], r["link_clicks_std"]),
        axis=1
    )
    df["bounce_rate"] = df.apply(
        lambda r: safe_div(r["custom_event_inc"], r["lpv_inc"]),
        axis=1
    )
    
    # Custos
    df["spend_std"] = pd.to_numeric(df.get("spend_std", 0.0), errors="coerce").fillna(0.0)
    
    # CAC
    df["cac"] = df.apply(
        lambda r: safe_div(r["spend_std"], r["inc_purchase"]) if r["inc_purchase"] > 0 else 0.0,
        axis=1
    )
    
    # Custo por checkout
    df["cost_per_checkout"] = df.apply(
        lambda r: safe_div(r["spend_std"], r["inc_initiate_checkout"]) if r["inc_initiate_checkout"] > 0 else 0.0,
        axis=1
    )
    
    # CPP padrão (std)
    df["cpp_std"] = df.apply(
        lambda r: safe_div(r["spend_std"], r["purchase_std"]) if r["purchase_std"] > 0 else float("inf"),
        axis=1
    )
    
    df["has_purchase_inc"] = df["inc_purchase"] > 0
    df["has_checkout_inc"] = df["inc_initiate_checkout"] > 0
    
    return df


# ==============================================================================
# 6) CALCULAR SCORE (COM HOOK RATE)
# ==============================================================================
def compute_score_with_hook_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula score 0-10 incluindo Hook Rate."""
    if df.empty:
        return df
    
    # Base para correlação
    base = df[(df["has_purchase_inc"]) & (df["cac"] > 0)].copy()
    
    # KPIs incluindo hook_rate
    kpis = ["ctr", "connect_rate", "bounce_rate", "cost_per_checkout", "cac", "hook_rate"]
    
    if len(base) < 5:
        logger.warning("Base insuficiente para correlação. Usando fallback de pesos.")
        weights = {
            "ctr": 1.2,
            "connect_rate": 1.2,
            "bounce_rate": 1.5,
            "cost_per_checkout": 1.5,
            "cac": 2.5,
            "hook_rate": 2.1,  # Hook rate tem peso significativo
        }
    else:
        corr = {}
        for k in kpis:
            try:
                c = base[k].corr(base["cac"])
                corr[k] = 0.0 if pd.isna(c) else float(abs(c))
            except Exception:
                corr[k] = 0.0
        
        # Garante CAC como maior peso
        max_other = max([v for kk, v in corr.items() if kk != "cac"] + [0.0])
        corr["cac"] = max(corr.get("cac", 0.0), max_other + 1e-6)
        
        # Normaliza para somar 10
        s = sum(corr.values()) if sum(corr.values()) > 0 else 1.0
        weights = {k: (10.0 * corr[k] / s) for k in kpis}
        
        logger.info("Pesos por correlação (normalizados p/ 10):")
        for k, w in weights.items():
            logger.info(f"   - {k}: {w:.4f}")
    
    # Min/Max para normalização
    ctr_min, ctr_max = df["ctr"].min(), df["ctr"].max()
    cr_min, cr_max = df["connect_rate"].min(), df["connect_rate"].max()
    br_min, br_max = df["bounce_rate"].min(), df["bounce_rate"].max()
    hr_min, hr_max = df["hook_rate"].min(), df["hook_rate"].max()
    
    cpc_vals = df[df["has_checkout_inc"]]["cost_per_checkout"]
    cpc_min = cpc_vals.min() if not cpc_vals.empty else 0.0
    cpc_max = cpc_vals.max() if not cpc_vals.empty else 1.0
    
    cac_vals = df[df["has_purchase_inc"]]["cac"]
    cac_min = cac_vals.min() if not cac_vals.empty else 0.0
    cac_max = cac_vals.max() if not cac_vals.empty else 1.0
    
    # Componentes normalizados 0..1
    df["n_ctr"] = df["ctr"].apply(lambda x: minmax_norm_pos(x, ctr_min, ctr_max))
    df["n_connect_rate"] = df["connect_rate"].apply(lambda x: minmax_norm_pos(x, cr_min, cr_max))
    df["n_bounce_rate"] = df["bounce_rate"].apply(lambda x: minmax_norm_pos(x, br_min, br_max))
    df["n_hook_rate"] = df["hook_rate"].apply(lambda x: minmax_norm_pos(x, hr_min, hr_max))
    
    def n_cpc(row):
        if not row["has_checkout_inc"]:
            return 0.0
        return minmax_norm_neg(row["cost_per_checkout"], cpc_min, cpc_max)
    
    def n_cac(row):
        if not row["has_purchase_inc"]:
            return 0.0
        return minmax_norm_neg(row["cac"], cac_min, cac_max)
    
    df["n_cost_per_checkout"] = df.apply(n_cpc, axis=1)
    df["n_cac"] = df.apply(n_cac, axis=1)
    
    # Score 0..10
    df["performance_score"] = (
        weights["ctr"] * df["n_ctr"] +
        weights["connect_rate"] * df["n_connect_rate"] +
        weights["bounce_rate"] * df["n_bounce_rate"] +
        weights["cost_per_checkout"] * df["n_cost_per_checkout"] +
        weights["cac"] * df["n_cac"] +
        weights["hook_rate"] * df["n_hook_rate"]
    ).apply(lambda x: round(clamp(float(x), 0.0, 10.0), 2))
    
    # Guardar pesos
    for k, w in weights.items():
        df[f"w_{k}"] = w
    
    return df


# ==============================================================================
# 7) PAUSAR ADS
# ==============================================================================
def pause_ads(ad_ids_to_pause: List[str], reason: str) -> None:
    """Pausa uma lista de ads."""
    if not ad_ids_to_pause:
        return
    
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.ad import Ad
    FacebookAdsApi.init(access_token=ACCESS_TOKEN)
    
    for ad_id in ad_ids_to_pause:
        try:
            ad = Ad(str(ad_id))
            ad.api_update(params={"status": "PAUSED"})
            logger.info(f"✅ Pausado (reason={reason}): ad_id={ad_id}")
        except Exception as e:
            logger.error(f"❌ Erro ao pausar ad_id={ad_id}: {e}")


# ==============================================================================
# 8) MAIN
# ==============================================================================
def main():
    if not ACCESS_TOKEN or not AD_ACCOUNT_ID:
        raise RuntimeError("Defina META_ACCESS_TOKEN e META_AD_ACCOUNT_ID")
    
    logger.info("=" * 60)
    logger.info("Iniciando Performance Score V2 - Todas as Campanhas de Vendas")
    logger.info("=" * 60)
    
    # 1) Buscar campanhas de vendas
    campaigns = fetch_sales_campaigns()
    if not campaigns:
        logger.warning("Nenhuma campanha de vendas encontrada")
        return
    
    campaign_ids = [c["campaign_id"] for c in campaigns]
    logger.info(f"Campanhas encontradas: {len(campaign_ids)}")
    for c in campaigns:
        logger.info(f"  - {c['campaign_name']} ({c['objective']})")
    
    # 2) Buscar insights STD
    logger.info("Buscando dados STD (padrão da conta)...")
    df_std = fetch_insights_all_campaigns(campaign_ids, "std", attribution_windows=WINDOW_STD)
    
    # 3) Buscar insights INC
    logger.info("Buscando dados INC (1d_click)...")
    df_inc = fetch_insights_all_campaigns(campaign_ids, "inc", attribution_windows=WINDOW_INC)
    
    if df_std.empty:
        logger.warning("Nenhum dado encontrado")
        return
    
    # 4) Merge
    df = df_std.merge(df_inc, on=["ad_id", "campaign_id"], how="left")
    
    # 5) Metadados
    logger.info("Buscando metadados (status + tipo de mídia)...")
    ad_ids = df["ad_id"].astype(str).unique().tolist()
    meta = fetch_ad_metadata(ad_ids)
    
    if not meta.empty:
        df = df.merge(
            meta[["ad_id", "effective_status", "created_dt", "media_type", "creative_id", "object_type"]],
            on="ad_id",
            how="left"
        )
    
    # 6) Placement breakdown
    logger.info("Buscando breakdown de placement...")
    placement = fetch_placement_breakdown(campaign_ids)
    if not placement.empty:
        df = df.merge(placement, on="ad_id", how="left")
    
    # 7) Calcular KPIs
    logger.info("Calculando KPIs...")
    df["pause_reason"] = ""
    df["is_top5"] = False
    df = compute_kpis(df)
    
    # 8) Calcular Score (para TODOS, ativos e pausados)
    logger.info("Calculando scores...")
    df = compute_score_with_hook_rate(df)
    
    # 9) Estatísticas
    active_mask = (df["effective_status"] == "ACTIVE")
    paused_mask = (df["effective_status"] == "PAUSED")
    
    active_count = int(active_mask.sum())
    paused_count = int(paused_mask.sum())
    total_count = len(df)
    
    logger.info(f"Total de criativos: {total_count}")
    logger.info(f"  - Ativos: {active_count}")
    logger.info(f"  - Pausados: {paused_count}")
    
    # Estatísticas por tipo de mídia
    video_count = int((df["media_type"] == "video").sum())
    image_count = int((df["media_type"] == "image").sum())
    logger.info(f"Por tipo de mídia:")
    logger.info(f"  - Vídeos: {video_count}")
    logger.info(f"  - Estáticos: {image_count}")
    
    # 10) Aplicar regras de pausa (apenas para ativos)
    logger.info("Aplicando regras de HARD STOP...")
    
    hard_candidates = df[
        (df["effective_status"] == "ACTIVE") &
        (df["spend_std"] >= HARD_SPEND_LIMIT) &
        ((df["purchase_std"] <= 0) | (df["cpp_std"] > HARD_CPP_LIMIT))
    ].copy()
    
    to_pause_hard = []
    remaining_active = active_count
    
    for _, row in hard_candidates.iterrows():
        if remaining_active <= MIN_ACTIVE_AFTER:
            break
        to_pause_hard.append(str(row["ad_id"]))
        remaining_active -= 1
    
    logger.info(f"  - Candidatos HARD: {len(hard_candidates)} | Vai pausar: {len(to_pause_hard)}")
    
    if to_pause_hard:
        df.loc[df["ad_id"].isin(to_pause_hard), "pause_reason"] = "HARD_STOP"
        pause_ads(to_pause_hard, "HARD_STOP")
    
    # 11) Aplicar regras de SCORE
    logger.info("Aplicando regras de SCORE...")
    
    now_dt = datetime.now(timezone.utc)
    
    score_candidates = df[
        (df["effective_status"] == "ACTIVE") &
        (~df["ad_id"].isin(to_pause_hard)) &
        (df["impressions_std"] >= MIN_IMPRESSIONS_SCORE) &
        (df["created_dt"].apply(lambda x: (now_dt - x).days if x else 0) >= MIN_AGE_DAYS_SCORE) &
        (df["performance_score"] < SCORE_CUTOFF)
    ].copy()
    
    to_pause_score = []
    
    for _, row in score_candidates.sort_values("performance_score").iterrows():
        if remaining_active <= MIN_ACTIVE_AFTER:
            break
        to_pause_score.append(str(row["ad_id"]))
        remaining_active -= 1
    
    logger.info(f"  - Elegíveis para SCORE: {len(score_candidates)}")
    logger.info(f"  - Abaixo do cutoff ({SCORE_CUTOFF}): {len(score_candidates)} | Vai pausar: {len(to_pause_score)}")
    
    if to_pause_score:
        df.loc[df["ad_id"].isin(to_pause_score), "pause_reason"] = "LOW_SCORE"
        pause_ads(to_pause_score, "LOW_SCORE")
    
    # 12) Marcar TOP 5
    top5_ids = df[df["effective_status"] == "ACTIVE"].nlargest(5, "performance_score")["ad_id"].tolist()
    df.loc[df["ad_id"].isin(top5_ids), "is_top5"] = True
    
    # 13) Salvar resultados
    Path("history").mkdir(exist_ok=True)
    
    # Colunas para exportar
    export_cols = [
        "ad_id", "ad_name_std", "campaign_id", "campaign_name_std", "adset_name_std",
        "effective_status", "media_type", "object_type", "creative_id",
        "impressions_std", "clicks_std", "spend_std", "link_clicks_std",
        "ctr", "cpm_calc", "cpc_calc", "hook_rate",
        "video_p25", "video_p50", "video_p75", "video_p100",
        "retention_25_to_50", "retention_50_to_75", "retention_75_to_100",
        "lpv_inc", "custom_event_inc", "inc_initiate_checkout", "inc_purchase",
        "purchase_std", "connect_rate", "bounce_rate",
        "cac", "cost_per_checkout", "cpp_std",
        "performance_score", "pause_reason", "is_top5",
        "top_placement", "top_position", "placement_count",
        "w_ctr", "w_connect_rate", "w_bounce_rate", "w_cost_per_checkout", "w_cac", "w_hook_rate",
        "created_dt",
    ]
    
    available_export_cols = [c for c in export_cols if c in df.columns]
    df_export = df[available_export_cols].copy()
    
    # Salvar histórico
    history_path = f"history/scores_{run_at}.csv"
    df_export.to_csv(history_path, index=False)
    logger.info(f"Histórico salvo em: {history_path}")
    
    # Salvar latest
    latest_path = "creative_score_automation.csv"
    df_export.to_csv(latest_path, index=False)
    logger.info(f"Latest salvo em: {latest_path}")
    
    # 14) Gerar resumo
    summary_lines = [
        "=" * 60,
        "Performance Score V2 - Resumo",
        "=" * 60,
        f"Data: {run_at}",
        f"Período: {DATE_START} a {DATE_END}",
        "",
        f"Campanhas de vendas: {len(campaigns)}",
        f"Total de criativos: {total_count}",
        f"  - Ativos: {active_count}",
        f"  - Pausados: {paused_count}",
        f"  - Vídeos: {video_count}",
        f"  - Estáticos: {image_count}",
        "",
        f"Pausados nesta execução:",
        f"  - HARD_STOP: {len(to_pause_hard)}",
        f"  - LOW_SCORE: {len(to_pause_score)}",
        "",
        f"Ativos restantes: {remaining_active}",
        "=" * 60,
    ]
    
    with open("summary_email.txt", "w") as f:
        f.write("\n".join(summary_lines))
    
    logger.info("Execução concluída com sucesso!")


if __name__ == "__main__":
    main()
