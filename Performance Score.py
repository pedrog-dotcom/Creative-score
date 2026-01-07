import math
import logging
import os
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
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
# Recomendo fortemente colocar no .env ao invés de hardcode
ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "COLOQUE_O_TOKEN_AQUI")
AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID", "act_679133514401382")
CAMPAIGN_ID = os.getenv("META_CAMPAIGN_ID", "120240998569260670")

# Datas
END_DATE_DT = datetime.now(timezone.utc).date()
START_DATE_DT = END_DATE_DT - timedelta(days=21)

DATE_START = START_DATE_DT.strftime("%Y-%m-%d")
DATE_END   = END_DATE_DT.strftime("%Y-%m-%d")

# Janelas
WINDOW_STD = None           # padrão da conta (geralmente 7d_click + 1d_view)
WINDOW_INC = ['1d_click']   # incremental (1d click puro)

# Regras do negócio
HARD_SPEND_LIMIT = 1500.0
HARD_CPP_LIMIT   = 1500.0

SCORE_CUTOFF = 3.0
MIN_ACTIVE_AFTER = 30

MIN_IMPRESSIONS_SCORE = 10000
MIN_AGE_DAYS_SCORE = 6

run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

# ==============================================================================
# HELPERS
# ==============================================================================
def safe_div(n, d):
    return n / d if d and not math.isclose(float(d), 0.0) else 0.0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def actions_to_dict(actions_list):
    """
    Converte lista de actions em dict somando valores por action_type.
    Como filtramos janela via API, o 'value' já vem filtrado para aquela janela.
    """
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

def get_action_exact(actions_list, action_type: str) -> float:
    d = actions_to_dict(actions_list)
    return float(d.get(action_type, 0.0))

def minmax_norm_pos(x, xmin, xmax):
    denom = (xmax - xmin)
    return safe_div((x - xmin), denom) if denom and not math.isclose(denom, 0.0) else 0.0

def minmax_norm_neg(x, xmin, xmax):
    denom = (xmax - xmin)
    return safe_div((xmax - x), denom) if denom and not math.isclose(denom, 0.0) else 0.0

def parse_meta_time(x: str):
    try:
        if isinstance(x, str) and x.endswith("+0000"):
            x = x[:-5] + "+00:00"
        return datetime.fromisoformat(x)
    except Exception:
        return None

# ==============================================================================
# 1) BUSCAR INSIGHTS (COM SUPORTE A JANELAS DE ATRIBUIÇÃO)
# ==============================================================================
def fetch_insights_by_ad(campaign_id: str, suffix_label: str, attribution_windows: list = None) -> pd.DataFrame:
    if not ACCESS_TOKEN or not AD_ACCOUNT_ID:
        raise RuntimeError("Defina META_ACCESS_TOKEN e META_AD_ACCOUNT_ID (de preferência via .env)")

    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount

    FacebookAdsApi.init(access_token=ACCESS_TOKEN)
    account = AdAccount(AD_ACCOUNT_ID)

    fields = [
        "campaign_id", "campaign_name",
        "adset_id", "adset_name",
        "ad_id", "ad_name",
        "impressions", "clicks",
        "spend",
        "inline_link_clicks",
        "actions",
    ]

    params = {
        "level": "ad",
        "time_range": {"since": DATE_START, "until": DATE_END},
        "filtering": [{"field": "campaign.id", "operator": "EQUAL", "value": campaign_id}],
        "limit": 5000,
    }

    if attribution_windows:
        params['action_attribution_windows'] = attribution_windows

    rows = []
    try:
        insights = account.get_insights(fields=fields, params=params)
    except Exception as e:
        logger.error(f"Erro ao buscar insights ({suffix_label}): {e}")
        return pd.DataFrame()

    for r in insights:
        rows.append(dict(r))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["impressions", "clicks", "spend", "inline_link_clicks"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    base_cols = ["ad_id", "ad_name", "adset_name", "campaign_name", "impressions", "clicks", "spend", "inline_link_clicks", "actions"]
    available_cols = [c for c in base_cols if c in df.columns]
    df = df[available_cols].copy()

    new_col_names = []
    for c in df.columns:
        if c == "ad_id":
            new_col_names.append(c)
        else:
            new_col_names.append(f"{c}_{suffix_label}")
    df.columns = new_col_names
    return df

# ==============================================================================
# 2) METADADOS (STATUS/DATA)
# ==============================================================================
def fetch_ad_metadata(ad_ids: list) -> pd.DataFrame:
    """
    Busca metadados de anúncios usando Batch API para evitar Rate Limit.
    """
    import requests
    import json
    
    API_VERSION = os.getenv("META_GRAPH_VERSION", "v24.0")
    url = f"https://graph.facebook.com/{API_VERSION}"
    
    meta_rows = []
    # Processa em lotes de 50
    for i in range(0, len(ad_ids), 50):
        chunk = ad_ids[i:i+50]
        batch = [
            {"method": "GET", "relative_url": f"{ad_id}?fields=id,effective_status,created_time"}
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
                    meta_rows.append({
                        "ad_id": str(data.get("id")),
                        "effective_status": data.get("effective_status"),
                        "created_time": data.get("created_time"),
                    })
        except Exception as e:
            print(f"Erro no batch metadata: {e}")
            
    dfm = pd.DataFrame(meta_rows)
    if dfm.empty:
        return dfm
    dfm["created_dt"] = dfm["created_time"].apply(parse_meta_time)
    return dfm

# ==============================================================================
# 3) PAUSAR ADS (COM GUARDA DE MÍNIMO ATIVO)
# ==============================================================================
def pause_ads(ad_ids_to_pause: list, reason: str):
    """
    Pausa uma lista de ads. (Loop simples)
    """
    if not ad_ids_to_pause:
        return

    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.ad import Ad
    FacebookAdsApi.init(access_token=ACCESS_TOKEN)

    for ad_id in ad_ids_to_pause:
        try:
            ad = Ad(str(ad_id))
            ad.api_update(params={"status": "PAUSED"})
            print(f"✅ Pausado (reason={reason}): ad_id={ad_id}")
        except Exception as e:
            print(f"❌ Erro ao pausar ad_id={ad_id}: {e}")

# ==============================================================================
# 4) KPIs + SCORE (PESOS VIA CORRELAÇÃO)
# ==============================================================================
def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # CTR (std)
    df["ctr"] = df.apply(lambda r: safe_div(r.get("clicks_std", 0.0), r.get("impressions_std", 0.0)), axis=1)

    # Link clicks (std): inline_link_clicks > actions(link_click)
    def link_clicks_std(row):
        v = float(row.get("inline_link_clicks_std", 0.0) or 0.0)
        if v > 0:
            return v
        return get_action_exact(row.get("actions_std", []), "link_click")

    df["link_clicks_std"] = df.apply(link_clicks_std, axis=1)

    # Incremental (1d click)
    df["lpv_inc"] = df.apply(lambda r: get_action_exact(r.get("actions_inc", []), "landing_page_view"), axis=1)
    df["custom_event_inc"] = df.apply(lambda r: get_action_exact(r.get("actions_inc", []), "offsite_conversion.fb_pixel_custom"), axis=1)
    df["inc_initiate_checkout"] = df.apply(lambda r: get_action_exact(r.get("actions_inc", []), "initiate_checkout"), axis=1)
    df["inc_purchase"] = df.apply(lambda r: get_action_exact(r.get("actions_inc", []), "purchase"), axis=1)

    # Purchase padrão (std) para HARD STOP
    df["purchase_std"] = df.apply(lambda r: get_action_exact(r.get("actions_std", []), "purchase"), axis=1)

    # KPIs de funil
    df["connect_rate"] = df.apply(lambda r: safe_div(r["lpv_inc"], r["link_clicks_std"]), axis=1)
    df["bounce_rate"] = df.apply(lambda r: safe_div(r["custom_event_inc"], r["lpv_inc"]), axis=1)

    # Custos
    df["spend_std"] = pd.to_numeric(df.get("spend_std", 0.0), errors="coerce").fillna(0.0)

    # CAC (custo por compra) para o SCORE (1d click, como você vinha usando)
    df["cac"] = df.apply(lambda r: (safe_div(r["spend_std"], r["inc_purchase"]) if r["inc_purchase"] > 0 else 0.0), axis=1)

    # Custo por checkout (1d click)
    df["cost_per_checkout"] = df.apply(lambda r: (safe_div(r["spend_std"], r["inc_initiate_checkout"]) if r["inc_initiate_checkout"] > 0 else 0.0), axis=1)

    # CPP padrão (std) para HARD STOP
    df["cpp_std"] = df.apply(lambda r: (safe_div(r["spend_std"], r["purchase_std"]) if r["purchase_std"] > 0 else float("inf")), axis=1)

    df["has_purchase_inc"] = df["inc_purchase"] > 0
    df["has_checkout_inc"] = df["inc_initiate_checkout"] > 0

    return df

def compute_score_with_correlation_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score final 0..10 com 2 casas.
    Pesos = |corr(KPI, CAC)|, mas CAC sempre maior peso.
    """
    if df.empty:
        return df

    # Base para correlação: apenas linhas com purchase_inc > 0 e cac > 0
    base = df[(df["has_purchase_inc"]) & (df["cac"] > 0)].copy()

    kpis = ["ctr", "connect_rate", "bounce_rate", "cost_per_checkout", "cac"]
    # Se não tiver base suficiente, fallback (bem conservador)
    if len(base) < 5:
        logger.warning("⚠️ Base insuficiente para correlação (poucas compras em 1d click). Usando fallback de pesos.")
        weights = {
            "ctr": 1.5,
            "connect_rate": 1.5,
            "bounce_rate": 2.0,
            "cost_per_checkout": 2.0,
            "cac": 3.0,
        }
    else:
        corr = {}
        for k in kpis:
            try:
                # Pearson (pandas)
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

        print("📌 Pesos por correlação (normalizados p/ 10):")
        for k, w in weights.items():
            print(f"   - {k}: {w:.4f}")

    # Min/Max para normalização
    ctr_min, ctr_max = df["ctr"].min(), df["ctr"].max()
    cr_min, cr_max = df["connect_rate"].min(), df["connect_rate"].max()
    br_min, br_max = df["bounce_rate"].min(), df["bounce_rate"].max()

    # custo/checkout: considerar apenas quem tem checkout
    cpc_vals = df[df["has_checkout_inc"]]["cost_per_checkout"]
    cpc_min = cpc_vals.min() if not cpc_vals.empty else 0.0
    cpc_max = cpc_vals.max() if not cpc_vals.empty else 1.0

    # cac: considerar apenas quem tem purchase
    cac_vals = df[df["has_purchase_inc"]]["cac"]
    cac_min = cac_vals.min() if not cac_vals.empty else 0.0
    cac_max = cac_vals.max() if not cac_vals.empty else 1.0

    # Componentes 0..1
    df["n_ctr"] = df["ctr"].apply(lambda x: minmax_norm_pos(x, ctr_min, ctr_max))
    df["n_connect_rate"] = df["connect_rate"].apply(lambda x: minmax_norm_pos(x, cr_min, cr_max))
    df["n_bounce_rate"] = df["bounce_rate"].apply(lambda x: minmax_norm_pos(x, br_min, br_max))

    # Para custos: se não tem evento, zera componente (não “premia” ausência de dado)
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
        weights["cac"] * df["n_cac"]
    ).apply(lambda x: round(clamp(float(x), 0.0, 10.0), 2))

    # Guardar pesos usados para auditoria
    df["w_ctr"] = weights["ctr"]
    df["w_connect_rate"] = weights["connect_rate"]
    df["w_bounce_rate"] = weights["bounce_rate"]
    df["w_cost_per_checkout"] = weights["cost_per_checkout"]
    df["w_cac"] = weights["cac"]

    return df

# ==============================================================================
# 5) MAIN
# ==============================================================================
if __name__ == "__main__":
    if not CAMPAIGN_ID:
        raise RuntimeError("Defina META_CAMPAIGN_ID no ambiente")

    print(f"--- Iniciando Automação de Score/Pause para Campanha {CAMPAIGN_ID} ---")

    print("1) Buscando dados STD (padrão da conta)...")
    df_std = fetch_insights_by_ad(CAMPAIGN_ID, "std", attribution_windows=WINDOW_STD)

    print("2) Buscando dados INC (1d_click)...")
    df_inc = fetch_insights_by_ad(CAMPAIGN_ID, "inc", attribution_windows=WINDOW_INC)

    if df_std.empty:
        print("Nenhum dado encontrado para a campanha.")
        raise SystemExit(0)

    # Merge
    df = df_std.merge(df_inc, on="ad_id", how="left")

    # Metadados
    print("3) Buscando metadados (status + created_time)...")
    ad_ids = df["ad_id"].astype(str).unique().tolist()
    meta = fetch_ad_metadata(ad_ids)
    if meta.empty:
        print("❌ Não foi possível obter metadados dos ads. Abortando.")
        raise SystemExit(1)

    df = df.merge(meta[["ad_id", "effective_status", "created_dt"]], on="ad_id", how="left")

    # KPIs
    print("4) Calculando KPIs...")
    # Inicializa colunas para auditoria e IA
    df["pause_reason"] = ""
    df["is_top5"] = False

    df = compute_kpis(df)

    # Contagem de ativos atual
    active_mask = (df["effective_status"] == "ACTIVE")
    active_now = int(active_mask.sum())
    print(f"📊 Ativos agora (no dataframe): {active_now}")

    # --------------------------------------------------------------------------
    # REGRA A: HARD STOP (qualquer criativo, sem filtro de idade/impressão)
    # --------------------------------------------------------------------------
    print("5) Aplicando HARD STOP (spend>=1500 e (purchase_std==0 ou cpp_std>1500))...")

    hard_candidates = df[
        (df["effective_status"] == "ACTIVE") &
        (df["spend_std"] >= HARD_SPEND_LIMIT) &
        (
            (df["purchase_std"] <= 0) |
            (df["cpp_std"] > HARD_CPP_LIMIT)
        )
    ].copy()

    # Ordena para pausar os “piores” primeiro (sem purchase primeiro, depois maior cpp)
    hard_candidates["hard_rank"] = hard_candidates.apply(
        lambda r: (1 if r["purchase_std"] <= 0 else 0, float(r["cpp_std"])),
        axis=1
    )
    hard_candidates = hard_candidates.sort_values(["hard_rank", "spend_std"], ascending=[False, False])

    to_pause_hard = []
    remaining_active = active_now

    for _, row in hard_candidates.iterrows():
        if remaining_active - 1 < MIN_ACTIVE_AFTER:
            break
        to_pause_hard.append(row["ad_id"])
        remaining_active -= 1

    print(f"   - Candidatos HARD: {len(hard_candidates)} | Vai pausar: {len(to_pause_hard)} | Ativos restantes: {remaining_active}")
    pause_ads(to_pause_hard, reason="HARD_STOP")

    # Atualiza status localmente (para não depender de uma nova leitura)
    df.loc[df["ad_id"].isin(to_pause_hard), "effective_status"] = "PAUSED"

    # Marca motivo de pausa (HARD STOP)
    df.loc[df["ad_id"].isin(to_pause_hard), "pause_reason"] = "HARD_STOP"

    # --------------------------------------------------------------------------
    # REGRA B: SCORE + PAUSAR score < 3 (somente >6 dias e >=10k imp)
    # --------------------------------------------------------------------------
    print("6) Calculando SCORE (somente para elegíveis) e pausando score < 3...")

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=MIN_AGE_DAYS_SCORE)

    eligible = df[
        (df["effective_status"] == "ACTIVE") &
        (df["created_dt"].notna()) &
        (df["created_dt"] <= cutoff_dt) &
        (df.get("impressions_std", 0.0) >= MIN_IMPRESSIONS_SCORE)
    ].copy()

    print(f"   - Elegíveis para SCORE: {len(eligible)}")

    if not eligible.empty:
        eligible = compute_score_with_correlation_weights(eligible)

        # Marca TOP 5 criativos (entre elegíveis)
        top5_ids = (
            eligible.sort_values("performance_score", ascending=False)
                    .head(5)["ad_id"]
                    .astype(str)
                    .tolist()
        )
        
        df.loc[df["ad_id"].astype(str).isin(top5_ids), "is_top5"] = True

        # Quem será pausado por score
        score_bad = eligible[eligible["performance_score"] < SCORE_CUTOFF].copy()
        # Ordena do pior para o “menos pior”
        score_bad = score_bad.sort_values("performance_score", ascending=True)

        # Reconta ativos após hard stop
        active_after_hard = int((df["effective_status"] == "ACTIVE").sum())
        remaining_active = active_after_hard

        to_pause_score = []
        for _, row in score_bad.iterrows():
            if remaining_active - 1 < MIN_ACTIVE_AFTER:
                break
            to_pause_score.append(row["ad_id"])
            remaining_active -= 1

        print(f"   - Abaixo do cutoff ({SCORE_CUTOFF}): {len(score_bad)} | Vai pausar: {len(to_pause_score)} | Ativos restantes: {remaining_active}")
        pause_ads(to_pause_score, reason="SCORE_LT_3")

        # Atualiza localmente
        df.loc[df["ad_id"].isin(to_pause_score), "effective_status"] = "PAUSED"

        # Marca motivo de pausa (SCORE)
        df.loc[df["ad_id"].isin(to_pause_score), "pause_reason"] = "SCORE_LT_3"

        # Grava score de volta no df principal
        df = df.merge(
            eligible[["ad_id", "performance_score", "w_ctr", "w_connect_rate", "w_bounce_rate", "w_cost_per_checkout", "w_cac"]],
            on="ad_id",
            how="left"
        )
    else:
        print("   - Nenhum criativo elegível para score nesta execução.")
        df["performance_score"] = None

    # --------------------------------------------------------------------------
    # RESUMO PARA EMAIL (Slack via email)
    # --------------------------------------------------------------------------
    def fmt_money(x):
        try:
            return f"R$ {float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return "R$ 0,00"
    
    def fmt_pct(x):
        try:
            return f"{float(x)*100:.2f}%"
        except Exception:
            return "0,00%"
    
    def fmt_num(x):
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "—"
    
    def fmt_cpa_vs_avg(cpa, avg):
        if avg is None or avg == 0 or (cpa is None) or (pd.isna(cpa)) or cpa == 0:
            return "—"
        return f"{(cpa/avg):.2f}x"
    
    # total spend da campanha na janela (para share)
    total_spend = float(df["spend_std"].sum()) if "spend_std" in df.columns else 0.0
    
    # média de CPA (usando CAC 1d_click, que é o que você usa no score)
    cpa_base = df[(df.get("has_purchase_inc", False)) & (df["cac"] > 0)].copy()
    avg_cpa = float(cpa_base["cac"].mean()) if not cpa_base.empty else None
    
    # helper para montar linhas
    def build_lines(subdf, title, max_rows=999):
        lines = []
        lines.append(title)
        lines.append("-" * len(title))
        if subdf.empty:
            lines.append("Nenhum item.\n")
            return lines
    
        # cabeçalho
        lines.append("Anúncio | Invest | Share | CPA(1d) | vs média | Score")
        lines.append("------ | ------ | ----- | ------- | -------- | -----")
    
        subdf = subdf.head(max_rows).copy()
    
        for _, r in subdf.iterrows():
            name = str(r.get("ad_name_std") or r.get("ad_name_inc") or r.get("ad_name") or r.get("ad_id"))
            spend = float(r.get("spend_std", 0.0) or 0.0)
            share = (spend / total_spend) if total_spend > 0 else 0.0
            cpa = float(r.get("cac", 0.0) or 0.0)
            score = r.get("performance_score", None)
    
            cpa_str = fmt_money(cpa) if cpa and cpa > 0 else "sem purchase (1d)"
            score_str = fmt_num(score) if score is not None and not pd.isna(score) else "—"
    
            lines.append(
                f"{name} | {fmt_money(spend)} | {fmt_pct(share)} | {cpa_str} | {fmt_cpa_vs_avg(cpa, avg_cpa)} | {score_str}"
            )
    
        lines.append("")  # linha em branco
        return lines
    
    # -------------------------
    # Conjuntos: pausados por má performance (score < 3)
    # -------------------------
    paused_score_ids = set(to_pause_score) if 'to_pause_score' in globals() else set()
    paused_score_df = df[df["ad_id"].isin(paused_score_ids)].copy()
    
    # garantir score e cac presentes
    if "performance_score" not in paused_score_df.columns:
        # se você só mergeia score depois, pode puxar do eligible (se existir)
        try:
            paused_score_df = paused_score_df.merge(
                eligible[["ad_id", "performance_score"]],
                on="ad_id", how="left"
            )
        except Exception:
            pass
    
    paused_score_df = paused_score_df.sort_values("performance_score", ascending=True)
    
    # -------------------------
    # Top 5 melhores criativos (entre elegíveis com score calculado)
    # Preferência: usar "eligible" se existir, porque ali score está garantido
    # -------------------------
    top_df = pd.DataFrame()
    try:
        top_df = eligible.copy()
        # se você quer só os que ficaram ativos após pausas:
        # top_df = top_df[top_df["ad_id"].isin(df[df["effective_status"]=="ACTIVE"]["ad_id"])]
        top_df = top_df.sort_values("performance_score", ascending=False).head(5)
    except Exception:
        # fallback: tenta usar df se tiver score
        if "performance_score" in df.columns:
            top_df = df.dropna(subset=["performance_score"]).sort_values("performance_score", ascending=False).head(5)
    
    # -------------------------
    # Montar arquivo do email
    # -------------------------
    run_url = os.getenv("GITHUB_SERVER_URL", "https://github.com") + "/" + os.getenv("GITHUB_REPOSITORY", "") + "/actions/runs/" + os.getenv("GITHUB_RUN_ID", "")
    run_line = f"Run: {run_url}" if os.getenv("GITHUB_RUN_ID") else "Run: (local)"
    
    lines = []
    lines.append("Meta Score — Resumo da Execução")
    lines.append("=" * 28)
    lines.append(f"Data (UTC): {run_at}")
    lines.append(run_line)
    lines.append("")
    lines.append(f"CPA médio (base: anúncios com purchase 1d): {fmt_money(avg_cpa) if avg_cpa else '—'}")
    lines.append(f"Total investido na janela (campanha): {fmt_money(total_spend)}")
    lines.append("")
    
    lines += build_lines(paused_score_df, f"Pausados por má performance (score < {SCORE_CUTOFF})", max_rows=200)
    lines += build_lines(top_df, "Top 5 melhores criativos (por score)", max_rows=5)
    
    with open("summary_email.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print("✅ Resumo do email gerado: summary_email.txt")
    
    # --------------------------------------------------------------------------
    # EXPORT (run atual + histórico)
    # --------------------------------------------------------------------------
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    
    # garante pasta history/
    history_dir = Path("history")
    history_dir.mkdir(parents=True, exist_ok=True)
    
    # arquivo "do run"
    run_file = history_dir / f"scores_{run_at}.csv"
    
    # mantém também um "latest" na raiz (opcional)
    latest_file = "creative_score_automation.csv"
    
    export_cols = [
        "campaign_name_std", "adset_name_std", "ad_name_std", "ad_id",
        "effective_status", "created_dt",
        "impressions_std", "clicks_std", "spend_std",
        "purchase_std", "cpp_std",
        "lpv_inc", "custom_event_inc", "inc_initiate_checkout", "inc_purchase",
        "ctr", "connect_rate", "bounce_rate", "cost_per_checkout", "cac",
        "performance_score",
        # flags p/ auditoria + IA
        "pause_reason",
        "is_top5",
        "w_ctr", "w_connect_rate", "w_bounce_rate", "w_cost_per_checkout", "w_cac",
    ]
    export_cols = [c for c in export_cols if c in df.columns]
    
    out = df[export_cols].copy()
    out["run_at"] = run_at
    
    # salva o run histórico
    out.to_csv(run_file, index=False)
    
    # salva o latest (para facilitar download rápido)
    out.to_csv(latest_file, index=False)
    
    print(f"✅ Histórico salvo em: {run_file}")
    print(f"✅ Latest salvo em: {latest_file}")

    active_final = int((df["effective_status"] == "ACTIVE").sum())
    print("-" * 70)
    print(f"📌 Ativos finais (no dataframe): {active_final} (mínimo requerido: {MIN_ACTIVE_AFTER})")
    print("-" * 70)





