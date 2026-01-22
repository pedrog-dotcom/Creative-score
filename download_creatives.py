#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_creatives.py - V2 (Reestruturado)

Baixa os criativos (vídeos e imagens) de TODAS as campanhas de vendas da conta,
incluindo criativos ativos e pausados.

Novidades:
- Busca todas as campanhas com objetivo de vendas
- Baixa criativos ativos E pausados
- Identifica claramente o tipo de mídia (vídeo vs estático)
- Extrai frames de vídeos para análise visual
"""

import os
import re
import csv
import json
import time
import hashlib
import logging
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# MoviePy compat
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

try:
    from PIL import Image
except ImportError:
    Image = None

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_creatives.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
API_VERSION = os.getenv("META_GRAPH_VERSION", "v24.0").strip() or "v24.0"
BASE_URL = f"https://graph.facebook.com/{API_VERSION}"

ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "").strip()
AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID", "").strip()

OUT_DIR = Path(os.getenv("OUT_DIR", "creatives_output"))
SCORE_CSV_PATH = Path(os.getenv("SCORE_CSV_PATH", "creative_score_automation.csv"))

VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "360"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "4"))
FRAMES_PER_VIDEO = int(os.getenv("FRAMES_PER_VIDEO", "12"))

IMAGES_DIR = OUT_DIR / "images"
VIDEOS_DIR = OUT_DIR / "videos"
FRAMES_ROOT = OUT_DIR / "frames"

RUN_AT_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# Objetivos de campanha de vendas
SALES_OBJECTIVES = [
    "OUTCOME_SALES",
    "CONVERSIONS",
    "PRODUCT_CATALOG_SALES",
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}


# ==============================================================================
# HELPERS
# ==============================================================================
def require_env():
    """Verifica variáveis de ambiente obrigatórias."""
    missing = []
    if not ACCESS_TOKEN:
        missing.append("META_ACCESS_TOKEN")
    if not AD_ACCOUNT_ID:
        missing.append("META_AD_ACCOUNT_ID")
    if missing:
        raise RuntimeError("Faltam variáveis de ambiente: " + ", ".join(missing))


def build_session() -> requests.Session:
    """Cria sessão HTTP com retry."""
    retry = Retry(
        total=6,
        backoff_factor=0.8,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


session = build_session()


def sanitize_filename(name: str) -> str:
    """Sanitiza nome de arquivo."""
    name = re.sub(r"[^\w\s-]", "", name)
    return name.strip().replace(" ", "_")[:150]


def graph_request(endpoint: str, params: dict, timeout: int = 60) -> dict:
    """Faz requisição à Graph API com retry."""
    url = f"{BASE_URL}/{endpoint}"
    params["access_token"] = ACCESS_TOKEN
    
    for attempt in range(5):
        try:
            r = session.get(url, params=params, timeout=timeout)
            
            if r.status_code == 429 or "User request limit reached" in r.text:
                wait = min(300, 60 * (attempt + 1))
                logger.warning(f"Rate limit. Esperando {wait}s...")
                time.sleep(wait)
                continue
            
            if r.status_code >= 500:
                logger.warning(f"Erro {r.status_code}. Tentando novamente...")
                time.sleep(5)
                continue
            
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Erro na requisição (tentativa {attempt + 1}): {e}")
            time.sleep(5)
    
    return {}


def graph_batch_request(batch: List[dict]) -> List[dict]:
    """Executa batch request na Graph API."""
    url = BASE_URL
    payload = {
        "access_token": ACCESS_TOKEN,
        "batch": json.dumps(batch)
    }
    
    for attempt in range(5):
        try:
            r = session.post(url, data=payload, timeout=120)
            
            if r.status_code == 429 or "User request limit reached" in r.text:
                wait = min(300, 60 * (attempt + 1))
                logger.warning(f"Batch rate limit. Esperando {wait}s...")
                time.sleep(wait)
                continue
            
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Erro no batch (tentativa {attempt + 1}): {e}")
            time.sleep(5)
    
    return []


def compute_sha256(filepath: Path) -> str:
    """Calcula SHA256 de um arquivo."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# 1) BUSCAR CAMPANHAS DE VENDAS
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
            })
    
    logger.info(f"Encontradas {len(campaigns)} campanhas de vendas")
    return campaigns


# ==============================================================================
# 2) BUSCAR ADS DE TODAS AS CAMPANHAS
# ==============================================================================
def fetch_ads_from_campaigns(campaign_ids: List[str]) -> List[Dict]:
    """Busca todos os ads das campanhas (ativos e pausados)."""
    logger.info("Buscando ads de todas as campanhas...")
    
    all_ads = []
    
    for campaign_id in campaign_ids:
        endpoint = f"{campaign_id}/ads"
        params = {
            "fields": "id,name,effective_status,creative{id,object_type,video_id,image_url,image_hash,thumbnail_url,object_story_spec}",
            "limit": 500,
        }
        
        data = graph_request(endpoint, params)
        
        for ad in data.get("data", []):
            creative = ad.get("creative", {})
            object_type = creative.get("object_type", "")
            
            # Determinar tipo de mídia
            if creative.get("video_id") or object_type == "VIDEO":
                media_type = "video"
            elif object_type in ["SHARE", "PHOTO"]:
                media_type = "image"
            else:
                media_type = "unknown"
            
            all_ads.append({
                "ad_id": ad.get("id"),
                "ad_name": ad.get("name"),
                "effective_status": ad.get("effective_status"),
                "campaign_id": campaign_id,
                "creative_id": creative.get("id"),
                "object_type": object_type,
                "media_type": media_type,
                "video_id": creative.get("video_id"),
                "image_url": creative.get("image_url"),
                "thumbnail_url": creative.get("thumbnail_url"),
            })
    
    logger.info(f"Total de ads encontrados: {len(all_ads)}")
    return all_ads


# ==============================================================================
# 3) BUSCAR URLs DE VÍDEOS
# ==============================================================================
def fetch_video_urls(video_ids: List[str]) -> Dict[str, str]:
    """Busca URLs de download dos vídeos."""
    logger.info(f"Buscando URLs de {len(video_ids)} vídeos...")
    
    video_urls = {}
    
    # Processa em lotes de 50
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        batch = [
            {"method": "GET", "relative_url": f"{vid}?fields=source"}
            for vid in chunk
        ]
        
        results = graph_batch_request(batch)
        
        for res in results:
            if res and res.get("code") == 200:
                body = json.loads(res.get("body", "{}"))
                vid = body.get("id")
                source = body.get("source")
                if vid and source:
                    video_urls[vid] = source
    
    logger.info(f"URLs obtidas: {len(video_urls)}")
    return video_urls


# ==============================================================================
# 4) DOWNLOAD DE MÍDIA
# ==============================================================================
def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Baixa arquivo de uma URL."""
    try:
        r = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout, stream=True)
        r.raise_for_status()
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        
        return True
    except Exception as e:
        logger.error(f"Erro ao baixar {url}: {e}")
        return False


def extract_frames(video_path: Path, frames_dir: Path, num_frames: int = 12) -> List[Path]:
    """Extrai frames de um vídeo."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    
    try:
        clip = VideoFileClip(str(video_path))
        duration = clip.duration
        
        if duration <= 0:
            clip.close()
            return []
        
        # Distribuir frames ao longo do vídeo
        times = [duration * i / (num_frames + 1) for i in range(1, num_frames + 1)]
        
        for i, t in enumerate(times):
            frame_path = frames_dir / f"frame_{i:03d}.jpg"
            try:
                frame = clip.get_frame(t)
                if Image:
                    img = Image.fromarray(frame)
                    img.save(str(frame_path), "JPEG", quality=85)
                    extracted.append(frame_path)
            except Exception as e:
                logger.warning(f"Erro ao extrair frame {i}: {e}")
        
        clip.close()
    except Exception as e:
        logger.error(f"Erro ao processar vídeo {video_path}: {e}")
    
    return extracted


# ==============================================================================
# 5) PROCESSAR CRIATIVOS
# ==============================================================================
def process_creatives(ads: List[Dict]) -> List[Dict]:
    """Processa todos os criativos (download + extração de frames)."""
    logger.info("Processando criativos...")
    
    # Criar diretórios
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Separar por tipo
    video_ads = [a for a in ads if a["media_type"] == "video" and a.get("video_id")]
    image_ads = [a for a in ads if a["media_type"] == "image" and a.get("image_url")]
    
    logger.info(f"Vídeos: {len(video_ads)} | Imagens: {len(image_ads)}")
    
    # Buscar URLs de vídeos
    video_ids = [a["video_id"] for a in video_ads]
    video_urls = fetch_video_urls(video_ids) if video_ids else {}
    
    catalog = []
    
    # Processar vídeos
    for ad in video_ads:
        ad_id = ad["ad_id"]
        video_id = ad["video_id"]
        video_url = video_urls.get(video_id)
        
        if not video_url:
            logger.warning(f"URL não encontrada para vídeo {video_id}")
            continue
        
        # Nome do arquivo
        safe_name = sanitize_filename(ad["ad_name"])
        video_path = VIDEOS_DIR / f"{ad_id}_{safe_name}.mp4"
        frames_dir = FRAMES_ROOT / ad_id
        
        # Download
        if not video_path.exists():
            logger.info(f"Baixando vídeo: {ad_id}")
            if not download_file(video_url, video_path):
                continue
        
        # Extrair frames
        if not frames_dir.exists() or not list(frames_dir.glob("*.jpg")):
            logger.info(f"Extraindo frames: {ad_id}")
            extract_frames(video_path, frames_dir, FRAMES_PER_VIDEO)
        
        # Calcular hash
        sha256 = compute_sha256(video_path) if video_path.exists() else ""
        
        catalog.append({
            "ad_id": ad_id,
            "ad_name": ad["ad_name"],
            "effective_status": ad["effective_status"],
            "campaign_id": ad["campaign_id"],
            "creative_id": ad["creative_id"],
            "media_type": "video",
            "object_type": ad["object_type"],
            "local_path": str(video_path),
            "frames_dir": str(frames_dir),
            "sha256": sha256,
            "video_id": video_id,
        })
    
    # Processar imagens
    for ad in image_ads:
        ad_id = ad["ad_id"]
        image_url = ad.get("image_url") or ad.get("thumbnail_url")
        
        if not image_url:
            continue
        
        # Nome do arquivo
        safe_name = sanitize_filename(ad["ad_name"])
        
        # Determinar extensão
        ext = ".jpg"
        if ".png" in image_url.lower():
            ext = ".png"
        
        image_path = IMAGES_DIR / f"{ad_id}_{safe_name}{ext}"
        frames_dir = FRAMES_ROOT / ad_id
        
        # Download
        if not image_path.exists():
            logger.info(f"Baixando imagem: {ad_id}")
            if not download_file(image_url, image_path):
                continue
        
        # Para imagens estáticas, copiar como "frame" único
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frames_dir / "frame_000.jpg"
        
        if not frame_path.exists() and image_path.exists():
            try:
                if Image:
                    img = Image.open(image_path)
                    img = img.convert("RGB")
                    img.save(str(frame_path), "JPEG", quality=85)
            except Exception as e:
                logger.warning(f"Erro ao converter imagem {ad_id}: {e}")
        
        # Calcular hash
        sha256 = compute_sha256(image_path) if image_path.exists() else ""
        
        catalog.append({
            "ad_id": ad_id,
            "ad_name": ad["ad_name"],
            "effective_status": ad["effective_status"],
            "campaign_id": ad["campaign_id"],
            "creative_id": ad["creative_id"],
            "media_type": "image",
            "object_type": ad["object_type"],
            "local_path": str(image_path),
            "frames_dir": str(frames_dir),
            "sha256": sha256,
            "video_id": "",
        })
    
    return catalog


# ==============================================================================
# 6) SALVAR CATÁLOGO
# ==============================================================================
def save_catalog(catalog: List[Dict]) -> None:
    """Salva o catálogo de criativos."""
    catalog_path = OUT_DIR / "catalog.csv"
    
    if not catalog:
        logger.warning("Catálogo vazio")
        return
    
    fieldnames = [
        "ad_id", "ad_name", "effective_status", "campaign_id", "creative_id",
        "media_type", "object_type", "local_path", "frames_dir", "sha256", "video_id"
    ]
    
    with open(catalog_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog)
    
    logger.info(f"Catálogo salvo: {catalog_path} ({len(catalog)} itens)")


# ==============================================================================
# 7) MAIN
# ==============================================================================
def main():
    require_env()
    
    logger.info("=" * 60)
    logger.info("Download Creatives V2 - Todas as Campanhas de Vendas")
    logger.info("=" * 60)
    
    # 1) Buscar campanhas
    campaigns = fetch_sales_campaigns()
    if not campaigns:
        logger.warning("Nenhuma campanha de vendas encontrada")
        return
    
    campaign_ids = [c["campaign_id"] for c in campaigns]
    
    # 2) Buscar ads
    ads = fetch_ads_from_campaigns(campaign_ids)
    if not ads:
        logger.warning("Nenhum ad encontrado")
        return
    
    # Estatísticas
    active_count = sum(1 for a in ads if a["effective_status"] == "ACTIVE")
    paused_count = sum(1 for a in ads if a["effective_status"] == "PAUSED")
    video_count = sum(1 for a in ads if a["media_type"] == "video")
    image_count = sum(1 for a in ads if a["media_type"] == "image")
    
    logger.info(f"Ads encontrados: {len(ads)}")
    logger.info(f"  - Ativos: {active_count}")
    logger.info(f"  - Pausados: {paused_count}")
    logger.info(f"  - Vídeos: {video_count}")
    logger.info(f"  - Imagens: {image_count}")
    
    # 3) Processar criativos
    catalog = process_creatives(ads)
    
    # 4) Salvar catálogo
    save_catalog(catalog)
    
    logger.info("=" * 60)
    logger.info(f"Download concluído: {len(catalog)} criativos processados")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
