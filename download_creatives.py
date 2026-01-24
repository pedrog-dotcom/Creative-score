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
# 3) BUSCAR URLs DE VÍDEOS E THUMBNAILS
# ==============================================================================
def fetch_video_urls(video_ids: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Busca URLs de download dos vídeos e thumbnails.
    Retorna dict com 'source' (URL do vídeo) e 'thumbnail' (URL da thumbnail).
    """
    logger.info(f"Buscando URLs de {len(video_ids)} vídeos...")
    
    video_data = {}
    
    # Campos a buscar - source para download, thumbnails como fallback
    fields = "id,source,thumbnails,picture,permalink_url"
    
    # Processa em lotes de 50
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        batch = [
            {"method": "GET", "relative_url": f"{vid}?fields={fields}"}
            for vid in chunk
        ]
        
        results = graph_batch_request(batch)
        
        for res in results:
            if res and res.get("code") == 200:
                body = json.loads(res.get("body", "{}"))
                vid = body.get("id")
                
                if not vid:
                    continue
                
                data = {"source": None, "thumbnail": None}
                
                # Tentar obter source (URL do vídeo)
                source = body.get("source")
                if source:
                    data["source"] = source
                    logger.info(f"  Vídeo {vid}: source encontrado")
                
                # Tentar obter thumbnail
                thumbnails = body.get("thumbnails", {}).get("data", [])
                if thumbnails:
                    # Pegar a maior thumbnail disponível
                    best_thumb = max(thumbnails, key=lambda x: x.get("height", 0) * x.get("width", 0))
                    data["thumbnail"] = best_thumb.get("uri")
                    logger.info(f"  Vídeo {vid}: thumbnail encontrada")
                elif body.get("picture"):
                    data["thumbnail"] = body.get("picture")
                    logger.info(f"  Vídeo {vid}: picture encontrada")
                
                if data["source"] or data["thumbnail"]:
                    video_data[vid] = data
            else:
                # Log do erro para debug
                error_body = res.get("body", "{}") if res else "{}"
                logger.warning(f"  Erro ao buscar vídeo: {error_body[:200]}")
    
    sources_found = sum(1 for v in video_data.values() if v.get("source"))
    thumbs_found = sum(1 for v in video_data.values() if v.get("thumbnail"))
    
    logger.info(f"URLs obtidas: {len(video_data)} (sources: {sources_found}, thumbnails: {thumbs_found})")
    return video_data


def fetch_video_thumbnails_from_creative(creative_id: str) -> Optional[str]:
    """
    Busca thumbnail do vídeo via AdCreative.
    Fallback quando a API de vídeo não retorna source.
    """
    try:
        data = graph_request(
            creative_id,
            {"fields": "thumbnail_url,image_url,object_story_spec"}
        )
        
        # Tentar thumbnail_url
        if data.get("thumbnail_url"):
            return data["thumbnail_url"]
        
        # Tentar image_url
        if data.get("image_url"):
            return data["image_url"]
        
        # Tentar dentro de object_story_spec
        spec = data.get("object_story_spec", {})
        video_data = spec.get("video_data", {})
        if video_data.get("image_url"):
            return video_data["image_url"]
        
    except Exception as e:
        logger.warning(f"Erro ao buscar thumbnail do creative {creative_id}: {e}")
    
    return None


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


def transcode_video(video_path: Path, target_path: Path) -> Optional[Path]:
    """Transcodifica vídeo para altura e FPS definidos."""
    if target_path.exists():
        return target_path

    try:
        clip = VideoFileClip(str(video_path))
        resized = clip.resize(height=VIDEO_HEIGHT) if VIDEO_HEIGHT else clip
        resized.write_videofile(
            str(target_path),
            fps=VIDEO_FPS,
            codec="libx264",
            audio=False,
            preset="medium",
            threads=2,
            logger=None,
        )
        clip.close()
        resized.close()
        return target_path
    except Exception as e:
        logger.error(f"Erro ao transcodificar vídeo {video_path}: {e}")
        return None


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
        creative_id = ad.get("creative_id", "")
        
        # Obter dados do vídeo (source e/ou thumbnail)
        video_info = video_urls.get(video_id, {})
        video_source = video_info.get("source") if isinstance(video_info, dict) else video_info
        video_thumb = video_info.get("thumbnail") if isinstance(video_info, dict) else None
        
        # Nome do arquivo
        safe_name = sanitize_filename(ad["ad_name"])
        video_path = VIDEOS_DIR / f"{ad_id}_{safe_name}.mp4"
        normalized_path = VIDEOS_DIR / f"{ad_id}_{safe_name}_{VIDEO_HEIGHT}p_{VIDEO_FPS}fps.mp4"
        frames_dir = FRAMES_ROOT / ad_id
        thumbnail_path = IMAGES_DIR / f"{ad_id}_{safe_name}_thumb.jpg"
        
        video_downloaded = False
        thumbnail_downloaded = False
        
        # Tentar baixar o vídeo se tiver source
        if video_source:
            if not video_path.exists():
                logger.info(f"Baixando vídeo: {ad_id}")
                video_downloaded = download_file(video_source, video_path)
            else:
                video_downloaded = True
            
            # Transcodificar para 360p/4fps quando possível
            if video_downloaded:
                transcoded = transcode_video(video_path, normalized_path)
                if transcoded:
                    video_path = transcoded
            
            # Extrair frames do vídeo
            if video_downloaded and (not frames_dir.exists() or not list(frames_dir.glob("*.jpg"))):
                logger.info(f"Extraindo frames: {ad_id}")
                extract_frames(video_path, frames_dir, FRAMES_PER_VIDEO)
        
        # Se não conseguiu baixar o vídeo, tentar thumbnail como fallback
        if not video_downloaded:
            # Tentar thumbnail da API de vídeo
            if video_thumb:
                logger.info(f"Baixando thumbnail do vídeo {video_id} (fallback)")
                thumbnail_downloaded = download_file(video_thumb, thumbnail_path)
            
            # Tentar thumbnail via AdCreative
            if not thumbnail_downloaded and creative_id:
                logger.info(f"Buscando thumbnail via AdCreative {creative_id}")
                creative_thumb = fetch_video_thumbnails_from_creative(creative_id)
                if creative_thumb:
                    thumbnail_downloaded = download_file(creative_thumb, thumbnail_path)
            
            # Se conseguiu thumbnail, criar um "frame" a partir dela
            if thumbnail_downloaded and thumbnail_path.exists():
                frames_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frames_dir / "frame_000.jpg"
                if not frame_path.exists():
                    try:
                        import shutil
                        shutil.copy(thumbnail_path, frame_path)
                        logger.info(f"  Thumbnail copiada como frame para {ad_id}")
                    except Exception as e:
                        logger.warning(f"  Erro ao copiar thumbnail: {e}")
        
        # Só adicionar ao catálogo se tiver alguma mídia
        has_video = video_downloaded and video_path.exists()
        has_frames = frames_dir.exists() and list(frames_dir.glob("*.jpg"))
        has_thumbnail = thumbnail_downloaded and thumbnail_path.exists()
        
        if not (has_video or has_frames or has_thumbnail):
            logger.warning(f"Nenhuma mídia disponível para vídeo {video_id} (ad {ad_id})")
            continue
        
        # Calcular hash
        sha256 = ""
        if has_video:
            sha256 = compute_sha256(video_path)
        elif has_thumbnail:
            sha256 = compute_sha256(thumbnail_path)
        
        catalog.append({
            "ad_id": ad_id,
            "ad_name": ad["ad_name"],
            "effective_status": ad["effective_status"],
            "campaign_id": ad["campaign_id"],
            "creative_id": creative_id,
            "media_type": "video",
            "object_type": ad["object_type"],
            "local_path": str(video_path) if has_video else str(thumbnail_path),
            "frames_dir": str(frames_dir),
            "sha256": sha256,
            "video_id": video_id,
            "has_video_file": has_video,
            "has_thumbnail": has_thumbnail,
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
