#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_creatives.py

O que este script faz (bem direto):
- Baixa os criativos (imagem/vídeo) de:
  1) TODOS os anúncios ATIVOS na conta (independente de campanha)
  2) TODOS os anúncios que foram pausados pelo seu score (SCORE baixo ou HARD STOP)
     dentro da campanha definida por META_CAMPAIGN_ID

- Salva os arquivos em 360p e 4fps (para reduzir custo em análises de IA)
- Gera um catálogo (catalog.csv) com o "mapa" de tudo que foi baixado:
  ad_id, ad_name, status, caminho do arquivo, etc.

Como configurar (via variáveis de ambiente / GitHub Secrets):
- META_ACCESS_TOKEN
- META_AD_ACCOUNT_ID        (ex: act_123...)
- META_CAMPAIGN_ID          (ex: 120240998569260670)

Opcional:
- SCORE_CSV_PATH            (default: creative_score_automation.csv)
- OUT_DIR                   (default: creatives_output)
- VIDEO_HEIGHT              (default: 360)
- VIDEO_FPS                 (default: 4)
- FRAMES_PER_VIDEO          (default: 12)  -> extrai frames para análise visual barata
"""

import os
import csv
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# MoviePy compat (1.x / 2.x)
try:
    from moviepy.editor import VideoFileClip  # moviepy 1.x
except ImportError:
    from moviepy import VideoFileClip  # moviepy 2.x

try:
    from PIL import Image
except ImportError:
    Image = None


# =========================
# Config
# =========================
ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "").strip()
AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID", "").strip()
CAMPAIGN_ID = os.getenv("META_CAMPAIGN_ID", "").strip()

SCORE_CSV_PATH = os.getenv("SCORE_CSV_PATH", "creative_score_automation.csv")
OUT_DIR = Path(os.getenv("OUT_DIR", "creatives_output"))

VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "360"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "4"))
FRAMES_PER_VIDEO = int(os.getenv("FRAMES_PER_VIDEO", "12"))

GRAPH_VERSION = os.getenv("META_GRAPH_VERSION", "v20.0")  # ajuste se necessário


def _require_env():
    missing = []
    if not ACCESS_TOKEN:
        missing.append("META_ACCESS_TOKEN")
    if not AD_ACCOUNT_ID:
        missing.append("META_AD_ACCOUNT_ID")
    if not CAMPAIGN_ID:
        missing.append("META_CAMPAIGN_ID")
    if missing:
        raise RuntimeError("Faltam variáveis de ambiente: " + ", ".join(missing))


# =========================
# HTTP session com retry
# =========================
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


session = make_session()


def graph_get(node: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET no Graph API:
    https://graph.facebook.com/{version}/{node}?access_token=...&fields=...
    """
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{node}"
    params = dict(params)
    params["access_token"] = ACCESS_TOKEN
    r = session.get(url, params=params, timeout=120)
    data = r.json()
    if r.status_code >= 400 or "error" in data:
        raise RuntimeError(f"Graph error ({r.status_code}): {json.dumps(data, ensure_ascii=False)[:2000]}")
    return data


def graph_get_paged(node: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Faz paginação automática (Graph API retorna 'paging'->'next')
    """
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{node}"
    params = dict(params)
    params["access_token"] = ACCESS_TOKEN

    out: List[Dict[str, Any]] = []
    while True:
        r = session.get(url, params=params, timeout=120)
        data = r.json()
        if r.status_code >= 400 or "error" in data:
            raise RuntimeError(f"Graph error ({r.status_code}): {json.dumps(data, ensure_ascii=False)[:2000]}")
        out.extend(data.get("data", []) or [])
        nxt = (data.get("paging") or {}).get("next")
        if not nxt:
            break
        # quando vem "next", a URL já vem completa
        url = nxt
        params = {}  # params já estão embutidos em next
    return out


# =========================
# Helpers
# =========================
def safe_filename(name: str, max_len: int = 120) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:max_len] if len(name) > max_len else name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = session.get(url, stream=True, timeout=300, allow_redirects=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def transcode_video(in_path: Path, out_path: Path, height: int, fps: int) -> Tuple[float, int, int]:
    """
    Converte para (height) e (fps).
    Retorna: (duration_sec, width, height)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(str(in_path))

    # Redimensiona por altura e mantém proporção
    clip_resized = clip.resize(height=height)

    # Ajusta fps
    clip_resized = clip_resized.set_fps(fps)

    # Codec padrão ok pro GitHub Actions
    clip_resized.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        threads=2,
        logger=None,
        preset="medium",
        bitrate="600k",
    )

    duration = float(clip.duration or 0.0)
    w, h = int(clip_resized.w), int(clip_resized.h)

    clip.close()
    clip_resized.close()
    return duration, w, h


def extract_frames(video_path: Path, frames_dir: Path, frames_count: int) -> List[Path]:
    """
    Extrai N frames igualmente espaçados. Ótimo para análise visual "barata".
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(str(video_path))
    duration = float(clip.duration or 0.0)
    if duration <= 0 or frames_count <= 0:
        clip.close()
        return []

    # pontos no tempo (evita 0 exato e evita final exato)
    times = []
    for i in range(frames_count):
        t = (i + 1) / (frames_count + 1) * duration
        times.append(t)

    out_paths: List[Path] = []
    for idx, t in enumerate(times, start=1):
        frame = clip.get_frame(t)
        out = frames_dir / f"frame_{idx:03d}.jpg"
        # salvar com pillow (mais leve)
        if Image is not None:
            Image.fromarray(frame).save(out, format="JPEG", quality=70, optimize=True)
        else:
            # fallback: moviepy (mais pesado)
            from imageio import imwrite
            imwrite(out, frame)
        out_paths.append(out)

    clip.close()
    return out_paths


# =========================
# Meta: coletar ads e creatives
# =========================
def get_active_ads_in_account() -> List[Dict[str, Any]]:
    fields = "id,name,effective_status,adset_id,campaign_id,creative{id}"
    node = f"{AD_ACCOUNT_ID}/ads"
    params = {
        "fields": fields,
        "effective_status": json.dumps(["ACTIVE"]),
        "limit": 200,
    }
    return graph_get_paged(node, params)


def get_ads_by_ids(ad_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Busca detalhes de anúncios específicos (útil para pegar paused ads).
    """
    out = []
    for ad_id in ad_ids:
        data = graph_get(ad_id, {"fields": "id,name,effective_status,campaign_id,creative{id}"})
        out.append(data)
    return out


def get_creative(creative_id: str) -> Dict[str, Any]:
    fields = (
        "id,name,object_type,"
        "image_url,thumbnail_url,video_id,"
        "object_story_spec,asset_feed_spec"
    )
    return graph_get(creative_id, {"fields": fields})


def get_video_source_url(video_id: str) -> Optional[str]:
    """
    Para vídeo: /{video_id}?fields=source
    """
    try:
        data = graph_get(video_id, {"fields": "source"})
        return data.get("source")
    except Exception:
        return None


def find_media_from_creative(creative: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Retorna (media_type, url, video_id)
    - media_type: "video" ou "image" ou "unknown"
    - url: url de download (quando disponível)
    """
    # 1) campos diretos
    if creative.get("video_id"):
        vid = str(creative["video_id"])
        src = get_video_source_url(vid)
        return "video", src, vid

    if creative.get("image_url"):
        return "image", creative.get("image_url"), None

    # 2) object_story_spec
    oss = creative.get("object_story_spec") or {}
    if isinstance(oss, dict):
        video_data = oss.get("video_data") or {}
        if isinstance(video_data, dict) and video_data.get("video_id"):
            vid = str(video_data["video_id"])
            src = get_video_source_url(vid)
            return "video", src, vid
        if isinstance(video_data, dict) and video_data.get("image_url"):
            return "image", video_data.get("image_url"), None

        link_data = oss.get("link_data") or {}
        if isinstance(link_data, dict):
            # imagem
            if link_data.get("image_url"):
                return "image", link_data.get("image_url"), None
            # vídeo raro (link_data->video_id)
            if link_data.get("video_id"):
                vid = str(link_data["video_id"])
                src = get_video_source_url(vid)
                return "video", src, vid

    # 3) asset_feed_spec (ex: dynamic)
    afs = creative.get("asset_feed_spec") or {}
    if isinstance(afs, dict):
        videos = afs.get("videos") or []
        if videos and isinstance(videos, list) and isinstance(videos[0], dict):
            if videos[0].get("video_id"):
                vid = str(videos[0]["video_id"])
                src = get_video_source_url(vid)
                return "video", src, vid
        images = afs.get("images") or []
        if images and isinstance(images, list) and isinstance(images[0], dict):
            # às vezes vem url direto; às vezes vem hash
            if images[0].get("url"):
                return "image", images[0].get("url"), None

    return "unknown", None, None


# =========================
# Score CSV: quais foram pausados
# =========================
def load_paused_ads_from_score_csv(score_csv: Path) -> Dict[str, Dict[str, Any]]:
    """
    Lê creative_score_automation.csv e retorna:
    {ad_id: {pause_reason, performance_score, spend_std, spend_share, cac, ...}}
    Apenas os que foram pausados por:
    - HARD_STOP (seu script usa reason=HARD_STOP ou parecido)
    - SCORE_LT_3 (ou SCORE_LT_*)
    """
    if not score_csv.exists():
        print(f"⚠️ Score CSV não encontrado: {score_csv}. Vou baixar apenas os ATIVOS da conta.")
        return {}

    paused: Dict[str, Dict[str, Any]] = {}
    with open(score_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ad_id = str(row.get("ad_id") or "").strip()
            if not ad_id:
                continue

            # seu script imprime reason=... no log, mas no CSV pode ter coluna "pause_reason" ou similar.
            # vamos tentar achar alguns nomes comuns:
            reason = (row.get("pause_reason") or row.get("paused_reason") or row.get("reason") or "").strip()

            # fallback: se não tem coluna, não conseguimos filtrar com certeza.
            if not reason:
                continue

            r_up = reason.upper()
            if ("HARD" in r_up) or ("SCORE_LT" in r_up) or ("SCORE" in r_up and "LT" in r_up):
                paused[ad_id] = {
                    "pause_reason": reason,
                    "performance_score": row.get("performance_score"),
                    "spend_std": row.get("spend_std") or row.get("spend"),
                    "cac": row.get("cac") or row.get("cpp_inc") or row.get("cost_per_purchase"),
                    "ad_name_std": row.get("ad_name_std") or row.get("ad_name") or "",
                }
    return paused


# =========================
# Main
# =========================
def main():
    _require_env()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    media_dir = OUT_DIR / "media"
    frames_root = OUT_DIR / "frames"
    media_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    print("=== Download de criativos (ativos da conta + pausados pelo score da campanha) ===")

    paused_map = load_paused_ads_from_score_csv(Path(SCORE_CSV_PATH))
    paused_ids = list(paused_map.keys())

    # 1) ativos na conta
    active_ads = get_active_ads_in_account()
    active_ids = [a["id"] for a in active_ads if a.get("id")]

    # 2) pausados pelo score (podem estar PAUSED agora)
    extra_ads = []
    if paused_ids:
        print(f"📌 Encontrados {len(paused_ids)} anúncios pausados no score CSV para baixar também.")
        # evita duplicar ids já ativos
        need = [i for i in paused_ids if i not in set(active_ids)]
        if need:
            extra_ads = get_ads_by_ids(need)

    # junta
    all_ads = active_ads + extra_ads

    print(f"Total para processar: {len(all_ads)} anúncios ({len(active_ads)} ativos + {len(extra_ads)} pausados do score).")

    catalog_rows: List[Dict[str, Any]] = []
    for idx, ad in enumerate(all_ads, start=1):
        ad_id = str(ad.get("id"))
        ad_name = str(ad.get("name") or "").strip()
        eff_status = str(ad.get("effective_status") or "")
        campaign_id = str(ad.get("campaign_id") or "")
        creative = (ad.get("creative") or {})
        creative_id = str(creative.get("id") or "")

        if not creative_id:
            print(f"[{idx}/{len(all_ads)}] ⚠️ Sem creative_id: ad_id={ad_id}")
            continue

        # pega infos do score (se existir)
        score_info = paused_map.get(ad_id, {})
        pause_reason = score_info.get("pause_reason", "")

        # filtra: queremos baixar todos os ATIVOS da conta, e também os PAUSADOS do score,
        # mas só se forem da campanha alvo.
        is_active_account = (eff_status.upper() == "ACTIVE")
        is_paused_by_score = bool(pause_reason)

        if (not is_active_account) and (not is_paused_by_score):
            continue

        if is_paused_by_score and campaign_id and campaign_id != CAMPAIGN_ID:
            # pausado por score mas de outra campanha (por segurança)
            continue

        print(f"[{idx}/{len(all_ads)}] ad_id={ad_id} | {safe_filename(ad_name)} | status={eff_status} | creative_id={creative_id}")

        # creative detalhado
        try:
            cr = get_creative(creative_id)
        except Exception as e:
            print(f"   ❌ Falha ao buscar creative {creative_id}: {e}")
            continue

        media_type, url, video_id = find_media_from_creative(cr)
        if not url:
            print(f"   ⚠️ Não encontrei URL de mídia (type={media_type}). Pulando.")
            continue

        # paths
        base_name = safe_filename(f"{ad_id}_{ad_name}") or ad_id
        raw_path = media_dir / f"{base_name}_raw"
        out_path = media_dir / f"{base_name}"

        # download + process
        local_media_path = None
        duration_sec = None
        w = None
        h = None
        frames_paths: List[Path] = []

        try:
            if media_type == "video":
                raw_mp4 = raw_path.with_suffix(".mp4")
                download_file(url, raw_mp4)

                out_mp4 = out_path.with_suffix(".mp4")
                duration_sec, w, h = transcode_video(raw_mp4, out_mp4, height=VIDEO_HEIGHT, fps=VIDEO_FPS)
                local_media_path = out_mp4

                # frames
                frames_dir = frames_root / ad_id
                frames_paths = extract_frames(out_mp4, frames_dir, FRAMES_PER_VIDEO)

            elif media_type == "image":
                raw_img = raw_path.with_suffix(".jpg")
                download_file(url, raw_img)

                # otimiza: re-salva em jpeg mais leve
                out_img = out_path.with_suffix(".jpg")
                out_img.parent.mkdir(parents=True, exist_ok=True)
                if Image is not None:
                    im = Image.open(raw_img).convert("RGB")
                    im.save(out_img, format="JPEG", quality=70, optimize=True)
                else:
                    # sem pillow, mantém original
                    out_img = raw_img
                local_media_path = out_img
            else:
                print(f"   ⚠️ Tipo desconhecido: {media_type}. Pulando.")
                continue
        except Exception as e:
            print(f"   ❌ Falha download/process: {e}")
            continue

        # hash (dedupe)
        file_hash = sha256_file(local_media_path) if local_media_path and local_media_path.exists() else ""

        catalog_rows.append({
            "ad_id": ad_id,
            "ad_name": ad_name,
            "effective_status": eff_status,
            "campaign_id": campaign_id,
            "creative_id": creative_id,
            "media_type": media_type,
            "video_id": video_id or "",
            "local_path": str(local_media_path.as_posix()) if local_media_path else "",
            "sha256": file_hash,
            "duration_sec": f"{duration_sec:.2f}" if duration_sec is not None else "",
            "width": w or "",
            "height": h or (VIDEO_HEIGHT if media_type == "video" else ""),
            "fps": VIDEO_FPS if media_type == "video" else "",
            "frames_dir": str((frames_root / ad_id).as_posix()) if frames_paths else "",
            "frames_count": len(frames_paths),
            "pause_reason": pause_reason,
            "performance_score": score_info.get("performance_score", ""),
            "spend_std": score_info.get("spend_std", ""),
            "cac": score_info.get("cac", ""),
            "is_active_account": int(is_active_account),
            "is_paused_by_score": int(is_paused_by_score),
        })

        time.sleep(0.15)  # evita martelar API

    # salvar catálogo
    # Sempre gera o catálogo, mesmo vazio.
    # Isso evita quebrar a etapa seguinte (AI), que depende da existência do arquivo.
    catalog_path = OUT_DIR / "catalog.csv"
    default_fields = [
        "ad_id",
        "ad_name",
        "ad_name_std",
        "effective_status",
        "campaign_id",
        "adset_id",
        "creative_id",
        "media_type",
        "local_path",
        "frames_dir",
        "source",
    ]
    fieldnames = list(catalog_rows[0].keys()) if catalog_rows else default_fields
    with open(catalog_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if catalog_rows:
            writer.writerows(catalog_rows)

    print(f"✅ Catálogo gerado: {catalog_path} ({len(catalog_rows)} itens)")
    print("✅ Pronto.")


if __name__ == "__main__":
    main()
