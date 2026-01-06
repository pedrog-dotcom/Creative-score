#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import time
import random
import inspect
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timezone
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
# CONFIG / ENV
# =========================
API_VERSION = os.getenv("META_GRAPH_VERSION", "v24.0").strip() or "v24.0"
BASE_URL = f"https://graph.facebook.com/{API_VERSION}"

ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "").strip()
USER_TOKEN = os.getenv("META_USER_TOKEN", "").strip()  # opcional (melhora muito page token)
AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID", "").strip()
CAMPAIGN_ID = os.getenv("META_CAMPAIGN_ID", "").strip()

OUT_DIR = Path(os.getenv("OUT_DIR", "creatives_output"))
SCORE_CSV_PATH = Path(os.getenv("SCORE_CSV_PATH", "creative_score_automation.csv"))

VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "360"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "4"))
FRAMES_PER_VIDEO = int(os.getenv("FRAMES_PER_VIDEO", "12"))

IMAGES_DIR = OUT_DIR / "images"
VIDEOS_DIR = OUT_DIR / "videos"
FRAMES_ROOT = OUT_DIR / "frames"
MEDIA_DIR = OUT_DIR / "media"  # compat com o pipeline anterior

RUN_AT_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}


def require_env():
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
# SESSION + RETRIES
# =========================
def build_session():
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


# =========================
# HELPERS / GRAPH
# =========================
def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name)
    return name.strip().replace(" ", "_")[:150]


class GraphServer500(Exception):
    pass


class GraphRequestError(Exception):
    def __init__(self, status: int, node: str, payload):
        super().__init__(f"HTTP {status} no node={node} | payload={payload}")
        self.status = status
        self.node = node
        self.payload = payload


def graph_get(node: str, token: str, fields: str | None = None, **params) -> dict:
    url = f"{BASE_URL}/{node.lstrip('/')}"
    q = {"access_token": token}
    if fields:
        q["fields"] = fields
    q.update(params)

    # tenta algumas vezes por rate-limit
    for attempt in range(8):
        r = session.get(url, params=q, timeout=60)

        # Graph internal error
        if r.status_code == 500:
            try:
                payload = r.json()
            except Exception:
                payload = r.text[:300]
            raise GraphServer500(payload)

        if r.status_code >= 400:
            try:
                payload = r.json()
            except Exception:
                payload = r.text[:300]

            # rate limit / request limit
            if isinstance(payload, dict):
                err = (payload.get("error") or {})
                code = err.get("code")
                sub = err.get("error_subcode")
                msg = err.get("message", "")

                if code == 17 or sub == 2446079 or "User request limit reached" in str(msg):
                    wait = min(60, (2 ** attempt) + random.random() * 3)
                    print(f"  [RATE LIMIT] code={code} sub={sub}. Esperando {wait:.1f}s e tentando novamente...")
                    time.sleep(wait)
                    continue

            raise GraphRequestError(r.status_code, node, payload)

        return r.json()

    raise RuntimeError("Rate limit persistente: excedeu tentativas no Graph API.")



def graph_get_paged(node: str, token: str, fields: str, **params):
    out = []
    after = None
    page = 1

    while True:
        p = dict(params)
        p["limit"] = p.get("limit", 50)
        if after:
            p["after"] = after

        for attempt in range(4):
            try:
                data = graph_get(node, token, fields=fields, **p)
                break
            except GraphServer500:
                wait = (2 ** attempt) + random.random()
                print(f"  [WARN] Graph 500 paginando (tentativa {attempt+1}/4). Esperando {wait:.1f}s...")
                time.sleep(wait)
        else:
            raise RuntimeError("Falhou ao paginar (HTTP 500 recorrente).")

        batch = data.get("data", []) or []
        out.extend(batch)

        cursors = ((data.get("paging") or {}).get("cursors") or {})
        after = cursors.get("after")
        if not after:
            break

        page += 1
        time.sleep(0.25)

    return out


def download_file(url: str, dest_path: Path, timeout: int = 300) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    r = session.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_videofile_compat(clip, out_path: Path, **kwargs):
    sig = inspect.signature(clip.write_videofile)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return clip.write_videofile(str(out_path), **filtered)


def transcode_video(in_path: Path, out_path: Path, height: int, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(str(in_path))

    try:
        clip_resized = clip.resized(height=height)  # moviepy 2.x
    except AttributeError:
        clip_resized = clip.resize(height=height)   # moviepy 1.x

    if hasattr(clip_resized, "with_fps"):
        clip_final = clip_resized.with_fps(fps)
    else:
        clip_final = clip_resized.set_fps(fps)

    write_videofile_compat(
        clip_final,
        out_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        temp_audiofile=str(out_path.with_suffix(".m4a")),
        remove_temp=True,
        logger=None,
        verbose=False,
    )

    duration = float(clip.duration or 0.0)
    w = int(getattr(clip_final, "w", 0) or 0)
    h = int(getattr(clip_final, "h", 0) or 0)

    try: clip_final.close()
    except: pass
    try: clip_resized.close()
    except: pass
    try: clip.close()
    except: pass

    return duration, w, h


def extract_frames(video_path: Path, frames_dir: Path, frames_count: int):
    frames_dir.mkdir(parents=True, exist_ok=True)
    clip = VideoFileClip(str(video_path))
    duration = float(clip.duration or 0.0)
    if duration <= 0 or frames_count <= 0:
        clip.close()
        return []

    times = [(i + 1) / (frames_count + 1) * duration for i in range(frames_count)]
    out_paths = []

    for idx, t in enumerate(times, start=1):
        frame = clip.get_frame(t)
        out = frames_dir / f"frame_{idx:03d}.jpg"
        if Image is not None:
            Image.fromarray(frame).save(out, format="JPEG", quality=70, optimize=True)
        else:
            from imageio import imwrite
            imwrite(out, frame)
        out_paths.append(out)

    clip.close()
    return out_paths


# =========================
# PAGE TOKEN CACHE (opcional)
# =========================
PAGE_TOKEN_CACHE = None


def load_page_tokens():
    global PAGE_TOKEN_CACHE
    if PAGE_TOKEN_CACHE is not None:
        return

    PAGE_TOKEN_CACHE = {}
    token_to_use = USER_TOKEN if USER_TOKEN else ACCESS_TOKEN

    try:
        data = graph_get("me/accounts", token_to_use, fields="id,access_token", limit=200)
    except Exception as e:
        print(f"  [WARN] Não consegui carregar Page tokens via /me/accounts. Continuando sem cache. Motivo: {e}")
        PAGE_TOKEN_CACHE = {}
        return

    for p in data.get("data", []) or []:
        pid = p.get("id")
        ptok = p.get("access_token")
        if pid and ptok:
            PAGE_TOKEN_CACHE[str(pid)] = ptok


def token_for_post_id(post_id: str) -> str:
    if "_" in str(post_id):
        page_id = str(post_id).split("_", 1)[0]
        load_page_tokens()
        if PAGE_TOKEN_CACHE and page_id in PAGE_TOKEN_CACHE:
            return PAGE_TOKEN_CACHE[page_id]
    return ACCESS_TOKEN


# =========================
# ADS + CREATIVES
# =========================
def get_active_ads_basic():
    account_id = AD_ACCOUNT_ID if AD_ACCOUNT_ID.startswith("act_") else f"act_{AD_ACCOUNT_ID}"
    node = f"{account_id}/ads"

    filtering = json.dumps([
        {"field": "effective_status", "operator": "IN", "value": ["ACTIVE"]}
    ])

    fields = "id,name,effective_status,creative{id}"
    return graph_get_paged(node, ACCESS_TOKEN, fields=fields, limit=25, filtering=filtering)


def get_creative_by_id(creative_id: str) -> dict:
    fields_full = (
        "id,name,object_type,"
        "image_url,thumbnail_url,video_id,"
        "effective_instagram_media_id,instagram_permalink_url,"
        "object_story_id,object_id,effective_object_story_id,"
        "object_story_spec,asset_feed_spec"
    )

    fields_light = (
        "id,name,object_type,"
        "image_url,thumbnail_url,video_id,"
        "effective_instagram_media_id,instagram_permalink_url,"
        "object_story_id,object_id,effective_object_story_id"
    )

    try:
        return graph_get(creative_id, ACCESS_TOKEN, fields=fields_full)
    except GraphServer500:
        return graph_get(creative_id, ACCESS_TOKEN, fields=fields_light)


def get_active_ads_and_creatives():
    ads = get_active_ads_basic()
    creative_cache = {}

    for ad in ads:
        c = ad.get("creative") or {}
        c_id = c.get("id")
        if not c_id:
            continue

        if c_id not in creative_cache:
            try:
                creative_cache[c_id] = get_creative_by_id(str(c_id))
            except Exception as e:
                print(f"  [WARN] Não consegui detalhar creative {c_id}: {e}")
                creative_cache[c_id] = c

            time.sleep(0.2)

        ad["creative"] = creative_cache[c_id]

    return ads


def get_ads_by_ids(ad_ids):
    out = []
    fields = "id,name,effective_status,campaign_id,creative{id}"
    for ad_id in ad_ids:
        try:
            out.append(graph_get(ad_id, ACCESS_TOKEN, fields=fields))
        except Exception as e:
            print(f"  [WARN] Falha ao buscar ad_id={ad_id}: {e}")
    return out


# =========================
# VIDEO RESOLUTION (seu método)
# =========================
def _safe_list(x):
    return x if isinstance(x, list) else []


def extract_video_ids(creative: dict):
    ids = []
    if not creative:
        return ids

    if creative.get("video_id"):
        ids.append(str(creative["video_id"]))

    spec = creative.get("object_story_spec") or {}
    vdid = (spec.get("video_data") or {}).get("video_id")
    if vdid:
        ids.append(str(vdid))

    link_data = spec.get("link_data") or {}
    for att in _safe_list(link_data.get("child_attachments")):
        vid = att.get("video_id") or ((att.get("video_data") or {}).get("video_id"))
        if vid:
            ids.append(str(vid))

    afs = creative.get("asset_feed_spec") or {}
    for v in _safe_list(afs.get("videos")):
        vid = v.get("video_id")
        if vid:
            ids.append(str(vid))

    out, seen = [], set()
    for v in ids:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def get_video_source_from_video_id(video_id: str, tokens_to_try):
    for tok in tokens_to_try:
        try:
            data = graph_get(video_id, tok, fields="source")
            src = data.get("source")
            if src:
                return src
        except Exception:
            continue
    return None


def get_video_from_ig_media(ig_media_id: str, tokens_to_try):
    for tok in tokens_to_try:
        try:
            data = graph_get(ig_media_id, tok, fields="media_type,media_url,thumbnail_url")
            if data.get("media_type") in ("VIDEO", "REELS"):
                return data.get("media_url"), data.get("thumbnail_url")
        except Exception:
            continue
    return None, None


def get_video_url_from_post(post_id: str):
    tok = token_for_post_id(post_id)
    fields = (
        "attachments{media_type,media{source,image},subattachments{media_type,media{source,image}}},"
        "full_picture"
    )
    try:
        data = graph_get(post_id, tok, fields=fields)
        full_picture = data.get("full_picture")

        attachments = ((data.get("attachments") or {}).get("data")) or []
        queue = list(attachments)

        while queue:
            item = queue.pop(0)
            media = item.get("media") or {}

            if media.get("source"):
                thumb = (media.get("image") or {}).get("src") or full_picture
                return media["source"], thumb

            sub = ((item.get("subattachments") or {}).get("data")) or []
            queue.extend(sub)

        return None, full_picture
    except Exception:
        return None, None


def resolve_media_for_ad(ad: dict):
    creative = ad.get("creative") or {}
    obj_type = str(creative.get("object_type", "")).upper()

    video_ids = extract_video_ids(creative)
    ig_media_id = creative.get("effective_instagram_media_id")
    post_id = creative.get("effective_object_story_id") or creative.get("object_id") or creative.get("object_story_id")

    is_video_creative = bool(video_ids) or bool(ig_media_id) or (obj_type == "VIDEO")

    tokens_to_try = [ACCESS_TOKEN]
    if post_id and "_" in str(post_id):
        load_page_tokens()
        page_id = str(post_id).split("_", 1)[0]
        if PAGE_TOKEN_CACHE and page_id in PAGE_TOKEN_CACHE:
            if PAGE_TOKEN_CACHE[page_id] not in tokens_to_try:
                tokens_to_try.append(PAGE_TOKEN_CACHE[page_id])

    v_url = None
    img_url = None
    v_id = None

    # 1) video_id -> source
    for vid in video_ids:
        v_url = get_video_source_from_video_id(vid, tokens_to_try)
        if v_url:
            v_id = vid
            break

    # 2) IG media
    if not v_url and ig_media_id:
        v_url, img_url = get_video_from_ig_media(str(ig_media_id), tokens_to_try)

    # 3) post attachments
    if not v_url and post_id:
        v_url, img_url = get_video_url_from_post(str(post_id))

    # 4) fallback thumb/imagem
    if not img_url:
        img_url = creative.get("image_url") or creative.get("thumbnail_url")
    if not img_url:
        spec = creative.get("object_story_spec") or {}
        img_url = (spec.get("link_data") or {}).get("image_url")
    if not img_url:
        spec = creative.get("object_story_spec") or {}
        attachments = (spec.get("link_data") or {}).get("child_attachments") or []
        if attachments:
            img_url = attachments[0].get("image_url")

    if v_url and is_video_creative:
        return "video", v_url, img_url, v_id
    if img_url:
        return "image", img_url, None, None
    return "unknown", None, None, None


# =========================
# SCORE CSV -> paused ads
# =========================
def load_paused_ads_from_score_csv(score_csv: Path):
    if not score_csv.exists():
        print(f"⚠️ Score CSV não encontrado: {score_csv}. Vou baixar apenas ATIVOS.")
        return {}

    paused = {}
    with open(score_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ad_id = str(row.get("ad_id") or "").strip()
            if not ad_id:
                continue

            reason = (row.get("pause_reason") or row.get("paused_reason") or row.get("reason") or "").strip()
            if not reason:
                continue

            r_up = reason.upper()
            if ("HARD" in r_up) or ("SCORE_LT" in r_up) or ("SCORE" in r_up and "LT" in r_up):
                paused[ad_id] = {
                    "pause_reason": reason,
                    "performance_score": row.get("performance_score", ""),
                    "spend_std": row.get("spend_std") or row.get("spend") or "",
                    "cac": row.get("cac") or row.get("cost_per_purchase") or "",
                }

    return paused


# =========================
# MAIN
# =========================
def main():
    require_env()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Download de criativos (ativos da conta + pausados pelo score da campanha) ===")

    paused_map = load_paused_ads_from_score_csv(SCORE_CSV_PATH)
    paused_ids = list(paused_map.keys())

    # 1) ativos na conta (já vem com creative detalhado)
    active_ads = get_active_ads_and_creatives()
    active_ids = [str(a.get("id")) for a in active_ads if a.get("id")]

    # 2) pausados do score (puxa ads e detalha creative)
    extra_ads = []
    if paused_ids:
        print(f"📌 Encontrados {len(paused_ids)} anúncios pausados no score CSV para baixar também.")
        need = [i for i in paused_ids if i not in set(active_ids)]
        if need:
            extra_ads = get_ads_by_ids(need)

            # detalha creatives com cache
            creative_cache = {}
            for ad in extra_ads:
                c = ad.get("creative") or {}
                cid = c.get("id")
                if not cid:
                    continue
                if cid not in creative_cache:
                    try:
                        creative_cache[cid] = get_creative_by_id(str(cid))
                    except Exception as e:
                        print(f"  [WARN] Não consegui detalhar creative {cid} (extra): {e}")
                        creative_cache[cid] = c
                    time.sleep(0.2)
                ad["creative"] = creative_cache[cid]

    all_ads = active_ads + extra_ads
    print(f"Total para processar: {len(all_ads)} anúncios ({len(active_ads)} ativos + {len(extra_ads)} pausados do score).")

    catalog_rows = []

    for i, ad in enumerate(all_ads, 1):
        ad_id = str(ad.get("id") or "").strip()
        ad_name = str(ad.get("name") or "sem_nome").strip()
        status = str(ad.get("effective_status") or "")
        campaign_id = str(ad.get("campaign_id") or "")

        pause_reason = (paused_map.get(ad_id) or {}).get("pause_reason", "")
        is_active = (status.upper() == "ACTIVE")
        is_paused_by_score = bool(pause_reason)

        # regras: baixa ativos, e pausados do score SOMENTE da campanha alvo
        if (not is_active) and (not is_paused_by_score):
            continue
        if is_paused_by_score and campaign_id and campaign_id != CAMPAIGN_ID:
            continue

        creative = ad.get("creative") or {}
        creative_id = str(creative.get("id") or "").strip()
        if not creative_id:
            print(f"[{i}/{len(all_ads)}] ⚠️ Sem creative_id: ad_id={ad_id}")
            continue

        print(f"[{i}/{len(all_ads)}] ad_id={ad_id} | {ad_name} | status={status} | creative_id={creative_id}")

        media_type, media_url, thumb_url, video_id = resolve_media_for_ad(ad)
        if not media_url:
            print(f"  ⚠️ Não encontrei URL de mídia (type={media_type}). Pulando.")
            continue

        base = sanitize_filename(ad_name) or ad_id
        base = f"{base}_{creative_id}"

        duration_sec = ""
        w = ""
        h = ""
        frames_dir = ""
        frames_count = 0

        try:
            if media_type == "video":
                tmp = MEDIA_DIR / f"{base}_raw.mp4"
                out = VIDEOS_DIR / f"{base}.mp4"
                download_file(media_url, tmp, timeout=600)
                d, ww, hh = transcode_video(tmp, out, height=VIDEO_HEIGHT, fps=VIDEO_FPS)
                duration_sec = f"{d:.2f}"
                w, h = str(ww), str(hh)

                frames_path = FRAMES_ROOT / ad_id
                frames = extract_frames(out, frames_path, FRAMES_PER_VIDEO)
                frames_dir = str(frames_path.as_posix())
                frames_count = len(frames)

                local_path = out

            elif media_type == "image":
                tmp = MEDIA_DIR / f"{base}_raw.jpg"
                out = IMAGES_DIR / f"{base}.jpg"
                download_file(media_url, tmp, timeout=180)
                if Image is not None:
                    im = Image.open(tmp).convert("RGB")
                    im.save(out, format="JPEG", quality=70, optimize=True)
                else:
                    out = tmp
                local_path = out

            else:
                print(f"  ⚠️ Tipo desconhecido: {media_type}. Pulando.")
                continue

        except Exception as e:
            print(f"  ❌ Falha download/process: {e}")
            # salva thumb diagnóstico quando vídeo falha
            if media_type == "video" and thumb_url:
                try:
                    diag = IMAGES_DIR / f"{base}_thumb.jpg"
                    download_file(thumb_url, diag, timeout=120)
                    print("  ℹ️ Thumb salva para diagnóstico.")
                except Exception:
                    pass
            continue

        file_hash = sha256_file(local_path) if local_path.exists() else ""

        score_info = paused_map.get(ad_id) or {}

        catalog_rows.append({
            "run_at_utc": RUN_AT_UTC,
            "ad_id": ad_id,
            "ad_name": ad_name,
            "effective_status": status,
            "campaign_id": campaign_id,
            "creative_id": creative_id,
            "media_type": media_type,
            "video_id": video_id or "",
            "local_path": str(local_path.as_posix()),
            "sha256": file_hash,
            "duration_sec": duration_sec,
            "width": w,
            "height": h if h else (str(VIDEO_HEIGHT) if media_type == "video" else ""),
            "fps": str(VIDEO_FPS) if media_type == "video" else "",
            "frames_dir": frames_dir,
            "frames_count": frames_count,
            "pause_reason": score_info.get("pause_reason", ""),
            "performance_score": score_info.get("performance_score", ""),
            "spend_std": score_info.get("spend_std", ""),
            "cac": score_info.get("cac", ""),
            "is_active_account": int(is_active),
            "is_paused_by_score": int(is_paused_by_score),
        })

        time.sleep(0.15)

    # sempre cria catalog.csv (mesmo vazio)
    catalog_path = OUT_DIR / "catalog.csv"
    default_fields = [
        "run_at_utc","ad_id","ad_name","effective_status","campaign_id","creative_id",
        "media_type","video_id","local_path","sha256","duration_sec","width","height","fps",
        "frames_dir","frames_count","pause_reason","performance_score","spend_std","cac",
        "is_active_account","is_paused_by_score"
    ]

    fieldnames = list(catalog_rows[0].keys()) if catalog_rows else default_fields
    with open(catalog_path, "w", encoding="utf-8", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=fieldnames)
        wri.writeheader()
        if catalog_rows:
            wri.writerows(catalog_rows)

    print(f"✅ Catálogo gerado: {catalog_path} ({len(catalog_rows)} itens)")
    print("✅ Pronto.")


if __name__ == "__main__":
    main()
