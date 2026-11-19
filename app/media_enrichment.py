"""Refresh dated menu photos using their own official daily pages.

Weekly tables often contain text without photos. This bounded operation only
revisits dates already collected; it never copies the selected day's photos or
menus into other days and never invokes an embedding or generation model.
"""

from dataclasses import replace
from datetime import date, datetime, timedelta

from app.assets import archive_assets
from app.crawler import MealCrawler
from app.domain import RESTAURANTS
from app.indexing import build_chunks, persist_raw


def _usable_menu_images(assets, meals):
    urls = {meal.get("image") for meal in meals if meal.get("image")}
    return {(asset["url"], asset["local_path"]) for asset in assets
            if asset.get("url") in urls and asset.get("downloaded") and asset.get("local_path")
            and (asset.get("content_type") or "").startswith("image/")}


def enrich_menu_images(store, settings, *, start_date: date, days: int, restaurant_codes=None):
    """Return counts/errors after at most seven days of existing meal snapshots.

    ``changed_collections`` counts changed menu data or usable menu photos.
    Image-only updates retain existing text embeddings through Store's normal
    snapshot replacement. The caller may index any genuinely changed text.
    ``images`` counts unique usable local menu photos across successful days.
    """
    if not isinstance(start_date, date) or isinstance(start_date, datetime):
        raise ValueError("start_date는 datetime.date여야 합니다")
    if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= 7:
        raise ValueError("사진 보강 범위는 1일부터 7일까지입니다")
    if isinstance(restaurant_codes, str):
        raise ValueError("restaurant_codes는 식당 코드 목록이어야 합니다")
    codes = list(dict.fromkeys(RESTAURANTS if restaurant_codes is None else restaurant_codes))
    if any(code not in RESTAURANTS for code in codes):
        raise ValueError("알 수 없는 식당 코드입니다")
    summary = {"start_date": start_date.isoformat(), "days": days, "attempted": 0,
               "updated_collections": 0, "changed_collections": 0, "updated_menus": 0,
               "images": 0, "skipped_missing": 0, "errors": []}
    image_paths = set()
    crawler = MealCrawler()
    try:
        for offset in range(days):
            target = start_date + timedelta(days=offset)
            for code in codes:
                try:
                    existing = store.collections(target.isoformat(), code)
                    if not existing:
                        summary["skipped_missing"] += 1
                        continue
                    previous = existing[0]
                    summary["attempted"] += 1
                    result = crawler.fetch(code, target)
                    if result.code != code or result.date != target.isoformat():
                        raise ValueError("사진 보강 요청과 식단의 날짜 또는 식당이 다릅니다")
                    assets = archive_assets(result.html, result.source_url, settings.data_dir, session=crawler.session)
                    result = replace(result, assets=assets)
                    usable = _usable_menu_images(assets, result.meals)
                    old_assets = previous.get("payload", {}).get("assets", [])
                    changed = (previous["meals"] != result.meals or
                               _usable_menu_images(old_assets, previous["meals"]) != usable)
                    persist_raw(result, settings)
                    # Intentionally do not iterate result.weekly_meals here.
                    store.replace_collection(result, build_chunks(result))
                    summary["updated_collections"] += 1
                    summary["changed_collections"] += int(changed)
                    summary["updated_menus"] += len(result.meals)
                    image_paths.update(local_path for _, local_path in usable)
                except Exception as exc:
                    summary["errors"].append({"date": target.isoformat(), "restaurant_code": code,
                                              "error": type(exc).__name__})
    finally:
        crawler.close()
    summary["images"] = len(image_paths)
    return summary
