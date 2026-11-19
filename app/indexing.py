from dataclasses import asdict
from hashlib import sha256
import json

from openai import OpenAI


def build_chunks(result):
    """Keep restaurant, date, meal, price and source together in every chunk."""
    chunks = []
    for meal_type in ("조식", "중식", "석식"):
        menus = [m for m in result.meals if m["meal_type"] == meal_type]
        if not menus:
            menus = [{"items": ["공식 페이지에 이 식사의 메뉴가 등록되지 않았습니다. 휴무 또는 운영 여부는 확인할 수 없습니다."], "price": "", "tags": []}]
        for menu in menus:
            title = f"{result.date} {result.name} {meal_type}"
            prefix = title + "\n"
            lines = ["메뉴: " + ", ".join(menu["items"]),
                     "가격: " + (menu.get("price") or "공개 정보 없음")]
            if menu.get("tags"):
                lines.append("표시: " + ", ".join(menu["tags"]))
            if menu.get("sale_status"):
                lines.append("수집 당시 페이지의 구매 상태: " + menu["sale_status"] + " (현재 상태 보장 아님)")
            text = "\n".join(lines)
            # Limit text size without detaching date/restaurant metadata.
            for start in range(0, max(len(text), 1), 800):
                body = prefix + text[start:start+800]
                identity = f"{result.code}:{result.date}:{meal_type}:{body}"
                images=local_images(result.assets,[menu.get("image","")],title)
                chunks.append({"id": sha256(identity.encode()).hexdigest()[:24],
                               "meal_type": meal_type, "title": title, "text": body,"images":images,
                               "metadata":{"sale_status":menu.get("sale_status",""),"raw_name":menu.get("raw_name","")}})
    return chunks


def local_images(assets, urls, title):
    wanted={url for url in urls if url}
    return [{"url":a["local_path"],"original_url":a["url"],"alt":a.get("alt") or title,
             "title":title,"kind":"source_image"} for a in assets
            if a.get("url") in wanted and a.get("downloaded") and a.get("content_type","").startswith("image/")]


def knowledge_chunks(scope, title, sections, *, kind="profile", images=None, metadata=None):
    chunks=[]
    for label,text in sections:
        if not text or not str(text).strip():
            continue
        for start in range(0,len(str(text)),800):
            body=title+"\n"+label+"\n"+str(text)[start:start+800]
            chunks.append({"id":sha256((scope+body).encode()).hexdigest()[:24],"title":title+" "+label,
                "text":body,"doc_type":kind,"images":images or [],"metadata":metadata or {}})
    return chunks


def index_facility(store, facility, assets, fetched_at, url, raw_record=None):
    info=facility["restaurant_info"]
    facility_id=str((raw_record or {}).get("id") or facility.get("facility_id"))
    from app.domain import RESTAURANTS
    code=next((code for code,r in RESTAURANTS.items() if r["facility_id"]==facility_id),"fac_"+facility_id)
    scope="facility:"+facility_id
    sections=[("시설 기본 정보","\n".join(f'{r["label"]}: {r["value"]}' for r in info.get("info_rows",[]))),
              ("종류",info.get("category","") or (raw_record or {}).get("category_display","")),
              ("운영 공지","\n".join(n["text"] for n in info.get("notices",[]))),
              ("상세 안내",info.get("description",{}).get("text",""))]
    image_urls=[info.get("cover_image",{}).get("url","")]+[g["url"] for g in info.get("gallery",[])]
    # Additional images inside rich descriptions are information assets too.
    image_urls += [i["url"] for i in facility.get("page_data",{}).get("images",[]) if i.get("role")!="menu"]
    images=local_images(assets,image_urls,info["name"])
    if images:
        sections.append(("시설 사진",info["name"]+"의 공개 시설 사진이 있습니다. 사진의 식재료나 건물 상태는 텍스트만으로 추정하지 않습니다."))
    chunks=knowledge_chunks(scope,info["name"],sections,images=images,metadata={"facility_id":facility_id,"category":info.get("category","")})
    store.replace_knowledge(scope,code=code,name=info["name"],url=url,fetched_at=fetched_at,
        payload={**facility,"assets":assets,"directory_record":raw_record or {}},chunks=chunks)


def index_notice(store,notice,notice_id):
    from app.domain import RESTAURANTS
    codes=notice.get("restaurant_codes") or ["campus"]
    # One source document, assigned to a specific cafeteria when unambiguous.
    code=codes[0] if len(codes)==1 else "campus"
    scope="notice:"+str(notice_id)
    chunks=knowledge_chunks(scope,notice["title"],[("공식 공지",notice["text"])],kind="notice",
        images=local_images(notice.get("assets",[]),[i["url"] for i in notice.get("images",[])],notice["title"]),
        metadata={"published":notice.get("metadata",""),"related_codes":codes})
    store.replace_knowledge(scope,code=code,name=notice["title"],url=notice["url"],fetched_at=notice["fetched_at"],payload=notice,chunks=chunks)


def persist_raw(result, settings):
    folder = settings.data_dir / "raw" / result.date / result.code
    folder.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload.pop("html")
    (folder / "source.html").write_text(result.html, encoding="utf-8")
    (folder / "normalized.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if settings.raw_s3_bucket:
        import boto3
        s3 = boto3.client("s3")
        prefix = f"raw/{result.date}/{result.code}"
        s3.put_object(Bucket=settings.raw_s3_bucket, Key=prefix+"/source.html",
                      Body=result.html.encode(), ContentType="text/html; charset=utf-8")
        s3.put_object(Bucket=settings.raw_s3_bucket, Key=prefix+"/normalized.json",
                      Body=json.dumps(payload, ensure_ascii=False).encode(), ContentType="application/json")


def index_pending(store, settings, client=None, *, menu_targets=None, scopes=None):
    if not settings.api_key and client is None:
        raise ValueError("OPENAI_API_KEY가 필요합니다. 키 없이 수집한 자료는 검색 미리보기로만 사용할 수 있습니다.")
    rows = store.pending()
    if menu_targets is not None:
        targets = {(str(day), code) for day, code in menu_targets}
        rows = [row for row in rows if row.get("doc_type", "menu") == "menu"
                and (row["day"], row["code"]) in targets]
    if scopes is not None:
        wanted=set(scopes)
        rows=[row for row in rows if row.get("scope") in wanted]
    if not rows:
        return 0
    client = client or OpenAI(api_key=settings.api_key, timeout=40, max_retries=2)
    for start in range(0, len(rows), 64):
        batch = rows[start:start+64]
        response = client.embeddings.create(model=settings.embedding_model,
            input=[r["text"] for r in batch], dimensions=settings.embedding_dimensions)
        ordered = sorted(response.data, key=lambda item: item.index)
        store.save_embeddings(batch, [item.embedding for item in ordered])
    return len(rows)
