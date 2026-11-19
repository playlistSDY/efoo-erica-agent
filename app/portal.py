"""Archive the public directory and notice bodies without executing page scripts."""
from datetime import datetime, timezone
from collections import Counter
from html import escape
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import requests

from app.assets import archive_assets
from app.domain import RESTAURANTS
from app.crawler import parse_facility_html
from app.indexing import index_facility,index_notice,knowledge_chunks

BASE = "https://life.hanyang.ac.kr"


def embedded_array(html, name):
    match = re.search(r"\bconst\s+" + re.escape(name) + r"\s*=\s*", html)
    if not match:
        raise ValueError(f"공개 페이지의 {name} 자료를 찾지 못했습니다")
    value, _ = json.JSONDecoder().raw_decode(html[match.end():].lstrip())
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("공개 페이지의 자료 형식이 변경되었습니다")
    return value


def parse_notice(html, url, fetched_at):
    soup = BeautifulSoup(html, "html.parser")
    title, content = soup.select_one(".post-title"), soup.select_one(".post-content")
    if not title or content is None:
        raise ValueError("공개 공지의 제목 또는 본문을 확인할 수 없습니다")
    text = content.get_text("\n", strip=True)
    related = []
    for code, info in RESTAURANTS.items():
        aliases = [info["name"], *info["aliases"]]
        if code == "re13":
            aliases += ["창의관식당", "기숙사"]
        if any(word in title.get_text()+text for word in aliases):
            related.append(code)
    return {"title": title.get_text(" ", strip=True), "text": text,
            "html": str(content), "url": url, "fetched_at": fetched_at,
            "metadata": soup.select_one(".post-meta").get_text(" ", strip=True) if soup.select_one(".post-meta") else "",
            "restaurant_codes": related,
            "links": [{"text": a.get_text(" ", strip=True), "url": urljoin(url,a["href"])} for a in content.select("a[href]")],
            "images": [{"url": urljoin(url,im["src"]), "alt": im.get("alt","")} for im in content.select("img[src]")]}


def collect_directory(settings, store, session=None):
    """Refresh the validated official directory with one index-page request.

    A missing/empty/malformed list cannot replace a previously valid directory.
    Public record fields and raw HTML remain audit data; page scripts are never
    executed. This operation does not call an embedding or generation model.
    """
    owned = session is None
    session = session if session is not None else requests.Session()
    source_url = BASE + "/theme/pages/facilities/index.php"
    try:
        response = session.get(source_url, timeout=(3, 12), allow_redirects=False,
            headers={"User-Agent": "EfooWebAcademicCrawler/1.0"})
        response.raise_for_status()
        final = urlparse(response.url or source_url)
        if (300 <= response.status_code < 400 or final.scheme != "https" or
            final.hostname != "life.hanyang.ac.kr" or final.port not in {None, 443} or
            final.username or final.password or final.path != "/theme/pages/facilities/index.php"):
            raise ValueError("공식 시설 목록 페이지를 확인할 수 없습니다")
        response.encoding = "utf-8"
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        main = soup.select_one("main")
        if main is None or main.select_one("#fac-grid, .fac-grid") is None:
            raise ValueError("시설 목록 페이지의 구조가 변경되었습니다")
        catalog = embedded_array(html, "dbFacilities")
        if not catalog:
            raise ValueError("비어 있는 시설 목록의 완전성을 확인할 수 없습니다")
        seen = set()
        for record in catalog:
            identifier = record.get("id")
            if (isinstance(identifier, bool) or not isinstance(identifier, int) or identifier < 1 or
                identifier in seen or not isinstance(record.get("name"), str) or not record["name"].strip() or
                not isinstance(record.get("category"), str) or not record["category"].strip()):
                raise ValueError("시설 목록의 식별자 또는 필수 정보가 올바르지 않습니다")
            if record.get("category_display") is not None and not isinstance(record["category_display"], str):
                raise ValueError("시설 분류 표시 정보가 올바르지 않습니다")
            if record.get("description") is not None and not isinstance(record["description"], str):
                raise ValueError("시설 설명 정보가 올바르지 않습니다")
            seen.add(identifier)
        declared = main.select_one('[data-cat="전체"] .fac-category-count')
        if declared is not None:
            total = declared.get_text(strip=True)
            if not total.isdecimal() or int(total) != len(catalog):
                raise ValueError("화면의 시설 수와 시설 목록이 일치하지 않습니다")
        fetched = datetime.now(timezone.utc).isoformat()
        categories = dict(Counter(record.get("category_display") or record["category"] for record in catalog))
        metadata = {"title": soup.title.get_text(" ", strip=True) if soup.title else "",
                    "language": soup.html.get("lang", "") if soup.html else "",
                    "meta": [dict(tag.attrs) for tag in soup.select("meta")]}
        overview = f"공식 복지포털의 공개 시설 목록은 총 {len(catalog)}곳입니다.\n" + "\n".join(
            f"{category}: {count}곳" for category, count in categories.items())
        directory = "\n".join(f'{record["name"]} / {record.get("category_display") or record["category"]} / {record.get("location", "")}'
                              for record in catalog)
        catalog_assets_html = html + "\n" + "\n".join(
            f'<img src="{escape(str(record.get("image_url") or ""), quote=True)}" alt="">' + (record.get("description") or "")
            for record in catalog)
        assets = archive_assets(catalog_assets_html, source_url, settings.data_dir, session=session)
        payload = {"source_url": source_url, "fetched_at": fetched, "count": len(catalog),
                   "categories": categories, "records": catalog, "page_metadata": metadata, "assets": assets}
        folder = settings.data_dir / "portal"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "facility-index.html").write_text(html, encoding="utf-8")
        (folder / "facility-directory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        scope = "page:facility-directory"
        store.replace_knowledge(scope, code="campus", name="ERICA 공개 캠퍼스 시설 전체 목록", url=source_url,
            fetched_at=fetched, payload=payload,
            chunks=knowledge_chunks(scope, "ERICA 공개 캠퍼스 시설 목록",
                [("종류와 개수", overview), ("전체 시설과 위치", directory)], kind="page"))
        return {"source_url": source_url, "fetched_at": fetched, "records": catalog,
                "categories": categories, "assets": assets}
    finally:
        if owned:
            session.close()


def _collect_portal(settings, store, session, progress):
    folder = settings.data_dir / "portal"
    folder.mkdir(parents=True, exist_ok=True)
    fetched = datetime.now(timezone.utc).isoformat()

    def fetch(path, name):
        url = BASE + path
        response = session.get(url, timeout=(3,12), headers={"User-Agent":"EfooWebAcademicCrawler/1.0"})
        response.raise_for_status()
        if response.url and not response.url.startswith(BASE + "/"):
            raise ValueError("공식 포털 외부로 이동했습니다")
        response.encoding = "utf-8"
        (folder / (name+".html")).write_text(response.text, encoding="utf-8")
        return response.text, url

    directory = collect_directory(settings, store, session=session)
    catalog, catalog_url = directory["records"], directory["source_url"]
    fetched, assets = directory["fetched_at"], directory["assets"]
    facilities, errors = [], []
    for record in catalog[:150]:
        facility_id=record.get("id")
        if not isinstance(facility_id,int):
            continue
        try:
            path="detail.php" if record.get("category")=="구내식당" else "shop-detail.php"
            body,url=fetch(f"/theme/pages/facilities/{path}?id={facility_id}",f"facility-{facility_id}")
            facility=parse_facility_html(body,expected_facility_id=str(facility_id))
            facility["facility_id"]=facility_id
            media=archive_assets(body,url,settings.data_dir,session=session)
            index_facility(store,facility,media,fetched,url,record)
            facilities.append({"id":facility_id,"name":facility["name"],"url":url,"assets":media})
            progress(f"시설 수집 {len(facilities)}/{len(catalog)} {facility['name']}")
        except (ValueError,requests.RequestException) as exc:
            errors.append({"facility_id":facility_id,"error":type(exc).__name__})
    home_html,home_url=fetch("/theme/pages/home.php","home")
    home_data={}
    for label in ("dbMenus","dbFacilitiesList","catMap"):
        try:
            match=re.search(r"\bconst\s+"+re.escape(label)+r"\s*=\s*",home_html)
            if match:
                home_data[label]=json.JSONDecoder().raw_decode(home_html[match.end():].lstrip())[0]
        except (ValueError,TypeError):
            pass
    # Keep public stock/payment/display fields as observations, not promises
    # that a purchase is possible now. No payment or authentication is invoked.
    (folder/"home-data.json").write_text(json.dumps({"url":home_url,"fetched_at":fetched,
        "data":home_data},ensure_ascii=False,indent=2),encoding="utf-8")
    home_assets=archive_assets(home_html,home_url,settings.data_dir,session=session)
    for menu in home_data.get("dbMenus",[]):
        if not isinstance(menu,dict) or not menu.get("is_payment_enabled"):
            continue
        facility_id=str(menu.get("facility_id"))
        code=next((c for c,r in RESTAURANTS.items() if r["facility_id"]==facility_id),"fac_"+facility_id)
        scope="purchase:"+str(menu.get("id"))
        title=f'{menu.get("target_date","")} {menu.get("facility_name","")} {menu.get("name","")} 공개 구매 안내'
        observed={key:menu.get(key) for key in ("target_date","meal_type","online_price","kiosk_price","stock_qty",
            "one_per_person","corner","is_sold_out","can_purchase","is_payment_enabled","purchase_closed_label")}
        body="공식 홈페이지가 수집 시점에 공개한 구매 관련 필드입니다. 현재 재고나 구매 가능 여부를 보장하지 않습니다.\n"+json.dumps(observed,ensure_ascii=False)
        store.replace_knowledge(scope,code=code,name=title,url=home_url,fetched_at=fetched,payload=menu,
            chunks=knowledge_chunks(scope,title,[("수집 당시 공개 구매 정보",body)],kind="notice",metadata={"target_date":menu.get("target_date")}))
    intro="한양대학교 ERICA 복지포털 HY-SQUARE의 공개 자료입니다. 구내식당, 일반음식점, 카페와 베이커리, 편의점, 인쇄·문구·서점·기념품, 자동판매기, 생활서비스 시설과 공지사항을 안내합니다. 학사·졸업요건·대학 전체 역사에 관한 자료는 이 수집 범위에서 확인되지 않았습니다."
    scope="page:home"
    store.replace_knowledge(scope,code="campus",name="한양 ERICA 복지포털 안내",url=home_url,fetched_at=fetched,
        payload={"data":home_data,"assets":home_assets},
        chunks=knowledge_chunks(scope,"한양 ERICA 복지포털",[("공개 서비스 범위",intro)],kind="page"))
    ticket_html,ticket_url=fetch("/theme/pages/payment/ticket-shops.php","ticket-shops")
    ticket_soup=BeautifulSoup(ticket_html,"html.parser")
    ticket_main=ticket_soup.select_one("main")
    ticket_text=ticket_main.get_text("\n",strip=True) if ticket_main else ""
    ticket_data={"text":ticket_text,"rows":[{"text":a.get_text(" ",strip=True),"url":urljoin(ticket_url,a["href"])}
        for a in ticket_soup.select(".hy-post-row[href]")]}
    scope="page:ticket-shops"
    store.replace_knowledge(scope,code="campus",name="공개 식권 판매처 안내",url=ticket_url,fetched_at=fetched,payload=ticket_data,
        chunks=knowledge_chunks(scope,"공개 식권 판매처 안내",[("공개 안내",ticket_text)],kind="page"))
    notice_html, notice_url = fetch("/theme/pages/notice/index.php?board_lang=ko", "notice-index")
    notice_records = embedded_array(notice_html,"dbNotices")
    notices = []
    seen = set()
    # The current Korean board embeds all its records; visual page buttons do
    # not represent server pages. Keep the complete embedded data as evidence.
    for record in notice_records[:100]:
        notice_id = record.get("id")
        if not isinstance(notice_id,int) or notice_id in seen:
            continue
        seen.add(notice_id)
        try:
            body, url = fetch(f"/theme/pages/notice/detail.php?id={notice_id}&board_lang=ko",f"notice-{notice_id}")
            notice = parse_notice(body,url,fetched)
            notice["raw_record"] = record
            notice["assets"] = archive_assets(body,url,settings.data_dir,session=session)
            index_notice(store,notice,notice_id)
            notices.append(notice)
        except (ValueError, requests.RequestException) as exc:
            errors.append({"notice_id":notice_id,"error":type(exc).__name__})
    output={"fetched_at":fetched,"catalog_source":catalog_url,"facility_count":len(catalog),
            "facilities":catalog,"facility_pages":facilities,"assets":assets,"notice_source":notice_url,
            "home_data":home_data,"home_assets":home_assets,
            "notice_records":notice_records,"notices":notices,"errors":errors}
    (folder/"portal.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    return output


def collect_portal(settings, store, session=None, progress=print):
    owned = session is None
    session = session if session is not None else requests.Session()
    try:
        return _collect_portal(settings, store, session, progress)
    finally:
        if owned:
            session.close()
