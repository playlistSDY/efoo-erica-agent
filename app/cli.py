import argparse
from datetime import date, timedelta
from dataclasses import replace
import json
import re
import sys

from app.crawler import MealCrawler
from app.domain import RESTAURANTS
from app.indexing import build_chunks, index_pending, persist_raw,index_facility
from app.assets import archive_assets
from app.rag import today
from app.settings import Settings
from app.store import Store
from app.refresh import MealRefreshService


def _refresh_date(value):
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("YYYY-MM-DD 형식의 실제 날짜를 입력해 주세요.") from None


def main():
    parser = argparse.ArgumentParser(description="공식 학식 수집 및 RAG 인덱스 관리")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="공식 HTML 수집 및 청크 저장")
    collect.add_argument("--date", type=date.fromisoformat, default=today())
    collect.add_argument("--days", type=int, default=1, choices=range(1, 8))
    collect.add_argument("--restaurants", nargs="+", choices=list(RESTAURANTS), default=list(RESTAURANTS))
    collect.add_argument("--index", action="store_true", help="수집 뒤 실제 OpenAI 임베딩 생성")
    collect.add_argument("--full",action="store_true",help="주간 식단·공개 시설 전체·공지·원본 이미지 함께 수집")
    photos=sub.add_parser("enrich-images",help="이미 수집한 날짜의 일별 메뉴 사진 보강")
    photos.add_argument("--date",type=date.fromisoformat,default=today())
    photos.add_argument("--days",type=int,default=1,choices=range(1,8))
    photos.add_argument("--restaurants",nargs="+",choices=list(RESTAURANTS),default=list(RESTAURANTS))
    photos.add_argument("--index",action="store_true",help="보강 중 바뀐 본문도 임베딩")
    refresh_command = sub.add_parser("refresh", help="지정한 포함 기간의 공식 식단 최신성 확인 및 갱신")
    refresh_command.add_argument("--start", type=_refresh_date, required=True, help="시작일 YYYY-MM-DD 포함")
    refresh_command.add_argument("--end", type=_refresh_date, required=True, help="종료일 YYYY-MM-DD 포함")
    refresh_command.add_argument("--restaurants", nargs="+", choices=list(RESTAURANTS), default=list(RESTAURANTS))
    refresh_command.add_argument("--force", action="store_true", help="유효한 캐시도 다시 확인하며 실패 재시도 대기시간은 유지")
    sub.add_parser("index", help="미처리 청크의 OpenAI 임베딩 생성")
    sub.add_parser("status", help="자료와 인덱스 상태")
    args = parser.parse_args()
    try:
        settings = Settings()
    except ValueError:
        if args.command == "refresh":
            refresh_command.error("수집 환경설정이 올바르지 않습니다. 날짜 범위와 갱신 설정을 확인해 주세요.")
        raise
    if args.command == "refresh":
        if args.end < args.start:
            refresh_command.error("--end는 --start와 같거나 이후 날짜여야 합니다.")
        maximum = settings.max_date_range_days
        if (args.end - args.start).days + 1 > maximum:
            refresh_command.error(f"시작일과 종료일을 포함한 기간은 최대 {maximum}일입니다.")
    store = Store(settings)
    if args.command == "refresh":
        report = MealRefreshService(store, settings).ensure(args.start, args.end, args.restaurants, args.force)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["errors"]:
            raise SystemExit(1)
        return
    if args.command == "status":
        print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
        return
    if args.command == "index":
        try:
            print(f"임베딩 생성: {index_pending(store, settings)}개 청크")
        except Exception as exc:
            print(f"인덱싱 실패: {type(exc).__name__}", file=sys.stderr)
            print("OPENAI_API_KEY, 모델 접근 권한, 네트워크와 사용 한도를 확인하세요.", file=sys.stderr)
            raise SystemExit(1)
        return
    if args.command=="enrich-images":
        from app.media_enrichment import enrich_menu_images
        report=enrich_menu_images(store,settings,start_date=args.date,days=args.days,restaurant_codes=args.restaurants)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if args.index:
            try:
                print(f"임베딩 생성: {index_pending(store,settings)}개 청크")
            except Exception as exc:
                print(f"인덱싱 실패: {type(exc).__name__}",file=sys.stderr)
                raise SystemExit(1)
        if report["errors"]:
            raise SystemExit(1)
        return
    failed = 0
    image_weeks=set()
    crawler = MealCrawler()
    try:
        for offset in range(args.days):
            target = args.date + timedelta(days=offset)
            for code in args.restaurants:
                try:
                    result = crawler.fetch(code, target)
                    if args.full:
                        if result.weekly_meals:
                            image_weeks.add(min(result.weekly_meals))
                        else:
                            image_weeks.add(result.date)
                        media=archive_assets(result.html,result.source_url,settings.data_dir,session=crawler.session)
                        result=replace(result,assets=media)
                    persist_raw(result, settings)
                    chunks = build_chunks(result)
                    store.replace_collection(result, chunks)
                    if result.restaurant_info:
                        record={"id":int(RESTAURANTS[code]["facility_id"])}
                        index_facility(store,{"restaurant_info":result.restaurant_info,"page_data":result.page_data},
                                       result.assets,result.fetched_at,result.source_url,record)
                    if args.full:
                        for week_day,menus in result.weekly_meals.items():
                            if week_day==result.date:
                                continue
                            snapshot=replace(result,date=week_day,meals=menus,
                                page_data={**result.page_data,"extraction":"visible_weekly_table","selected_date":result.date})
                            persist_raw(snapshot,settings)
                            store.replace_collection(snapshot,build_chunks(snapshot))
                    print(f"수집 완료 {target} {result.name}: 메뉴 {len(result.meals)}개, 청크 {len(chunks)}개")
                except Exception as exc:
                    failed += 1
                    print(f"수집 실패 {target} {code}: {type(exc).__name__}. 기존 자료를 보존합니다.", file=sys.stderr)
    finally:
        if args.full:
            try:
                from app.portal import collect_portal
                portal=collect_portal(settings,store,session=crawler.session)
                print(f"공개 시설 {len(portal['facility_pages'])}/{portal['facility_count']}곳, 공지 {len(portal['notices'])}건 수집")
                if portal["errors"]:
                    failed+=len(portal["errors"])
                    print(json.dumps(portal["errors"],ensure_ascii=False),file=sys.stderr)
            except Exception as exc:
                failed+=1
                print(f"공개 포털 수집 실패: {type(exc).__name__}",file=sys.stderr)
        crawler.close()
    if args.full:
        from app.media_enrichment import enrich_menu_images
        for week in sorted(image_weeks):
            report=enrich_menu_images(store,settings,start_date=date.fromisoformat(week),days=7,restaurant_codes=args.restaurants)
            print(f"일별 사진 보강 {week}: {report['updated_collections']}개 식단, 사진 {report['images']}개")
            if report["errors"]:
                failed+=len(report["errors"])
                print(json.dumps(report["errors"],ensure_ascii=False),file=sys.stderr)
    if args.index:
        try:
            print(f"임베딩 생성: {index_pending(store, settings)}개 청크")
        except Exception as exc:
            failed += 1
            print(f"인덱싱 실패: {type(exc).__name__}. 수집 자료는 보존됩니다.", file=sys.stderr)
    elif not settings.api_key:
        print("API 키 없음: 수집 자료 조회만 가능합니다. 키 설정 후 python -m app.cli index를 실행하세요.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
