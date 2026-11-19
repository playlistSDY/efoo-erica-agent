"""Resolve campus entities and answer typed fields from cited public records.

Entity IDs, field requests and conversation references are resolved before
retrieval; factual fields never depend on a model guessing a shop or number.
"""
from dataclasses import dataclass
from datetime import datetime,timezone,timedelta
import json
import re


_SLOTS={
    "location":r"위치|어디|주소|찾아가|어딨|어딜",
    "contact":r"전화|연락|문의.*번호|번호.*문의",
    "hours":r"영업|운영.*시간|24\s*시간|무인|유인|지금.*(?:열|여)|몇\s*시|언제.*(?:열|닫|끝|해)|닫|마감|휴무|쉬는\s*날|문\s*(?:열|여)|오픈|break|브레이크|휴식\s*시간|(?:오늘|내일|모레)\s*(?:은|는)?\s*(?:안\s*)?(?:해|하나|하나요|쉬어|쉬나요)|(?:주말|평일|[월화수목금토일]요일|공휴일).*(?:해|하나|돼|가능|이용)",
    "building":r"건물\s*번호|건물\s*코드",
    "images":r"사진|이미지|전경|모습",
}
_MEALS=re.compile(r"학식|식단|메뉴|아침밥|뭐\s*나(?:와|왔|오)|(?:아침|점심|저녁|조식|중식|석식).*(?:먹|밥|메뉴)")
_NOTICE_REQUEST=re.compile(r"공지|결제|키오스크|식권|구매|온라인")


def requires_notice_context(message):
    return bool(_NOTICE_REQUEST.search(message))


def requested_school_fields(message):
    return tuple(slot for slot,pattern in _SLOTS.items() if re.search(pattern,message,re.I))
_ALIAS_RULES=(
    (r"7[ -]*eleven",("세븐일레븐","세븐 일레븐")),
    (r"emart\s*24",("이마트24","이마트 24")),
    (r"^CU$",("씨유",)),
    (r"우편|우체국",("우체국","우편취급국","우편 취급국")),
    (r"은행",("은행",)),
    (r"서적|서점",("서점","책방")),
)


def _compact(text):
    return re.sub(r"[^0-9a-z가-힣]","",str(text).casefold())


def catalog_aliases(records):
    enriched=[]
    for record in records:
        copy={**record}
        aliases=[record.get("name","")]
        aliases.extend(record.get(key,"") for key in ("name_en","name_zh"))
        aliases.extend(re.sub(r"[\s-]+","",value) for value in tuple(aliases) if isinstance(value,str) and re.search(r"[a-z]",value,re.I))
        for pattern,extra in _ALIAS_RULES:
            if re.search(pattern,record.get("name",""),re.I):
                aliases.extend(extra)
        copy["aliases"]=list(dict.fromkeys(value for value in aliases if isinstance(value,str) and value))
        enriched.append(copy)
    return enriched


def explicit_facilities(message,records):
    from app.school_context import match_explicit_facilities
    return match_explicit_facilities(message,catalog=records)


@dataclass(frozen=True)
class SchoolPlan:
    slots: tuple[str,...]
    facility_ids: tuple[str,...]
    reference_facility_ids: tuple[str,...] = ()
    clarification: str | None = None
    current_status: bool = False
    active: bool = False
    hours_filter: str | None = None
    unresolved: bool = False
    weekday: str | None = None
    discovery_query: object | None = None


class SchoolService:
    def __init__(self,store,settings,directory,*,refresher=None):
        self.store,self.settings,self.directory=store,settings,directory
        self.refresher=refresher

    def _records(self):
        return self.directory._records(self.directory._catalog()) or []

    def refresh_targets(self,facility_ids):
        chosen=[record for record in self._records() if str(record["id"]) in set(facility_ids)]
        if not chosen:
            return None
        if self.refresher is None:
            from app.facility_refresh import FacilityRefreshService
            self.refresher=FacilityRefreshService(self.store,self.settings)
        return self.refresher.ensure(chosen)

    def plan(self,message,history=()):
        from app.school_context import resolve_school_context
        slots=requested_school_fields(message)
        # A field-only answer cannot satisfy the notice/payment part of a
        # mixed request. Keep every such request on the grounded RAG path.
        if _MEALS.search(message) or requires_notice_context(message):
            return SchoolPlan((),(),active=False)
        records=catalog_aliases(self._records())
        if not records:
            return SchoolPlan((),(),active=False)
        context=resolve_school_context(message,history,catalog=records)
        has_reference=bool(re.search(r"거기|그곳|그중|이중|둘\s*다|나머지|번째|그\s*매장",message))
        if not slots:
            if explicit_facilities(message,records) and not has_reference:
                return SchoolPlan((),(),active=False)
            if context.is_followup and context.selection in {"filter","ordinal","remaining","both","all","ambiguous"}:
                slots=("location",)
            else:
                return SchoolPlan((),(),active=False)
        explicit=explicit_facilities(message,records)
        hours_filter="continuous24" if re.search(r"24\s*시간",message) else None
        filter_message=re.sub(r"24\s*시간", "",message) if hours_filter else message
        current_query=self.directory.parse(filter_message,(),allow_details=True)
        discovery_query=None
        dependent=has_reference
        # Plain location/list requests already have a complete catalog path;
        # do not turn an aggregate into an accidental detail/ordinal answer.
        collection=bool(re.search(r"목록|지점|전부|모두|전체|어디어디",message))
        if slots==("location",) and not dependent and (not explicit or collection) and self.directory.parse(message,()):
            return SchoolPlan((),(),active=False)
        ids=tuple(context.facility_ids)
        references=tuple(context.previous_facility_ids)
        clarification=context.clarification
        if explicit and not ids and clarification and "같은 이름" in clarification:
            ids=tuple(context.candidate_ids)
            clarification=None
        elif current_query and not dependent and (not explicit or current_query.businesses):
            # Generic services may also be name-derived aliases of one shop.
            # Official product descriptions can prove the same capability for
            # other shops, so a name alias must not hide those candidates.
            ids=tuple(str(record["id"]) for record in self.directory.select_records(current_query,records=records))
            references=ids
            clarification=None
            if current_query.businesses:
                discovery_query=current_query
        elif current_query and dependent and ids:
            allowed={str(record["id"]) for record in self.directory.select_records(current_query,records=records)}
            ids=tuple(identity for identity in ids if identity in allowed)
        current=bool(re.search(r"지금|현재.*영업|오늘.*(?:열|영업|해|하나|쉬)|문\s*열려|영업\s*중",message))
        if not ids and not clarification and not current_query:
            return SchoolPlan(slots,(),active=True,unresolved=True)
        weekday=re.search(r"[월화수목금토일]요일|주말|평일|공휴일",message)
        return SchoolPlan(slots,ids,references or ids,clarification,current,True,hours_filter,
                          weekday=weekday[0] if weekday else None,discovery_query=discovery_query)

    @staticmethod
    def _fresh(row,hours):
        try:
            timestamp=datetime.fromisoformat(row["fetched_at"])
            if timestamp.tzinfo is None:
                timestamp=timestamp.replace(tzinfo=timezone.utc)
            return timedelta(0)<=datetime.now(timezone.utc)-timestamp<timedelta(hours=hours)
        except (KeyError,TypeError,ValueError):
            return False

    @staticmethod
    def _field(info,field):
        direct={"location":"location","contact":"contact","hours":"opening_hours","building":"building_no"}
        value=info.get(direct[field],"")
        if isinstance(value,str) and value.strip() and value.strip() not in {"정보 없음","없음","-"}:
            return value.strip()
        labels={"location":r"위치|소재지","contact":r"연락|전화","hours":r"영업|운영.*시간","building":r"건물.*번호"}
        values=[str(row.get("value","")).strip() for row in info.get("info_rows",[])
                if isinstance(row,dict) and re.search(labels[field],row.get("label","")) and row.get("value")
                and str(row.get("value","")).strip() not in {"정보 없음","없음","-","N/A"}]
        return "\n".join(values)

    @staticmethod
    def _continuous_hours(value):
        if not value or value.strip() in {"정보 없음","없음","-"}:
            return "공개 운영시간이 없어 확인할 수 없어요"
        if re.search(r"24\s*(?:시간|h)",value,re.I):
            return "공식 안내에 24시간으로 명시돼 있어요"
        # Only an unqualified staffed/unstaffed timetable can establish
        # continuous posted hours. Day/season-specific schedules need care.
        if re.search(r"평일|주말|월|화|수|목|금|토|일|휴무|공휴|학기|방학",value):
            return "요일·운영 조건이 있어 24시간으로 단정할 수 없어요"
        intervals=[]
        for start_h,start_m,end_h,end_m in re.findall(r"(\d{1,2}):(\d{2})\s*[~～–—-]\s*(\d{1,2}):(\d{2})",value):
            start,end=int(start_h)*60+int(start_m),int(end_h)*60+int(end_m)
            if start>1440 or end>1440:
                continue
            intervals.extend([(start,end)] if end>start else [(start,1440),(0,end)])
        covered=0
        for start,end in sorted(intervals):
            if start>covered:
                break
            covered=max(covered,end)
        return ("유인·무인 운영시간을 합쳐 하루 전체가 안내돼 있어요" if covered==1440 else
                "안내된 시간만으로 24시간 운영을 확인할 수 없어요")

    @staticmethod
    def _weekday_known(weekday,value):
        if not value:
            return False
        text=re.sub(r"\s+","",value).replace("요일","")
        if re.search(r"매일|연중무휴",text):
            return True
        if weekday in {"주말","평일","공휴일"}:
            if weekday in text:
                return True
            if weekday=="주말":
                return all(SchoolService._weekday_known(day+"요일",value) for day in ("토","일"))
            if weekday=="평일":
                return all(SchoolService._weekday_known(day+"요일",value) for day in "월화수목금")
            return False
        order="월화수목금토일"
        day=weekday[0]
        if day not in order:
            return False
        if "평일" in text and day in "월화수목금":
            return True
        if "주말" in text and day in "토일":
            return True
        for start,end in re.findall(r"([월화수목금토일])[~～–—-]([월화수목금토일])",text):
            first,last=order.index(start),order.index(end)
            actual=order.index(day)
            if (first<=actual<=last) if first<=last else (actual>=first or actual<=last):
                return True
        return bool(re.search(r"(?<![가-힣])"+day+r"(?=\d|[,·:/()]|휴무|운영|$)",text))

    def answer(self,plan,*,refresh=False):
        if not plan or not plan.active:
            return None
        if plan.unresolved:
            return {"answer":"연결된 공식 복지포털 목록에서 요청하신 시설이나 정보를 명확히 확인하지 못했어요. 다른 시설의 정보로 대신 답하지 않을게요. 정확한 매장명이나 지점이 있으면 함께 알려 주세요.",
                    "mode":"insufficient","model":None,"sources":[],"images":[],"selected_facility_ids":[],"reference_facility_ids":[],
                    "trace":["시설 대상을 확인하지 못해 다른 매장의 정보 사용 방지"]}
        if plan.clarification:
            return {"answer":plan.clarification,"mode":"clarification","model":None,"sources":[],"images":[],
                    "selected_facility_ids":[],"reference_facility_ids":list(plan.reference_facility_ids),
                    "trace":["대화의 시설 대상 확인 필요"]}
        records=self._records()
        by_id={str(record["id"]):record for record in records}
        chosen=[by_id[identity] for identity in plan.facility_ids if identity in by_id]
        report=None
        if refresh and (chosen or plan.discovery_query):
            if self.refresher is None:
                from app.facility_refresh import FacilityRefreshService
                self.refresher=FacilityRefreshService(self.store,self.settings)
            # A product/service may be described only in a profile. Refresh
            # discovery coverage first, rather than selecting solely from old
            # descriptions and never checking the missed shop.
            report=self.refresher.ensure(records if plan.discovery_query else chosen)
        excluded={str(target["id"]) for target in (report or {}).get("targets",[]) if not target.get("usable",False)}
        if plan.discovery_query:
            chosen=[record for record in self.directory.select_records(plan.discovery_query,records=records)
                    if str(record["id"]) not in excluded]
        profiles={row["scope"]:row for row in self.store.knowledge() if row["scope"].startswith("facility:")}
        labels={"location":"위치","contact":"연락처","hours":"공식 영업시간","building":"건물 번호"}
        fields=[slot for slot in plan.slots if slot in labels]
        sources=[]
        rows=[]
        images=[]
        failed=[]
        selected=[]
        for record in chosen:
            identity=str(record["id"])
            profile=profiles.get("facility:"+identity)
            valid=profile is not None and identity not in excluded and self._fresh(profile,self.settings.max_age_hours)
            if not valid:
                failed.append(record["name"])
                continue
            info=profile["payload"].get("restaurant_info",{})
            source_id=f"S{len(sources)+1}"
            values={field:self._field(info,field) for field in fields}
            # Always expose a branch's official location when a shared brand
            # resolves to several IDs, even if only a phone/time was requested.
            if len(chosen)>1 and "location" not in values:
                values={"location":self._field(info,"location"),**values}
            pictures=[]
            for chunk in self.store.candidates(datetime.now(timezone.utc).date().isoformat(),profile["code"],doc_types=["profile"]):
                for picture in json.loads(chunk.get("images") or "[]"):
                    if re.fullmatch(r"/media/[a-f0-9]{64}\.(jpg|jpeg|png|gif|webp|avif)",picture.get("url","")) and not any(p["url"]==picture["url"] for p in pictures):
                        pictures.append({**picture,"source_url":profile["url"]})
            text=record["name"]+"\n"+"\n".join(labels[field]+": "+(value or "수집한 공식 문자 안내에서 확인하지 못한 항목") for field,value in values.items())
            if "images" in plan.slots:
                text+="\n사진: "+("공식 수집 사진 있음" if pictures else "수집한 공식 사진 없음")
            sources.append({"id":source_id,"title":record["name"]+" 공식 시설 안내","url":profile["url"],
                "fetched_at":profile["fetched_at"],"date":profile["fetched_at"][:10],"restaurant_code":profile["code"],
                "doc_type":"profile","text":text,"images":pictures,"metadata":{"facility_id":record["id"],"requested_fields":list(plan.slots)}})
            rows.append((record,values,source_id,bool(pictures)))
            selected.append(identity)
            for picture in pictures:
                if len(images)<6 and not any(image["url"]==picture["url"] for image in images):
                    images.append(picture)
        if not rows:
            return {"answer":"요청하신 시설의 최신 공식 상세 정보를 확인하지 못했어요. 다른 매장 정보로 대신 안내하지 않을게요.",
                    "mode":"insufficient","model":None,"sources":[],"images":[],"selected_facility_ids":[],
                    "reference_facility_ids":list(plan.reference_facility_ids),"trace":["요청 시설의 최신 상세 근거 없음"]}
        escape=self.directory._escape
        lines=[]
        if plan.current_status:
            lines.append("실시간 영업 여부는 확인할 수 없어요. 방문에 참고할 수 있는 공식 운영 안내를 정리해 드릴게요.")
        if plan.hours_filter:
            lines.append("24시간 운영 여부를 공식 시간표로 구분했어요. 현재 영업 여부나 휴일의 예외 운영을 보장하는 안내는 아니에요.")
        if len(rows)==1:
            record,values,source_id,has_photos=rows[0]
            lines.append("**"+escape(record["name"])+"** 안내예요.")
            for field,value in values.items():
                lines.append(f"- **{labels[field]}**: {escape(value) if value else '공개된 공식 문자 안내에서 확인하지 못했어요.'} [{source_id}]")
            if "images" in plan.slots:
                lines.append(("공식 자료에서 수집한 사진도 함께 보여드릴게요." if has_photos else "수집한 공식 사진은 없어요.")+f" [{source_id}]")
            if plan.hours_filter:
                lines.append(self._continuous_hours(values.get("hours",""))+f" [{source_id}]")
        else:
            all_fields=list(dict.fromkeys(field for _,values,_,_ in rows for field in values))
            lines += [f"요청하신 **{len(rows)}개 시설**을 각각 정리했어요.","",
                "| 매장 | "+" | ".join(labels[field] for field in all_fields)+(" | 사진" if "images" in plan.slots else "")+(" | 24시간 안내" if plan.hours_filter else "")+" | 출처 |",
                "| --- | "+" | ".join("---" for _ in all_fields)+(" | ---" if "images" in plan.slots else "")+(" | ---" if plan.hours_filter else "")+" | --- |"]
            for record,values,source_id,has_photos in rows:
                lines.append("| "+escape(record["name"])+" | "+" | ".join(escape(values.get(field) or "공식 문자 안내에서 확인 못함") for field in all_fields)+
                    ((" | 있음" if has_photos else " | 수집 사진 없음") if "images" in plan.slots else "")+
                    (" | "+self._continuous_hours(values.get("hours","")) if plan.hours_filter else "")+f" | [{source_id}] |")
        if failed:
            lines += ["","다음 시설은 최신 상세 확인에 실패했어요: "+", ".join(escape(name) for name in failed)+"."]
        if plan.weekday and "hours" in plan.slots:
            for record,values,source_id,_ in rows:
                if not self._weekday_known(plan.weekday,values.get("hours","")):
                    lines += ["",f"**{escape(record['name'])}**의 {plan.weekday} 적용 여부는 공식 문자 안내에서 명확히 확인하지 못했어요. 평일이나 다른 요일의 시간을 대신 적용하지 않을게요. [{source_id}]"]
        references=selected if plan.discovery_query else list(plan.reference_facility_ids or selected)
        return {"answer":"\n\n".join(lines) if len(rows)==1 else "\n".join(lines),"mode":"retrieval","model":None,
            "sources":sources,"images":images,"selected_facility_ids":selected,"reference_facility_ids":references,
            "facility_context":{"selected_facility_ids":selected,"reference_facility_ids":references},
            "trace":["요청한 시설과 답변 항목 분리","시설 ID별 공식 상세 필드 조회","모든 요청 대상의 출처와 누락 필드 검증"],
            **({"facility_refresh":report} if report is not None else {})}
