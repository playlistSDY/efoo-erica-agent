"""Exact, cited aggregation over the complete official facility directory."""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
import threading
from urllib.parse import urldefrag, urljoin, urlparse

from app.text_normalization import normalize_korean_input


_BUSINESSES={
    "분식":(r"분식\s*(?:집|점)?|떡볶이\s*(?:집|점)|김밥\s*(?:집|점)|떡볶이(?=\s*(?:먹|파는|어디|찾))",r"분식|떡볶이|김밥"),
    "버거":(r"햄버거\s*(?:집|점)?|버거\s*(?:집|점)?",r"햄버거|버거|burger"),
    "치킨":(r"치킨\s*(?:집|점)?",r"치킨|chicken"),
    "라멘":(r"라멘\s*(?:집|점)?",r"라멘|ramen"),
    "일식":(r"일식\s*(?:집|점)|일본\s*음식",r"일식|일본식|일본라멘|라멘|ramen"),
    "중식 음식점":(r"중국집|중식당|중화\s*요리",r"중국|중화|짜장|짬뽕|chinese"),
    "서점":(r"서점|책방|(?:대학\s*)?교재|전공\s*책|서적|책",r"서점|책방|서적|교재|전공\s*책|(?<![가-힣a-z])책(?=\s*(?:[을은는이가도]|판매|취급|제공|파는|살|[,·/]|$))|bookstore|textbook"),
    "문구":(r"문구(?:\s*(?:류|점|사))?|학용품|필기구|stationery",r"문구(?:\s*(?:류|점|사))?|학용품|필기구|stationery"),
    "기념품":(r"기념품(?:\s*점)?|굿즈(?:\s*(?:샵|숍))?|souvenir",r"기념품|굿즈|souvenir"),
    "미용실":(r"미용실|머리\s*(?:자르|자를|깎)",r"미용실|헤어|hair"),
    "꽃집":(r"꽃집|꽃\s*(?:파는|살|사는)\s*곳",r"꽃집|플라워|flower"),
    "커피":(r"커피(?!숍)(?:를|가|는|이|도)?(?=\s*(?:마실|마시는|살|파는|판다|판매|취급|먹))",r"커피|coffee"),
    "복사":(r"복사|포토\s*카피",r"복사|photocopy|(?<![a-z])copy(?![a-z])"),
    "출력":(r"출력|프린트|프린팅|인쇄",r"출력|프린트|프린팅|인쇄|print"),
}


@dataclass(frozen=True)
class DirectoryQuery:
    action: str
    categories: tuple[str,...] = ()
    brand: str | None = None
    location: str | None = None
    name_terms: tuple[str,...] = ()
    raw_text: str = ""
    floor: str | None = None
    business: str | None = None
    brands: tuple[str,...] = ()
    businesses: tuple[str,...] = ()
    relation: str = "or"
    # Every group must match; alternatives inside each group use OR.
    filter_groups: tuple[tuple[tuple[str,str],...],...] = ()
    unsupported_terms: tuple[str,...] = ()
    include_details: bool = False
    goals: tuple[str,...] = ()


_CATEGORY_PATTERNS=(
    ("편의점",r"편의점"),
    ("카페/베이커리",r"카페|커피숍|베이커리|빵집"),
    ("구내식당",r"구내\s*식당|교내\s*식당"),
    ("일반음식점",r"일반\s*음식점|음식점|음식\s*매장"),
    ("자동판매기",r"자동\s*판매기|자판기"),
    ("생활서비스",r"생활\s*서비스"),
    ("인쇄/문구/서점/기념품",r"인쇄[·/\s]*문구[·/\s]*서점[·/\s]*기념품"),
)
_BRANDS={
    "7-eleven":r"세븐\s*일레븐|세븐일레븐|7[\s-]*eleven",
    "CU":r"씨유|(?<![a-z])cu(?![a-z])",
    "emart24":r"이마트\s*24|e[\s-]*mart\s*24",
    "GS25":r"지에스\s*25|gs\s*25",
    "STARBUCKS":r"스타벅스|starbucks",
}
_COUNT=re.compile(r"몇\s*(?:개|곳|군데|지점)|개수|갯수|총\s*몇|(?:시설|매장|지점)\s*수(?:는|가|야|요|[?？!\s]|$)")
_LIST=re.compile(r"목록|리스트|어디|위치|어떤|뭐\s*(?:있|가)|뭐뭐|알려|보여|소개|찾|추천|있(?:어|나요|니|나|는지|을까)|(?:먹을|마실|살|사는|파는|자를)\s*곳|할\s*수\s*있는\s*곳")
_DETAIL=re.compile(r"몇\s*시|운영\s*시간|영업\s*시간|연락처|전화|번호|사진|이미지|상세|24\s*시간|무인|유인|언제|(?:오늘|내일|모레)\s*(?:은|는)?\s*(?:안\s*)?(?:해(?:요)?|하(?:나|나요)|쉬(?:어|나요))|열(?:어|려|리)|여는|닫|마감|지금|현재\s*영업|오늘\s*영업|영업중|영업하는|주말|평일|야간|새벽|밤에|오픈|가격|메뉴|식단|학식|조식|중식(?!당)|석식|점심|저녁|아침|뭐\s*나(?:와|왔|오)|얼마|알레르|칼로리|근처|주변|가까|도보")

_LOCATION_ALIASES={"기숙사":"창의인재원","복지관":"학생복지관","제삼공학관":"제3공학관","3공학관":"제3공학관"}
_GOAL_RULES={
    "커피": (r"커피(?:를)?\s*(?:마실|마시는|한\s*잔\s*할)\s*(?:곳|매장|가게)", ("카페/베이커리",)),
    "식사": (r"(?:식사|밥)(?:을|를)?\s*(?:할|하는|먹을|먹는)(?:\s*수\s*있는|\s*만한)?\s*(?:곳|매장|가게)",
             ("구내식당", "일반음식점")),
}


def _plain(value):
    return re.sub(r"[^0-9a-z가-힣]","",str(value).casefold())


def _floor(text):
    match=re.search(r"지하\s*(\d+)\s*층?",text)
    if match:
        return "B"+match[1]
    match=re.search(r"B\s*(\d+)(?!\d)",text,re.I)
    if match:
        return "B"+match[1]
    match=re.search(r"(\d+)\s*(?:층|F)(?![a-z])",text,re.I)
    return match[1] if match else None


def _location(message,records):
    compact=_plain(message)
    candidates={"학생복지관","제1공학관","제2공학관","제3공학관","제4공학관","제5공학관","창의인재원","버스승강장","학술정보관","ERICA컨벤션센터"}
    for record in records:
        place=record.get("location","")
        place=re.sub(r"지하\s*\d+\s*층?|B\s*\d+|\d+\s*(?:층|F)","",place,flags=re.I)
        for token in re.findall(r"[가-힣0-9]+",place):
            if len(token)>1 and re.search(r"관$|센터$|인재원$|승강장$|교육동$|강의동$|동관$|서관$",token):
                candidates.add(token)
    matches=[word for word in candidates if _plain(word) in compact]
    matches += [canonical for alias,canonical in _LOCATION_ALIASES.items() if alias in message]
    return max(matches,key=len) if matches else None


def _occurrences(text, value):
    """Locate a catalog name despite Korean spacing, preserving ASCII boundaries."""
    compact, positions = [], []
    for index, character in enumerate(text.casefold()):
        if re.match(r"[0-9a-z가-힣]", character):
            compact.append(character)
            positions.append(index)
    compact = "".join(compact)
    needle = _plain(value)
    if not needle:
        return []
    result, start = [], 0
    while (found := compact.find(needle, start)) >= 0:
        end = found + len(needle)
        ascii_name = bool(re.search(r"[a-z]", needle))
        if not ascii_name or not ((found and compact[found-1].isascii() and compact[found-1].isalpha()) or
                                  (end < len(compact) and compact[end].isascii() and compact[end].isalpha())):
            result.append((positions[found], positions[end-1] + 1))
        start = found + 1
    return result


def _groups(query):
    if query.filter_groups:
        return query.filter_groups
    groups = []
    for kind, values in (("category", query.categories), ("brand", query.brands or ((query.brand,) if query.brand else ())),
                         ("business", query.businesses or ((query.business,) if query.business else ())),
                         ("name", query.name_terms), ("goal", query.goals)):
        if not values:
            continue
        if query.relation == "and" and kind == "business":
            groups.extend(((kind, value),) for value in values)
        else:
            groups.append(tuple((kind, value) for value in values))
    return tuple(groups)


def _businesses(query):
    return tuple(dict.fromkeys(value for group in _groups(query) for kind, value in group if kind == "business"))


def _goal_atoms(text, atoms, records):
    if re.search(r"명시|기재|적혀|적힌|표기|확인된", text):
        return atoms, ()
    goals=[]
    for label,(pattern,taxonomy) in _GOAL_RULES.items():
        for match in re.finditer(pattern,text):
            categories={value for kind,value,_,_ in atoms if kind=="category"}
            other_services={value for kind,value,_,_ in atoms if kind=="business" and value!=label}
            if other_services or (categories and not categories.issubset(taxonomy)):
                continue
            brands=[value for kind,value,_,_ in atoms if kind=="brand"]
            named=[record for record in records if any(
                re.search(_BRANDS[brand],record["name"],re.I) if brand in _BRANDS else _plain(brand)==_plain(record["name"])
                for brand in brands)]
            if named and not any((record.get("category_display") or record["category"]) in taxonomy for record in named):
                continue
            atoms=[atom for atom in atoms if not (atom[0]=="business" and atom[1]==label and
                                                   match.start()<=atom[2] and atom[3]<=match.end())]
            atoms.append(("goal",label,match.start(),match.end()))
            if label not in goals:
                goals.append(label)
    return atoms,tuple(goals)


def _filter_groups(atoms, text, relation):
    if not atoms:
        return ()
    parent = list(range(len(atoms)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        parent[root(right)] = root(left)

    if relation != "and":
        for index, atom in enumerate(atoms):
            for previous in range(index):
                if atoms[previous][0] == atom[0]:
                    join(previous, index)
        for index in range(1, len(atoms)):
            previous, current = atoms[index-1], atoms[index]
            gap = text[previous[3]:current[2]]
            if re.search(r"이나|또는|혹은|아니면|랑|하고|그리고|및|와|과|나", gap) and not re.search(r"중|에서", gap):
                join(index-1, index)
    grouped = {}
    for index, (kind, value, _, _) in enumerate(atoms):
        grouped.setdefault(root(index), [])
        if (kind, value) not in grouped[root(index)]:
            grouped[root(index)].append((kind, value))
    return tuple(tuple(group) for group in grouped.values())


def _unparsed_conditions(text, atoms, location, floor, *, allow_details):
    # Remove recognized entities and request grammar, never an unrecognized
    # condition. A known category must not hide an unknown qualifier.
    chars = list(text)
    spans = [(atom[2], atom[3]) for atom in atoms]
    if location:
        for name in [location, *(alias for alias, value in _LOCATION_ALIASES.items() if value == location)]:
            spans.extend(_occurrences(text, name))
    if floor:
        spans.extend(match.span() for match in re.finditer(r"지하\s*\d+\s*층?|B\s*\d+|\d+\s*(?:층|F)", text, re.I))
    for start, end in spans:
        chars[start:end] = " " * (end-start)
    residue = "".join(chars)
    if allow_details:
        residue = _DETAIL.sub(" ", residue)
        residue = re.sub(r"오늘|내일|모레|어제|이번\s*주|다음\s*주|월요일|화요일|수요일|목요일|금요일|토요일|일요일|문|여는|이용|방법", " ", residue)
    noise = (
        r"한양대학교|한양대|한양|에리카|ERICA|우리\s*학교|학교|캠퍼스|교내|공식|복지포털|등록|"
        r"어디어디|어떤|뭐뭐|어디(?:에서|서)?|위치|이름|목록|리스트|시설|매장|가게|지점|종류|분류|카테고리|업종|"
        r"전체|모든|모두|전부|총|각각|각|별|몇\s*(?:개|곳|군데|지점)|개수|갯수|"
        r"알려\s*(?:줘|주세요)|보여\s*(?:줘|주세요)|찾아\s*(?:줘|주세요)|"
        r"소개\s*(?:해\s*)?(?:줘|주세요)?|추천\s*(?:해\s*)?(?:줘|주세요)?|정리\s*(?:해\s*)?(?:줘|주세요)?|"
        r"있(?:어|어요|나요|니|나|는지|는|을까|다)|없(?:어|어요|나요|니|나|는지)|"
        r"할\s*수\s*있는|할\s*수|할수있는|해\s*주는|하는|하기|할|어떻게|돼요|돼|되는|되나요|되요|되|가능한|가능|인\s*곳|먹을|먹는|마실|마시는|살|사는|파는|팔고|"
        r"판매한다고|판다고|판매하는|취급하는|제공하는|운영하는|명시된|명시되어|기재된|적힌|표기된|둘\s*다|동시에|함께|같이|"
        r"그럼|그러면|거기|그곳|그거|그건|또는|혹은|아니면|그리고|하고|이면서|"
        r"에는|에서|중에서|으로|이라|이야|인가|인지|줄래|알려|보여|찾|"
        r"[\s?？!.,/·()+]|[의에은는이가를을도랑와과나만중로야요수개곳군데다]"
    )
    previous = None
    while residue != previous:
        previous, residue = residue, re.sub(noise, "", residue, flags=re.I)
    return residue.strip()


def parse_directory_query(message, history=(), *, records=(), allow_details=False):
    text = normalize_korean_input(message)
    if (not allow_details and _DETAIL.search(text)) or re.search(r"서울\s*캠|전국|다른\s*학교", text):
        return None
    if re.search(r"없는|아닌|제외|빼고|말고|않는|못하는|안\s*(?:파는|하는)", text):
        # Negated facility predicates need a verified NOT relation. Leaving
        # them to the broader resolver is safer than silently dropping one.
        return None
    atoms = []
    for kind, patterns in (("category", _CATEGORY_PATTERNS), ("business", [(name, value[0]) for name, value in _BUSINESSES.items()]),
                           ("brand", list(_BRANDS.items()))):
        for value, pattern in patterns:
            atoms.extend((kind, value, match.start(), match.end()) for match in re.finditer(pattern, text, re.I))
    brand_atoms = [atom for atom in atoms if atom[0] == "brand"]
    for name in {record.get("name", "") for record in records if record.get("name")}:
        for start, end in _occurrences(text, name):
            if not any(atom[2] <= start and atom[3] >= end for atom in brand_atoms):
                brand_atoms.append(("brand", name, start, end))
    # Qualified official names win over their embedded generic category/brand.
    brand_atoms = [atom for atom in brand_atoms if not any(
        other[2] <= atom[2] and other[3] >= atom[3] and other[3]-other[2] > atom[3]-atom[2]
        for other in brand_atoms)]
    categories = [atom for atom in atoms if atom[0] == "category" and not any(brand[2] <= atom[2] and brand[3] >= atom[3] for brand in brand_atoms)]
    businesses = [atom for atom in atoms if atom[0] == "business" and not any(
        outer[2] <= atom[2] and outer[3] >= atom[3] for outer in [*brand_atoms, *categories])]
    atoms = [*categories, *businesses, *brand_atoms]
    # A generic restaurant is the union of both official restaurant categories.
    for match in re.finditer("식당", text):
        if not any(atom[2] <= match.start() and atom[3] >= match.end() for atom in atoms):
            atoms += [("category", category, match.start(), match.end()) for category in ("구내식당", "일반음식점")]
    name_terms = ()
    if not atoms and "은행" in text:
        name_terms = ("은행", "bank")
        atoms += [("name", term, match.start(), match.end()) for match in re.finditer("은행", text) for term in name_terms]
    atoms,goals=_goal_atoms(text,atoms,records)
    atoms.sort(key=lambda atom: (atom[2], atom[3], atom[0]))
    # Read alternatives from gaps between recognized entities, so every
    # product/service supports 나/이나 without widening unknown qualifiers.
    explicit_or = any(re.search(r"이나|또는|혹은|아니면|(?:^|\s)나(?:\s|$)",
        text[left[3]:right[2]]) for left,right in zip(atoms,atoms[1:]))
    service_count=len({value for kind,value,_,_ in atoms if kind=="business"})
    intersection = bool(re.search(r"둘\s*다|동시에|이면서", text)) or (not explicit_or and bool(re.search(
        r"(?:함께|같이|모두).*(?:파는|판매|취급|제공|운영)|(?:파는|판매|취급|제공).*(?:함께|같이|모두)|할\s*수\s*있는", text))) or (
        not explicit_or and service_count>1 and bool(re.search(r"함께|같이",text)))
    relation = "and" if intersection else "or"
    groups = _filter_groups(atoms, text, relation)
    location, floor = _location(text, records), _floor(text)
    residue = _unparsed_conditions(text, atoms, location, floor, allow_details=allow_details)
    if residue:
        return None
    categories = tuple(dict.fromkeys(value for kind, value, _, _ in atoms if kind == "category"))
    brands = tuple(dict.fromkeys(value for kind, value, _, _ in atoms if kind == "brand"))
    businesses = tuple(dict.fromkeys(value for kind, value, _, _ in atoms if kind == "business"))
    previous = None
    followup = bool(re.search(r"그럼|그러면|거기|그곳|그거|그건|어디어디|(?:은|는)[?？!\s]*$", text))
    if followup or (not atoms and _LIST.search(text)):
        for item in reversed(history):
            if item.get("role") == "user":
                previous = parse_directory_query(item["content"], records=records, allow_details=allow_details)
                if previous:
                    break
    if previous and not atoms:
        categories = previous.categories
        brands = previous.brands or ((previous.brand,) if previous.brand else ())
        businesses = previous.businesses or ((previous.business,) if previous.business else ())
        name_terms, relation, groups = previous.name_terms, previous.relation, _groups(previous)
        goals=previous.goals
    if previous and not location and not re.search(r"전체|모두|모든", text):
        location = previous.location
    if previous and floor is None and location == previous.location and not re.search(r"전체|모두|모든", text):
        floor = previous.floor
    aggregate = bool(re.search(r"시설|매장|가게", text))
    kinds = bool(re.search(r"종류|분류|카테고리|업종", text))
    if not (categories or brands or name_terms or businesses or goals or aggregate or kinds):
        return None
    count = bool(_COUNT.search(text))
    if not (count or _LIST.search(text) or goals or kinds or previous or service_count>1 or (allow_details and groups)):
        return None
    action = "count" if count else "categories" if kinds else "list" if _LIST.search(text) or goals or allow_details or service_count>1 else previous.action
    return DirectoryQuery(action, categories, brands[0] if len(brands) == 1 else None, location,
        name_terms, text, floor, businesses[0] if len(businesses) == 1 else None,
        brands=brands, businesses=businesses, relation=relation, filter_groups=groups, include_details=allow_details,goals=goals)


def _profile_fields(info):
    return [("name", info.get("name", "")), ("category", info.get("category", "")),
            ("description", info.get("description", {}).get("text", "")),
            *[("notices", notice.get("text", "")) for notice in info.get("notices", []) if isinstance(notice, dict)]]


def _business_evidence(label, fields):
    from bs4 import BeautifulSoup

    definition = _BUSINESSES.get(label)
    if definition is None:
        return [], False
    pattern = definition[1]
    found, negative = [], False
    for field, value in fields:
        if not isinstance(value, str) or (field == "category" and re.search(r"[/·]", value)):
            continue
        text = BeautifulSoup(value, "html.parser").get_text("\n", strip=True) if "<" in value else value.strip()
        for line in text.splitlines():
            if not re.search(pattern, line, re.I):
                continue
            if re.search(r"(?:" + pattern + r")\s*(?:집|점)?(?:을|를|이|은|는|가)?\s*(?:메뉴(?:가|는|도)?\s*)?"
                         r"(?:아니|아닌|아닙|아님|없|안\s*함|(?:판매|취급|제공|운영)\s*"
                         r"(?:하지|안(?:\s*(?:해|하|함|합|한|되|돼)|(?=$|[.!?]))|중단|종료))", line, re.I):
                negative = True
                continue
            if (field, line) not in found:
                found.append((field, line))
    return found, negative


def _capability_proof(label, record, info):
    details, negative = _business_evidence(label, _profile_fields(info or {}))
    if negative:
        return None
    if details:
        return "profile", details
    catalog, negative = _business_evidence(label, [
        ("name", record.get("name", "")),
        ("category", record.get("category_display") or record.get("category", "")),
        ("description", record.get("description", "")),
        ("notice", record.get("notice", "")),
    ])
    return ("catalog", catalog) if catalog and not negative else None


def facility_capabilities(record, info=None):
    """Expose independently evidenced services, without assigning one shop type.

    Callers provide an already freshness-checked official profile. A composite
    catalog category cannot prove each individual product or service, and these
    descriptions do not establish today's stock or actual opening status.
    """
    result={}
    for label in _BUSINESSES:
        proof=_capability_proof(label,record,info)
        if proof:
            source,fields=proof
            result[label]={"source":source,"evidence":[{"field":field,"text":text} for field,text in fields]}
    return result


def capability_excerpt(label, text, max_chars=1200):
    """Keep a service's actual evidence term inside a bounded text excerpt.

    The source text is not rewritten; ellipses identify omitted context. The
    character limit includes those markers. Unknown labels or unmatched text
    use a prefix excerpt, without introducing any claimed service evidence.
    """
    if not isinstance(text,str):
        raise TypeError("Capability evidence must be text")
    if isinstance(max_chars,bool) or not isinstance(max_chars,int) or max_chars<1:
        raise ValueError("Excerpt length must be a positive integer")
    if len(text)<=max_chars:
        return text
    definition=_BUSINESSES.get(label)
    match=re.search(definition[1],text,re.I) if definition else None
    if not match:
        return text[:max_chars-1]+"…"
    if max_chars<3:
        return text[match.start():match.start()+max_chars]
    width=max_chars-2
    midpoint=(match.start()+match.end())//2
    start=max(0,min(midpoint-width//2,len(text)-width))
    end=start+width
    if start==0:
        end=max_chars-1
    elif end==len(text):
        start=len(text)-(max_chars-1)
    return ("…" if start else "")+text[start:end]+("…" if end<len(text) else "")


def _location_matches(wanted, actual):
    variants = {"창의인재원": ("창의인재원", "창의관", "인재관", "기숙사"),
                "학생복지관": ("학생복지관", "복지관"),
                "제3공학관": ("제3공학관", "제삼공학관", "3공학관")}
    return any(_plain(value) in _plain(actual) for value in variants.get(wanted, (wanted,)))


class DirectoryService:
    def __init__(self,store,settings,loader=None,*,now=None,facility_refresher=None):
        self.store,self.settings,self.loader=store,settings,loader
        self._now=now or (lambda:datetime.now(timezone.utc))
        self._lock=threading.Lock()
        self._retry_at=None
        self.facility_refresher=facility_refresher

    def _catalog(self):
        return next((row for row in self.store.knowledge() if row["scope"]=="page:facility-directory"),None)

    @staticmethod
    def _records(catalog):
        raw=(catalog or {}).get("payload",{}).get("records")
        if not isinstance(raw,list) or not raw:
            return None
        unique={}
        for record in raw:
            if (not isinstance(record,dict) or isinstance(record.get("id"),bool)
                or not isinstance(record.get("id"),int) or not record.get("name")
                or not (record.get("category_display") or record.get("category"))):
                return None
            identity=record["id"]
            if identity in unique and unique[identity]!=record:
                return None
            unique[identity]=record
        return list(unique.values())

    def _fresh(self,catalog):
        try:
            fetched=datetime.fromisoformat(catalog["fetched_at"])
            if fetched.tzinfo is None:
                fetched=fetched.replace(tzinfo=timezone.utc)
            age=self._now()-fetched
            return timedelta(0)<=age<timedelta(hours=self.settings.max_age_hours)
        except (KeyError,TypeError,ValueError):
            return False

    def parse(self,message,history=(),*,allow_details=False):
        return parse_directory_query(message,history,records=self._records(self._catalog()) or (),allow_details=allow_details)

    def _profile_map(self, profiles=None):
        rows = self.store.knowledge() if profiles is None else profiles.values() if isinstance(profiles, dict) else profiles
        return {row["scope"]: row for row in rows if isinstance(row, dict) and row.get("scope", "").startswith("facility:")}

    def _proof(self, kind, value, record, info):
        category = record.get("category_display") or record["category"]
        if kind == "category":
            return ("catalog", [("category", category)]) if category == value else None
        if kind == "goal":
            rule=_GOAL_RULES.get(value)
            return ("catalog",[("category",category)]) if rule and category in rule[1] else None
        if kind == "brand":
            pattern = _BRANDS.get(value)
            matched = re.search(pattern, record["name"], re.I) if pattern else _plain(value) == _plain(record["name"])
            return ("catalog", [("name", record["name"])]) if matched else None
        if kind == "name":
            return ("catalog", [("name", record["name"])]) if _plain(value) in _plain(record["name"]) else None
        if kind == "business":
            return _capability_proof(value,record,info)
        return None

    def _match(self, query, record, info):
        if query.unsupported_terms:
            return None
        location = info.get("location") or record.get("location", "")
        if query.location and not _location_matches(query.location, location):
            return None
        if query.floor and query.floor != _floor(location):
            return None
        proofs = {"catalog": [], "profile": []}
        for group in _groups(query):
            alternatives = [proof for kind, value in group if (proof := self._proof(kind, value, record, info))]
            if not alternatives:
                return None
            # One alternative proves an OR group. Prefer detailed subtype
            # evidence; keep all matched subtypes for a useful explanation.
            for source, fields in alternatives:
                for item in fields:
                    if item not in proofs[source]:
                        proofs[source].append(item)
        return proofs

    def select_records(self, query, records=None, profiles=None):
        if query is None or query.unsupported_terms:
            return []
        records = self._records(self._catalog()) or [] if records is None else records
        profiles = self._profile_map(profiles)
        selected = []
        for record in records:
            profile = profiles.get(f"facility:{record['id']}")
            info = profile.get("payload", {}).get("restaurant_info", {}) if profile and self._fresh(profile) else {}
            if self._match(query, record, info) is not None:
                selected.append(record)
        return selected

    def answer(self,query,*,refresh=False):
        if query is None or query.unsupported_terms:
            return None
        with self._lock:
            catalog=self._catalog()
            records=self._records(catalog)
            if not records or not self._fresh(catalog):
                if not refresh:
                    return None
                try:
                    if self._retry_at and self._now()<self._retry_at:
                        return self._unavailable()
                    loader=self.loader
                    if loader is None:
                        from app.portal import collect_directory
                        loader=collect_directory
                    loader(self.settings,self.store)
                    catalog=self._catalog()
                    records=self._records(catalog)
                    if not records or not self._fresh(catalog):
                        raise ValueError("InvalidDirectory")
                    if query.raw_text:
                        query=parse_directory_query(query.raw_text,records=records) or query
                    self._retry_at=None
                except Exception:
                    self._retry_at=self._now()+timedelta(minutes=getattr(self.settings,"refresh_retry_minutes",5))
                    return self._unavailable()
        if _businesses(query):
            return self._business_answer(query,records,catalog,refresh=refresh)
        # This route cites the catalog itself. Profile-based location or
        # service proofs belong to the detailed discovery/detail routes.
        selected=self.select_records(query,records=records,profiles={})
        selected.sort(key=lambda row:(row["name"].casefold(),row.get("location",""),row["id"]))
        counts=dict(Counter(row.get("category_display") or row["category"] for row in selected))
        label=self._label(query)
        floor=("지하"+query.floor[1:]+"층" if query.floor.startswith("B") else query.floor+"층") if query.floor else ""
        place=(query.location or "한양 ERICA")+(f" {floor}" if floor else "")
        area=place+"의 "
        last=label[-1]
        particle="는" if ("가"<=last<="힣" and (ord(last)-0xAC00)%28==0) or label in {"CU","emart24","GS25"} else "은"
        if selected:
            opening=f"{area}**{label}{particle} 공식 복지포털 등록 목록 기준 {len(selected)}곳**이에요. [S1]"
        else:
            opening=f"공식 복지포털 등록 목록에서 {area}{label}에 해당하는 시설은 찾지 못했어요. 학교 전체에 없다는 뜻은 아니에요. [S1]"
        lines=[opening]
        show_counts=query.action=="categories" or len(query.categories)>1 or not _groups(query)
        if show_counts:
            if counts:
                lines += ["","| 공식 분류 | 등록 시설 수 |","| --- | --- |"]
                lines += [f"| {self._escape(category)} | {count}곳 |" for category,count in counts.items()]
        if selected and (query.action=="list" or not show_counts):
            lines += ["","| 매장 | 위치 |","| --- | --- |"]
            lines += [f"| {self._escape(row['name'])} | {self._escape(row.get('location') or '공개 위치 정보 없음')} |" for row in selected]
        if query.categories and any("/" in category for category in query.categories):
            lines += ["","공식 목록의 복합 분류 전체를 기준으로 집계했어요."]
        if "커피" in query.goals:
            lines += ["","커피를 마실 곳을 찾는 질문은 공식 카페/베이커리 복합 분류의 후보를 함께 보여드려요. 개별 매장의 커피 판매나 좌석 제공 여부가 확인됐다는 뜻은 아니에요."]
        if "식사" in query.goals:
            lines += ["","식사할 곳을 찾는 질문은 공식 구내식당·일반음식점 분류의 후보를 보여드려요. 날짜별 메뉴나 현재 영업 여부를 확인한 결과는 아니에요."]
        if query.name_terms:
            lines += ["","시설 이름에 해당 표현이 있는 항목만 골랐어요. 업종 전체의 개수를 뜻하지는 않아요."]
        if len({row["name"] for row in selected})<len(selected):
            lines += ["","같은 브랜드라도 등록된 지점은 각각 셌어요."]
        public=[{"id":row["id"],"name":row["name"],"category":row.get("category_display") or row["category"],"location":row.get("location","")} for row in selected]
        evidence="공식 공개 시설 전체 "+str(len(records))+"곳에서 필터한 등록 시설 "+str(len(selected))+"곳.\n"+"\n".join(
            f"ID {row['id']}: {row['name']} / {row['category']} / {row['location']}" for row in public)
        images=self._catalog_images(catalog,selected)
        return {"answer":"\n".join(lines),"mode":"retrieval","model":None,"images":images,
            "sources":[{"id":"S1","title":"ERICA 공식 시설 전체 목록의 등록 시설 집계","url":catalog["url"],
                        "fetched_at":catalog["fetched_at"],"date":catalog["fetched_at"][:10],"restaurant_code":"campus",
                        "doc_type":"page","text":evidence,"images":images,"metadata":{"aggregation":"unique_facility_id","goals":list(query.goals)}}],
            "directory":{"count":len(selected),"records":public,"categories":counts,"scope":"official_portal",
                         "filters":self._filter_metadata(query)},
            "trace":["공식 전체 시설 카탈로그 조회","시설 ID별 중복 제거와 분류·브랜드·위치 필터","전체 목록 기준 정확 집계"]}

    @staticmethod
    def _escape(text):
        from html import escape
        value=escape(str(text),quote=False).replace("\n"," ").replace("\r"," ")
        return re.sub(r"([\\|*_[\]`~])",r"\\\1",value)

    @staticmethod
    def _label(query):
        if query.name_terms:
            return "시설명에 해당 표현이 있는 시설"
        groups=_groups(query)
        goal_names={"커피":"카페/베이커리 후보","식사":"구내식당·일반음식점 후보"}
        return " 중 ".join(" 또는 ".join(goal_names.get(value,value) if kind=="goal" else value for kind,value in group) for group in groups) if groups else "공개 시설"

    @staticmethod
    def _filter_metadata(query):
        return {"categories":list(query.categories),"brand":query.brand,"brands":list(query.brands),
                "business":query.business,"businesses":list(_businesses(query)),"relation":query.relation,
                "filter_groups":_groups(query),"location":query.location,"floor":query.floor,"name_terms":list(query.name_terms),"goals":list(query.goals)}

    @staticmethod
    def _catalog_images(catalog,records):
        wanted={}
        for record in records:
            value=record.get("image_url")
            if isinstance(value,str) and value.strip():
                try:
                    wanted.setdefault(urldefrag(urljoin(catalog["url"],value))[0],record)
                except ValueError:
                    continue
        images=[]
        digests=set()
        assets=catalog.get("payload",{}).get("assets",[])
        if not isinstance(assets,list):
            return images
        for asset in assets:
            if not isinstance(asset,dict) or not asset.get("downloaded") or not isinstance(asset.get("url"),str):
                continue
            try:
                original=urldefrag(asset["url"])[0]
            except ValueError:
                continue
            record=wanted.get(original)
            local=asset.get("local_path")
            content_type=str(asset.get("content_type") or "").split(";",1)[0].strip().lower()
            formats={"image/jpeg":("jpg","jpeg"),"image/jpg":("jpg","jpeg"),"image/png":("png",),
                     "image/gif":("gif",),"image/webp":("webp",),"image/avif":("avif",)}
            if record is None or not isinstance(local,str) or content_type not in formats:
                continue
            try:
                source=urlparse(original)
                official=(source.scheme=="https" and source.hostname=="life.hanyang.ac.kr" and source.port in {None,443}
                          and not source.username and not source.password)
            except ValueError:
                official=False
            match=re.fullmatch(r"/media/(?P<hash>[a-f0-9]{64})\.(jpg|jpeg|png|gif|webp|avif)",local)
            if not official or not match or match[2] not in formats[content_type] or (asset.get("sha256") and asset["sha256"]!=match["hash"]) or match["hash"] in digests:
                continue
            images.append({"url":local,"original_url":original,"alt":asset.get("alt") or record["name"],
                           "title":record["name"],"facility_id":record["id"],"kind":"source_image","source_url":catalog["url"]})
            digests.add(match["hash"])
            if len(images)==6:
                break
        return images

    def _business_answer(self,query,records,catalog,*,refresh=False):
        import json

        report=None
        if refresh:
            if self.facility_refresher is None:
                from app.facility_refresh import FacilityRefreshService
                self.facility_refresher=FacilityRefreshService(self.store,self.settings)
            report=self.facility_refresher.ensure(records)
        excluded={target["id"] for target in (report or {}).get("targets",[]) if not target.get("usable",False)}
        profiles=self._profile_map()
        matches=[]
        checked=0
        for record in records:
            profile=profiles.get(f"facility:{record['id']}")
            usable=profile is not None and record["id"] not in excluded and self._fresh(profile)
            checked+=int(usable)
            info=profile.get("payload",{}).get("restaurant_info",{}) if usable else {}
            proofs=self._match(query,record,info)
            if proofs is None:
                continue
            images=[]
            if usable:
                for row in self.store.candidates(self._now().date().isoformat(),profile["code"],doc_types=["profile"]):
                    for picture in json.loads(row.get("images") or "[]"):
                        if isinstance(picture,dict) and isinstance(picture.get("url"),str) and re.fullmatch(
                            r"/media/[a-f0-9]{64}\.(jpg|jpeg|png|gif|webp|avif)",picture["url"]):
                            if not any(image["url"]==picture["url"] for image in images):
                                images.append({**picture,"source_url":profile["url"]})
            matches.append({"record":record,"location":info.get("location") or record.get("location",""),
                "hours":info.get("opening_hours","") if usable else "", "profile":profile if usable else None,
                "proofs":proofs,"images":images})
        matches.sort(key=lambda match:(match["record"]["name"],match["location"],match["record"]["id"]))
        complete=checked==len(records)
        label=self._label(query)
        place=(query.location+"에서 ") if query.location else "교내에서 "
        lines=[f"{place}**{label} 관련 안내가 확인된 매장 {len(matches)}곳**을 찾았어요." if matches else
               f"확인한 공식 시설 이름과 상세 안내에서는 {label} 관련 매장을 찾지 못했어요. 실제로 없다는 뜻은 아니에요."]
        sources,pictures,public=[],[],[]
        if matches:
            lines += ["","| 매장 | 위치 | 공식 영업시간 | 근거 |","| --- | --- | --- | --- |"]
        for match in matches:
            record=match["record"]
            citations=[]
            combined=[]
            proof_order=[kind for kind in ("profile","catalog") if match["proofs"][kind]]
            if not proof_order:
                proof_order=["catalog"]
                match["proofs"]["catalog"]=[("name",record["name"])]
            if "profile" not in proof_order and match["profile"]:
                proof_order.append("profile")
            first_proof_id=None
            for kind in proof_order:
                source=match["profile"] if kind=="profile" else catalog
                fields=match["proofs"][kind]
                if kind=="profile" and not fields:
                    fields=[("location",match["location"]),("opening_hours",match["hours"])]
                if first_proof_id is None and match["proofs"][kind]:
                    first_proof_id=f"S{len(sources)+1}"
                source_id=f"S{len(sources)+1}"
                citations.append(f"[{source_id}]")
                location=(match["location"] if kind=="profile" else record.get("location","")) or "공개 정보 없음"
                hours=(match["hours"] if kind=="profile" else record.get("operating_time","")) or "공개 정보 없음"
                text=record["name"]+"\n위치: "+location+"\n영업시간: "+hours+"\n"+"\n".join(field+": "+value for field,value in fields)
                sources.append({"id":source_id,"title":record["name"]+(" 업종·시설 안내" if match["proofs"][kind] else " 위치·영업시간·사진"),
                    "url":source["url"],"fetched_at":source["fetched_at"],"date":source["fetched_at"][:10],
                    "restaurant_code":source["code"],"doc_type":"profile" if kind=="profile" else "page",
                    "text":text,"images":match["images"] if kind=="profile" else [],
                    "metadata":{"facility_id":record["id"],"business":label,"businesses":list(_businesses(query)),
                        "matching_fields":list(dict.fromkeys(field for field,_ in fields)),"matched_texts":[value for _,value in fields]}})
                combined.extend(field for field in fields if field not in combined)
            match["business_source_id"]=first_proof_id or citations[0][1:-1]
            match["fields"]=match["proofs"]["profile"] or match["proofs"]["catalog"]
            lines.append(f"| {self._escape(record['name'])} | {self._escape(match['location'] or '공개 위치 정보 없음')} | {self._escape(match['hours'] or '공개 시간 정보 없음')} | {' '.join(citations)} |")
            public.append({"id":record["id"],"name":record["name"],"category":record.get("category_display") or record["category"],
                "location":match["location"],"matching_fields":list(dict.fromkeys(field for field,_ in combined)),
                "matched_texts":[value for _,value in combined]})
            for picture in match["images"]:
                if len(pictures)<6 and not any(existing["url"]==picture["url"] for existing in pictures):
                    pictures.append(picture)
        if matches:
            lines += ["","찾은 근거도 함께 적어둘게요."]
            for match in matches:
                snippet=match["fields"][0][1]
                lines.append(f"- **{self._escape(match['record']['name'])}**: {self._escape(snippet)} [{match['business_source_id']}]")
        if not complete:
            lines += ["",f"공식 목록 {len(records)}곳 중 상세 안내의 최신성을 확인한 곳은 {checked}곳이에요. 나머지 시설은 상세 검색이 완료되지 않아 추가 매장이 있는지는 단정하지 않을게요."]
        elif not matches:
            lines += ["",f"공식 목록 {len(records)}곳의 이름과 상세 안내를 확인한 결과예요."]
        return {"answer":"\n".join(lines),"mode":"retrieval","model":None,"sources":sources,"images":pictures,
            "directory":{"count":len(matches),"records":public,"categories":dict(Counter(record["category"] for record in public)),
                "scope":"official_portal","business":label,"businesses":list(_businesses(query)),
                "coverage":{"listed":len(records),"details_checked":checked,"complete":complete},
                "filters":self._filter_metadata(query)},
            "trace":["공식 전체 시설과 ID별 상세 안내 연결","구조화된 분류·브랜드·업종 조건과 출처별 증거 교차 검색",f"상세 확인 {checked}/{len(records)}곳"],
            **({"facility_refresh":report} if report is not None else {})}

    @staticmethod
    def _unavailable():
        return {"answer":"공식 시설 전체 목록을 새로 확인하지 못해서 지금은 정확한 개수를 말씀드리기 어려워요. 잠시 후 다시 물어봐 주세요.",
                "mode":"insufficient","model":None,"sources":[],"images":[],"trace":["공식 전체 시설 목록 최신성 확인 실패"]}
