"""Combine semantic rank with corpus-wide lexical evidence, before capping."""
from collections import Counter
import math
import re


_CONCEPTS=(
    ("결제","구매","키오스크","온라인","식권"),
    ("공지","안내","변경"),
    ("전화","연락처","전화번호"),
    ("시간","영업시간","운영시간","마감"),
    ("프린트","인쇄","출력","복사"),
    ("분식","떡볶이","김밥"),
    ("커피","카페","coffee"),
)
_STOP={"학교","한양","에리카","교내","알려줘","알려주세요","찾아줘","보여줘","뭐야","있어","어떻게","좀","그럼","오늘","내일","이번","다음","관련","대해"}


def query_terms(text):
    text=text.casefold()
    terms=set()
    for word in re.findall(r"[a-z0-9]+|[가-힣]+",text):
        word=re.sub(r"(?:에서|에는|으로|에게|은|는|이|가|을|를|에|해줘|해|인가요|야)$","",word)
        if len(word)>1 and word not in _STOP:
            terms.add(word)
    for group in _CONCEPTS:
        if any(word in text for word in group):
            terms.update(group)
    return terms


def add_lexical_rank(query,ranked):
    terms=query_terms(query)
    texts=[(row.get("title","")+"\n"+row.get("text","")).casefold() for row in ranked]
    frequencies=Counter(term for text in texts for term in terms if term in text)
    weights={term:math.log(1+(len(texts)+1)/(1+frequencies[term])) for term in terms}
    total=sum(weights.values()) or 1.
    result=[]
    for row,text in zip(ranked,texts):
        lexical=sum(weights[term] for term in terms if term in text)/total
        semantic=float(row.get("score",0))
        result.append({**row,"semantic_score":semantic,"lexical_score":lexical,"score":semantic+.35*lexical})
    return sorted(result,key=lambda row:row["score"],reverse=True)
