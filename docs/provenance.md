# 학식 크롤러 코드 출처

이 프로젝트의 학식 HTML 파싱 로직과 회귀 테스트용 HTML은 사용자가 지정한
[playlistSDY/efoo-kakao-bot](https://github.com/playlistSDY/efoo-kakao-bot)에서 가져와
웹 RAG 서비스에 맞게 분리했다. 원격 저장소에는 변경을 전송하지 않았다.

- 기준 커밋: `0900e921556eb020a44ea37e3f580a6a1cf28fc6`
- 참고 파일: `app/services/meals/fetcher.py`, `app/domain/meal_images.py`,
  `app/domain/restaurants/info.py`, `tests/test_meal_fetcher.py`
- 테스트 HTML: 원본 `tests/fixtures/meals/*.html`을 `tests/fixtures/`에 보존했다.
  날짜가 2026년 9월인 회귀 검증 자료이며 현재 식단으로 사용하지 않는다.
- 적용 범위: 식당 코드와 시설 ID, 식단 HTML 선택자, 한글 메뉴와 가격 파싱,
  날짜 및 식당 검증, 빈 식단과 수집 실패의 구분
- 새로 구성한 부분: DB 및 카카오톡 의존성 제거, `CrawlResult` 수집 결과 계약,
  수집 시각과 원문 URL, 웹 서비스용 문서 적재 연결

기준 커밋에는 LICENSE 또는 COPYING 파일이 없어 공개 라이선스가 선언되었다고
가정하지 않았다. 지정된 프로젝트에서 사용자가 요청한 범위의 코드를 가져왔으며,
제3자에게 배포하거나 별도 라이선스를 부여할 경우 원본 코드의 권한을 확인한다.

식단 데이터 출처는 [한양대학교 ERICA 장학복지회](https://life.hanyang.ac.kr) 공개
시설 상세 페이지이다. 로그인 없이 제공되는 HTML에서 요청한 날짜의 메뉴를
확인한다. 공개 열람 가능 여부가 재배포 권한을 뜻하지는 않으므로 과제 실습에
필요한 텍스트와 출처만 보관하고, 서비스 공개 전 이용 조건 및 수집 정책을
확인한다. 사진 파일은 전체 수집을 요청한 사용자 지시에 따라 공식 출처에서 다운로드하고 원문 URL·해시·수집 시각을 보존한다. 학교 안내를 위한 새 포털·자산 수집 코드와 기존 학식 파서의 출처는 구분한다.
