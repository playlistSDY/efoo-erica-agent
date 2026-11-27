# EFOO 3주차 AWS MVP

2주차 ECR, ECS, ALB와 CloudFront를 재사용합니다. 이 폴더의 템플릿 생성기는 기존 스택에서 논리 리소스 이름을 유지합니다. 상태가 정상인 에푸 전용 스택 3개를 확인한 뒤 같은 계정과 리전에서만 실행합니다.

1. 로컬에서 공개 자료를 현재 인덱스에 적재한 뒤 프로젝트 루트에서 `python deploy/aws-mvp/prepare.py /별도경로/efoo-mvp.zip`을 실행합니다.
2. ZIP을 CloudShell에 올려 새 폴더에 풉니다. `bash deploy.sh <40자리 Git SHA> image`는 ECR 이미지까지 빌드합니다.
3. 서울 Secrets Manager의 `efoo-week3/openai-api-key`에는 OpenAI 키 원문을 SecretString으로 보관합니다. 이미지·Git·태스크 환경 변수 선언에는 키를 넣지 않습니다.
4. `bash deploy.sh <40자리 Git SHA>`가 에푸 런타임을 갱신하고 CloudFront 주소를 출력합니다. 브라우저에서 질문을 전송하고 출처와 새로고침 후 대화 복원을 확인합니다.

배포용 SQLite 시드는 공개 컬렉션·자료·임베딩만 포함합니다. 단일 Fargate 작업의 로컬 DB를 사용하므로 작업 교체 시 새 대화와 새 수집 캐시가 초기화됩니다. 초기 자료는 이미지에서 복원합니다. 실습 클라우드는 요청 시 자료를 갱신하고 에이전트는 최대 4회 모델 응답을 사용합니다. 백그라운드 정기 수집은 로컬에서 운영합니다.

브라우저 구간은 HTTPS, CloudFront에서 ALB까지는 HTTP입니다. ALB는 CloudFront origin-facing 주소 범위만 허용합니다. CloudFront는 POST, 세션 쿠키, 쿼리 문자열을 전달하고 응답을 캐시하지 않습니다. 실습 서버 전체에 시간당 30개 질문과 동시 2개 제한이 적용됩니다. 질문을 포함한 채팅 데이터는 테스트용으로 취급합니다.
