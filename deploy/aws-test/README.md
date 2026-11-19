# EFOO 2주차 AWS 환경 검증 앱

이 폴더는 **EFOO 전용 신규 환경 검증 코드**입니다. 다른 `erica-rag-week2` 프로젝트나 원본 Kakao 저장소를 사용하거나 변경하지 않습니다. 실제 EFOO 챗봇 전체를 배포한 결과물은 아닙니다. 2주차의 “간단한 테스트 앱/API AWS 배포” 범위에서 컨테이너 실행, S3 접근 권한, 로그 연결을 검증합니다.

## 실행 범위

- `GET /`: 실제 `/api/environment` 응답을 표시하는 한국어 웹 화면.
- `GET /healthz`: 프로세스 상태만 확인. S3 장애가 있어도 프로세스가 살아 있으면 200.
- `GET /api/environment`: 비공개 S3의 `samples/manifest.json`을 **실제 GetObject**하여 프로젝트, 리전, 빌드 버전, ETag, 버전 ID, 출처 목록 표시. 읽기/검증 실패는 **503**. 샘플 파일을 자동으로 대신 읽지 않습니다.
- `rag_enabled=false`: 출처 **메타데이터**만 저장하며 원문 본문·이미지·벡터 DB·임베딩·생성 모델은 연결하지 않았습니다. OpenAI 키와 대화 내용은 필요하지 않습니다.

## 배포 구성과 1주차 설계 연결

`bootstrap.json`은 ECR 저장소와 S3 버킷을 생성합니다. Fargate 서비스는 이미지가 이미 ECR에 있어야 시작하므로, 이미지 생성보다 먼저 두 번째 스택을 만들지 않습니다. 샘플도 런타임 스택 배포 전에 S3에 업로드합니다.

`environment.json`은 **새 EFOO 전용 VPC**와 두 AZ의 public subnet, internet-facing ALB, IP target group, ECS/Fargate(0.25 vCPU/512 MiB, 한 task), CloudWatch Logs(14일), execution/task 역할을 구성합니다. task ingress는 ALB SG의 TCP 8000만 허용합니다. Execution role은 해당 ECR 이미지 pull과 로그 쓰기, task role은 해당 S3 버킷의 `samples/*` GetObject만 허용합니다. S3는 public access block 4개, SSE-S3 AES256, versioning, TLS 강제 정책을 적용합니다.

| TaskPlacement | 네트워크 | 1주차 설계와 차이 |
|---|---|---|
| `Public` (기본) | Public subnet의 Fargate task에 public IP. ALB SG만 인바운드 허용. NAT 없음. | 2주차 비용 절감 실습 구성. private task+NAT 원안과 다름. public IP가 있어도 task port를 인터넷에 직접 개방하지 않음. |
| `Private` | 두 AZ private subnet, task public IP 없음. Public A의 NAT 한 개로 아웃바운드 HTTPS. | private Fargate 원안에 가까움. 단일 NAT이므로 NAT 장애에 대한 다중 AZ 가용성은 구현하지 않음. |

ALB는 현재 **HTTP 80 테스트 URL**입니다. HTTPS/인증/CloudFront, 프런트엔드 전용 S3 배포, RDS/pgvector, OpenAI 호출, 본문·이미지 S3 적재, 정기 수집은 이번 검증 앱에 포함하지 않습니다. ALB/태스크/NAT(Private 선택 시)은 실행 중 비용이 발생합니다. 템플릿에는 자동 삭제가 없고 ECR/S3에 Retain을 적용합니다.

## CloudShell에서 실행

AWS 콘솔 계정과 서울 리전을 확인한 뒤 이 폴더가 들어 있는 ZIP을 CloudShell에 업로드합니다. AWS credentials는 외부로 복사하지 않습니다. 기본 스크립트는 계정 `496393363424`, `ap-northeast-2`를 검사하고 다른 계정/리전이면 중단합니다.

```bash
unzip -q efoo-week2-aws-test.zip -d efoo-week2-upload
cd efoo-week2-upload/aws-test
bash deploy.sh
```

이미 폴더가 있다면 새 빈 디렉터리에 압축을 해제합니다. `deploy.sh`는 AWS CLI, Docker daemon, Python 3을 요구합니다. CloudShell의 현재 세션 자격으로만 실행하며, AWS 로그인 설정을 변경하거나 자격을 내보내지 않습니다. 동일 이름 기존 스택의 태그가 `Project=EFOO`, `Stage=Week2`가 아니면 변경을 거부합니다.

옵션:

```bash
# Private subnet + single NAT 실습 (추가 비용)
TASK_PLACEMENT=Private bash deploy.sh

# Git commit을 알고 있으면 명시. 생략 시 빌드 입력 파일 SHA256을 사용.
APP_VERSION=실제커밋SHA bash deploy.sh

# IAM 생성이 제한된 계정: 권한이 검증된 별도 역할만 지정
EXISTING_EXECUTION_ROLE_ARN=arn:aws:iam::496393363424:role/검증된실행역할 \
EXISTING_TASK_ROLE_ARN=arn:aws:iam::496393363424:role/검증된S3읽기역할 \
bash deploy.sh
```

역할을 지정하는 것만으로 필요한 권한이 생기지 않습니다. execution 역할의 ECR pull/로그 쓰기와 task 역할의 대상 S3 `samples/*` 읽기, 배포 사용자의 `iam:PassRole`을 실제로 검증해야 합니다. ECS/IAM 권한 부족 시 이 앱을 다른 인프라에 배포했다면 최종 보고서에서 Fargate 배포 성공으로 적지 않습니다. 실패한 스택은 자동 삭제하지 않으며 원인을 확인한 뒤 별도로 처리합니다.

배포 순서: 계정/리전/소유권 확인 → CloudFormation 문법 검증 → ECR+S3 bootstrap → S3 manifest 업로드 → `linux/amd64` 이미지 빌드·push → 런타임 스택 생성 → 서비스 안정화 → 실제 URL/API/S3 보안/ALB target/로그/빌드 버전 검증. API가 성공해야 최종 검증이 통과합니다. `/healthz`만 성공한 상태를 전체 연결 성공으로 간주하지 않습니다.

## 결과와 보고서 증거

`evidence/verification.json`은 실제 계정 ARN, 리소스 ID, URL, API 응답, 서비스 상태, ALB target health, S3 public access/encryption/policy, 로그 스트림, SG를 저장합니다. `checks` 12개가 모두 true일 때만 `status=passed`입니다. `evidence/ecr-image.json`에는 실제 이미지 digest, `build-fingerprint.txt`에는 빌드 입력 SHA256이 있습니다. 실패하면 `deploy-failure.json`에 실패 단계가 기록됩니다. 스크린샷은 이 JSON을 대체하지 않습니다.

최종 보고서에는 실제 웹 성공 화면(URL/리전/빌드버전/RAG 미연결), ECS RUNNING/ALB healthy, S3 비공개/암호화, CloudWatch 로그, EFOO GitHub 주소와 commit, 그리고 위 Public/Private 차이를 포함합니다. 클라우드 배포 및 화면 캡처는 **실제 실행 후에만** 완료로 표시합니다.

## 로컬 점검

```bash
python3 -m venv .qa-venv
.qa-venv/bin/pip install -r requirements.txt cfn-lint
.qa-venv/bin/python -m unittest -v test_app.py
.qa-venv/bin/cfn-lint bootstrap.json environment.json
bash -n deploy.sh
```

테스트는 HTTP 라우팅과 실제 GetObject 호출 여부, S3 실패/빈 설정/타 프로젝트/비공식 URL/과대·손상 JSON에 대한 503 처리를 확인합니다. AWS 호출 없이 S3 client를 mock합니다. Docker 이미지 실행과 AWS 생성 성공은 로컬 테스트로 검증할 수 없습니다.

템플릿을 변경할 때 `make_templates.py`를 수정하고 `python3 make_templates.py`로 JSON을 재생성합니다. 배포 스크립트는 재생성 없이 검토된 JSON 파일을 사용합니다. 이 스캐폴드 작성 시 Docker daemon이 꺼져 있어 실제 이미지 빌드는 미검증입니다.

공식 근거: [ECS/Fargate 시작](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/getting-started-fargate.html), [ALB의 awsvpc/IP target 요건](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/alb.html), [Public IP와 private NAT 아웃바운드 연결](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/networking-outbound.html).
