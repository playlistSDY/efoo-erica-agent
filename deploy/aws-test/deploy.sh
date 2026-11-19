#!/usr/bin/env bash
# Reviewed execution only. Creates dedicated EFOO resources; performs no deletes.
set -Eeuo pipefail
cd "$(dirname "$0")"
export AWS_PAGER=""
export AWS_REGION="${AWS_REGION:-ap-northeast-2}"
export AWS_DEFAULT_REGION="$AWS_REGION"
EXPECTED_ACCOUNT_ID="${EXPECTED_ACCOUNT_ID:-496393363424}"
PROJECT_NAME="${PROJECT_NAME:-efoo-week2}"
TASK_PLACEMENT="${TASK_PLACEMENT:-Public}"
EXISTING_EXECUTION_ROLE_ARN="${EXISTING_EXECUTION_ROLE_ARN:-}"
EXISTING_TASK_ROLE_ARN="${EXISTING_TASK_ROLE_ARN:-}"
[[ "$PROJECT_NAME" =~ ^efoo[a-z0-9-]{0,20}$ ]] || { echo 'PROJECT_NAME must begin with efoo and contain at most 24 lowercase letters, digits or hyphens.' >&2; exit 2; }
[[ "$TASK_PLACEMENT" == Public || "$TASK_PLACEMENT" == Private ]] || { echo 'TASK_PLACEMENT must be Public or Private.' >&2; exit 2; }
[[ "$AWS_REGION" == ap-northeast-2 ]] || { echo 'This reviewed draft targets Seoul only.' >&2; exit 2; }
mkdir -p evidence
STAGE=preflight
export PROJECT_NAME TASK_PLACEMENT
trap 'FAILED_EXIT=$?; export STAGE FAILED_EXIT; python3 -c '\''import json,os; from pathlib import Path; Path("evidence/deploy-failure.json").write_text(json.dumps({"project":"EFOO","stage":os.environ["STAGE"],"exit_code":int(os.environ["FAILED_EXIT"])}))'\''; echo "Deployment stopped at stage: $STAGE. Inspect CloudFormation events before retrying." >&2; exit "$FAILED_EXIT"' ERR
command -v aws >/dev/null
command -v docker >/dev/null
command -v python3 >/dev/null
docker info >/dev/null
aws sts get-caller-identity --output json > evidence/identity.json
ACTUAL_ACCOUNT_ID=$(python3 -c 'import json; print(json.load(open("evidence/identity.json"))["Account"])')
[[ "$ACTUAL_ACCOUNT_ID" == "$EXPECTED_ACCOUNT_ID" ]] || { echo "Account mismatch: expected $EXPECTED_ACCOUNT_ID, got $ACTUAL_ACCOUNT_ID" >&2; exit 2; }
export ACTUAL_ACCOUNT_ID
BOOTSTRAP_STACK="$PROJECT_NAME-bootstrap"
RUNTIME_STACK="$PROJECT_NAME-runtime"
export BOOTSTRAP_STACK RUNTIME_STACK
# Refuse to update an unrelated existing stack with a matching name.
python3 - <<'PY'
import json, os, subprocess
for name in (os.environ['BOOTSTRAP_STACK'], os.environ['RUNTIME_STACK']):
    proc=subprocess.run(['aws','cloudformation','describe-stacks','--stack-name',name,'--output','json'],capture_output=True,text=True)
    if proc.returncode:
        if 'does not exist' not in proc.stderr:
            raise SystemExit(proc.stderr)
        continue
    stack=json.loads(proc.stdout)['Stacks'][0]
    tags={entry['Key']:entry['Value'] for entry in stack.get('Tags',[])}
    if tags.get('Project')!='EFOO' or tags.get('Stage')!='Week2':
        raise SystemExit('Refusing to update an unowned stack: '+name)
    if stack['StackStatus'] in ('ROLLBACK_COMPLETE','ROLLBACK_IN_PROGRESS','DELETE_IN_PROGRESS'):
        raise SystemExit('Existing stack requires explicit review before retry: '+name+' '+stack['StackStatus'])
PY
python3 - <<'PY'
import hashlib, pathlib
digest=hashlib.sha256()
for name in ('app.py','index.html','requirements.txt','Dockerfile','data/manifest.json'):
    digest.update(name.encode());digest.update(pathlib.Path(name).read_bytes())
pathlib.Path('evidence/build-fingerprint.txt').write_text('sha256:'+digest.hexdigest()+'\n')
PY
APP_VERSION="${APP_VERSION:-$(cat evidence/build-fingerprint.txt)}"
export APP_VERSION
STAGE=bootstrap
aws cloudformation validate-template --template-body file://bootstrap.json --output json > evidence/bootstrap-template-validation.json
aws cloudformation validate-template --template-body file://environment.json --output json > evidence/runtime-template-validation.json
aws cloudformation deploy --stack-name "$BOOTSTRAP_STACK" --template-file bootstrap.json --parameter-overrides "ProjectName=$PROJECT_NAME" --tags Project=EFOO Stage=Week2 --no-fail-on-empty-changeset
aws cloudformation describe-stacks --stack-name "$BOOTSTRAP_STACK" --output json > evidence/bootstrap-stack.json
REPOSITORY_URI=$(python3 -c 'import json; print(next(x["OutputValue"] for x in json.load(open("evidence/bootstrap-stack.json"))["Stacks"][0]["Outputs"] if x["OutputKey"]=="RepositoryUri"))')
DOCUMENT_BUCKET=$(python3 -c 'import json; print(next(x["OutputValue"] for x in json.load(open("evidence/bootstrap-stack.json"))["Stacks"][0]["Outputs"] if x["OutputKey"]=="DocumentBucketName"))')
STAGE=sample-upload
aws s3api put-object --bucket "$DOCUMENT_BUCKET" --key samples/manifest.json --body data/manifest.json --content-type application/json --server-side-encryption AES256 --output json > evidence/manifest-upload.json
STAGE=image-build
IMAGE_TAG="week2-$(date -u +%Y%m%dT%H%M%SZ)"
IMAGE_URI="$REPOSITORY_URI:$IMAGE_TAG"
export IMAGE_URI
# Credentials stay inside CloudShell and Docker's registry login; never export them.
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${REPOSITORY_URI%%/*}"
docker build --platform linux/amd64 --tag "$IMAGE_URI" .
docker push "$IMAGE_URI"
aws ecr describe-images --repository-name "$PROJECT_NAME" --image-ids "imageTag=$IMAGE_TAG" --output json > evidence/ecr-image.json
STAGE=runtime
aws cloudformation deploy --stack-name "$RUNTIME_STACK" --template-file environment.json --capabilities CAPABILITY_IAM --parameter-overrides "ProjectName=$PROJECT_NAME" "RepositoryName=$PROJECT_NAME" "ImageUri=$IMAGE_URI" "DocumentBucketName=$DOCUMENT_BUCKET" "AppVersion=$APP_VERSION" "TaskPlacement=$TASK_PLACEMENT" "ExistingExecutionRoleArn=$EXISTING_EXECUTION_ROLE_ARN" "ExistingTaskRoleArn=$EXISTING_TASK_ROLE_ARN" --tags Project=EFOO Stage=Week2 --no-fail-on-empty-changeset
STAGE=verification
aws ecs wait services-stable --cluster "$PROJECT_NAME" --services "$PROJECT_NAME"
python3 verify_deployment.py --stack "$RUNTIME_STACK" --output evidence/verification.json
STAGE=complete
trap - ERR
echo 'EFOO Week2 environment verified. See evidence/verification.json for the live URL and checks.'
