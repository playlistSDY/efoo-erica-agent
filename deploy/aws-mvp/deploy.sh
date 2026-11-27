#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export AWS_PAGER="" AWS_REGION=ap-northeast-2 AWS_DEFAULT_REGION=ap-northeast-2
account=496393363424
project=efoo-week2
[[ "$(aws sts get-caller-identity --query Account --output text)" == "$account" ]]
[[ "${1:-}" =~ ^[0-9a-f]{40}$ ]] || { echo 'Pass the source commit SHA.' >&2; exit 2; }
commit=$1
stage=${2:-deploy}
[[ "$stage" == deploy || "$stage" == image ]] || exit 2

# Validate all existing stacks before touching the registry or service.
python3 - <<'PY'
import json, subprocess
for part in ("bootstrap", "runtime", "frontend"):
    raw = subprocess.check_output(["aws", "cloudformation", "describe-stacks", "--stack-name",
                                  "efoo-week2-" + part, "--output", "json"])
    stack = json.loads(raw)["Stacks"][0]
    tags = {item["Key"]:item["Value"] for item in stack.get("Tags", [])}
    if tags.get("Project") != "EFOO" or tags.get("Stage") != "Week2":
        raise SystemExit("Stack ownership mismatch: " + part)
    if stack["StackStatus"] not in ("CREATE_COMPLETE", "UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE"):
        raise SystemExit("Stack is not ready: " + part)
    if part == "runtime":
        params = {item["ParameterKey"]:item["ParameterValue"] for item in stack["Parameters"]}
        if params["TaskPlacement"] != "Public" or any(params.get(name) for name in
            ("ExistingExecutionRoleArn", "ExistingTaskRoleArn")):
            raise SystemExit("This lab update requires the dedicated Week 2 roles and Public placement.")
PY

bucket=$(aws cloudformation describe-stacks --stack-name "$project-bootstrap" --query "Stacks[0].Outputs[?OutputKey=='DocumentBucketName'].OutputValue | [0]" --output text)
registry="$account.dkr.ecr.$AWS_REGION.amazonaws.com"
seed_digest=$(python3 -c 'import hashlib; print(hashlib.sha256(open("seed/efoo.sqlite3","rb").read()).hexdigest()[:12])')
tag="week3-${commit:0:12}-$seed_digest"
image="$registry/$project:$tag"
if ! aws ecr describe-images --repository-name "$project" --image-ids "imageTag=$tag" >/dev/null 2>&1; then
  aws ecr get-login-password | docker login --username AWS --password-stdin "$registry"
  docker build --platform linux/amd64 -t "$image" .
  docker push "$image"
  docker logout "$registry"
fi
printf 'MVP image %s\n' "$image"
if [[ "$stage" == image ]]; then exit 0; fi
secret=$(aws secretsmanager describe-secret --secret-id efoo-week3/openai-api-key --query ARN --output text)
prefix=$(aws ec2 describe-managed-prefix-lists --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing --query 'PrefixLists[0].PrefixListId' --output text)
[[ "$prefix" =~ ^pl-[a-f0-9]+$ ]]
python3 templates.py
aws cloudformation deploy --stack-name "$project-runtime" --template-file mvp-runtime.json \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
  --parameter-overrides ProjectName="$project" RepositoryName="$project" ImageUri="$image" \
  DocumentBucketName="$bucket" AppVersion="$commit" TaskPlacement=Public \
  OpenAISecretArn="$secret" CloudFrontPrefixList="$prefix"
aws ecs update-service --cluster "$project" --service "$project" --desired-count 1 --query 'service.[serviceName,desiredCount]' --output text
aws ecs wait services-stable --cluster "$project" --services "$project"
alb=$(aws cloudformation describe-stacks --stack-name "$project-runtime" --query "Stacks[0].Outputs[?OutputKey=='BaseUrl'].OutputValue | [0]" --output text)
aws cloudformation deploy --stack-name "$project-frontend" --template-file mvp-frontend.json \
  --no-fail-on-empty-changeset --parameter-overrides AlbDnsName="${alb#http://}"
aws cloudformation describe-stacks --stack-name "$project-frontend" --query "Stacks[0].Outputs[?OutputKey=='FrontendUrl'].OutputValue | [0]" --output text
