"""Read-only post-deployment evidence capture. Invoked explicitly by deploy.sh."""
import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def aws(*arguments):
    process = subprocess.run(["aws", *arguments, "--output", "json"], capture_output=True, text=True, check=True)
    return json.loads(process.stdout)


def request(base, path):
    with urllib.request.urlopen(base + path, timeout=15) as response:
        body = response.read().decode()
        return {"http_status": response.status, "content_type": response.headers.get("Content-Type"), "body": json.loads(body) if "json" in response.headers.get("Content-Type", "") else body}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", required=True)
    parser.add_argument("--output", default="verification.json")
    args = parser.parse_args()
    stack = aws("cloudformation", "describe-stacks", "--stack-name", args.stack)["Stacks"][0]
    outputs = {item["OutputKey"]: item["OutputValue"] for item in stack["Outputs"]}
    identity = aws("sts", "get-caller-identity")
    evidence = {"project": "EFOO", "stage": "week2", "checked_at": datetime.now(timezone.utc).isoformat(), "identity": identity, "stack_id": stack["StackId"], "outputs": outputs, "checks": {}, "status": "pending", "rag_enabled": False}
    target = Path(args.output)
    try:
        for attempt in range(12):
            try:
                evidence["healthz"] = request(outputs["BaseUrl"], "/healthz")
                evidence["environment"] = request(outputs["BaseUrl"], "/api/environment")
                evidence["ui"] = request(outputs["BaseUrl"], "/")
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 11:
                    raise
                time.sleep(5)
        service = aws("ecs", "describe-services", "--cluster", outputs["ClusterName"], "--services", outputs["ServiceName"])["services"][0]
        targets = aws("elbv2", "describe-target-health", "--target-group-arn", outputs["TargetGroupArn"])["TargetHealthDescriptions"]
        public_access = aws("s3api", "get-public-access-block", "--bucket", outputs["DocumentBucketName"])["PublicAccessBlockConfiguration"]
        encryption = aws("s3api", "get-bucket-encryption", "--bucket", outputs["DocumentBucketName"])
        object_info = aws("s3api", "head-object", "--bucket", outputs["DocumentBucketName"], "--key", "samples/manifest.json")
        streams = aws("logs", "describe-log-streams", "--log-group-name", outputs["LogGroupName"], "--order-by", "LastEventTime", "--descending", "--limit", "5")
        groups = aws("ec2", "describe-security-groups", "--group-ids", outputs["TaskSecurityGroupId"], outputs["AlbSecurityGroupId"])
        policy = aws("s3api", "get-bucket-policy", "--bucket", outputs["DocumentBucketName"])
        evidence.update(service=service, target_health=targets, s3_public_access_block=public_access, s3_encryption=encryption, s3_manifest_object=object_info, log_streams=streams, security_groups=groups, s3_bucket_policy=json.loads(policy["Policy"]))
        ingress = next(group["IpPermissions"] for group in groups["SecurityGroups"] if group["GroupId"] == outputs["TaskSecurityGroupId"])
        task_ingress_restricted = len(ingress) == 1 and ingress[0].get("FromPort") == 8000 and ingress[0].get("ToPort") == 8000 and not ingress[0].get("IpRanges") and not ingress[0].get("Ipv6Ranges") and len(ingress[0].get("UserIdGroupPairs", [])) == 1 and ingress[0]["UserIdGroupPairs"][0]["GroupId"] == outputs["AlbSecurityGroupId"]
        response = evidence["environment"]["body"]
        evidence["checks"] = {
            "process_health": evidence["healthz"]["body"].get("status") == "ok",
            "s3_actual_read": response.get("storage", {}).get("read_ok") is True,
            "correct_project": response.get("project") == "EFOO",
            "correct_build_version": response.get("version") == outputs["AppVersion"],
            "no_false_rag_claim": response.get("rag_enabled") is False,
            "ui_loads": evidence["ui"]["http_status"] == 200 and "RAG 미연결" in evidence["ui"]["body"],
            "ecs_running": service.get("runningCount", 0) >= 1 and service.get("pendingCount", 0) == 0,
            "alb_targets_healthy": bool(targets) and all(item["TargetHealth"]["State"] == "healthy" for item in targets),
            "s3_blocks_public_access": all(public_access.get(name) is True for name in ("BlockPublicAcls", "BlockPublicPolicy", "IgnorePublicAcls", "RestrictPublicBuckets")),
            "s3_encrypted": any(item["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"] == "AES256" for item in encryption["ServerSideEncryptionConfiguration"]["Rules"]),
            "logs_created": bool(streams.get("logStreams")),
            "task_ingress_only_from_alb": task_ingress_restricted,
        }
        evidence["status"] = "passed" if all(evidence["checks"].values()) else "failed"
    except Exception as error:
        evidence["status"] = "failed"
        evidence["error_type"] = type(error).__name__
        raise
    finally:
        target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str) + "\n")
        print(json.dumps({"status": evidence["status"], "evidence_file": str(target), "base_url": outputs["BaseUrl"], "checks": evidence["checks"]}, ensure_ascii=False, indent=2))
    if evidence["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
