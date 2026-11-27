"""Derive the MVP stack updates from the Week 2 templates."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT if (ROOT / "environment.json").exists() else ROOT.parent / "aws-test"

def read(name):
    return json.loads((BASE / name).read_text())

def write(name, template):
    (Path.cwd() / name).write_text(json.dumps(template, indent=2) + "\n")

def main():
    runtime = read("environment.json")
    runtime["Description"] = "EFOO Week 3 RAG on the Week 2 runtime"
    runtime["Parameters"].update(
        OpenAISecretArn={"Type": "String", "AllowedPattern": r"arn:aws:secretsmanager:ap-northeast-2:496393363424:secret:efoo-week3/openai-api-key-[A-Za-z0-9]+"},
        CloudFrontPrefixList={"Type": "String", "AllowedPattern": "pl-[a-f0-9]+"},
    )
    r = runtime["Resources"]
    role = r["ExecutionRole"]["Properties"]
    role["Policies"].append({
        "PolicyName": "ReadEfooModelKey",
        "PolicyDocument": {"Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": {"Ref": "OpenAISecretArn"}}
        ]},
    })
    container = r["TaskDefinition"]["Properties"]["ContainerDefinitions"][0]
    container["Environment"] = [
        {"Name": name, "Value": value} for name, value in {
            "APP_VERSION": {"Ref": "AppVersion"},
            "SECURE_SESSION_COOKIE": "true",
            "CHAT_HOURLY_LIMIT": "30",
            "AGENTIC_MAX_TURNS": "4",
            "SCHEDULE_ENABLED": "false",
        }.items()
    ]
    container["Secrets"] = [{"Name": "OPENAI_API_KEY", "ValueFrom": {"Ref": "OpenAISecretArn"}}]
    r["Alb"]["Properties"]["LoadBalancerAttributes"] = [{"Key": "idle_timeout.timeout_seconds", "Value": "75"}]
    ingress = r["AlbSecurityGroup"]["Properties"]["SecurityGroupIngress"][0]
    del ingress["CidrIp"]
    ingress["SourcePrefixListId"] = {"Ref": "CloudFrontPrefixList"}
    write("mvp-runtime.json", runtime)

    frontend = read("frontend-probe.json")
    frontend["Description"] = "EFOO Week 3 HTTPS chat using the Week 2 CDN"
    cfg = frontend["Resources"]["Frontend"]["Properties"]["DistributionConfig"]
    cfg["Comment"] = "EFOO Week 3 RAG chat"
    cfg["Origins"][0]["CustomOriginConfig"]["OriginReadTimeout"] = 60
    cfg["Origins"][0]["ConnectionAttempts"] = 1
    cfg["DefaultCacheBehavior"].update(
        AllowedMethods=["GET", "HEAD", "OPTIONS", "PUT", "PATCH", "POST", "DELETE"],
        OriginRequestPolicyId={"Ref": "ChatOriginPolicy"},
    )
    frontend["Resources"]["ChatOriginPolicy"] = {
        "Type": "AWS::CloudFront::OriginRequestPolicy",
        "Properties": {"OriginRequestPolicyConfig": {
            "Name": "efoo-week3-chat-cookies",
            "CookiesConfig": {"CookieBehavior": "whitelist", "Cookies": ["efoo_session"]},
            "HeadersConfig": {"HeaderBehavior": "whitelist", "Headers": ["Content-Type"]},
            "QueryStringsConfig": {"QueryStringBehavior": "all"},
        }},
    }
    write("mvp-frontend.json", frontend)


if __name__ == "__main__":
    main()
