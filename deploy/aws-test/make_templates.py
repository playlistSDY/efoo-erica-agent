"""Generate reviewable CloudFormation JSON without third-party YAML tooling."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ref(name):
    return {"Ref": name}


def sub(value):
    return {"Fn::Sub": value}


def attr(name, field):
    return {"Fn::GetAtt": [name, field]}


def choose(condition, yes, no):
    return {"Fn::If": [condition, yes, no]}


def resource(kind, **properties):
    return {"Type": "AWS::" + kind, "Properties": properties}


def tags(name):
    return [{"Key": "Project", "Value": "EFOO"}, {"Key": "Stage", "Value": "Week2"}, {"Key": "Name", "Value": sub("${ProjectName}-" + name)}]


project = {"Type": "String", "Default": "efoo-week2", "MinLength": 3, "MaxLength": 24, "AllowedPattern": "[a-z][a-z0-9-]*"}
bootstrap = {
    "AWSTemplateFormatVersion": "2010-09-09",
    "Description": "EFOO Week2 bootstrap: private source metadata S3 and ECR. No model or chat data.",
    "Parameters": {"ProjectName": project},
    "Resources": {
        "Repository": resource("ECR::Repository", RepositoryName=ref("ProjectName"), ImageTagMutability="IMMUTABLE", ImageScanningConfiguration={"ScanOnPush": True}, EncryptionConfiguration={"EncryptionType": "AES256"}, EmptyOnDelete=False, Tags=tags("images")),
        "DocumentBucket": resource("S3::Bucket", BucketName=sub("${ProjectName}-sources-${AWS::AccountId}-${AWS::Region}"), PublicAccessBlockConfiguration={"BlockPublicAcls": True, "BlockPublicPolicy": True, "IgnorePublicAcls": True, "RestrictPublicBuckets": True}, BucketEncryption={"ServerSideEncryptionConfiguration": [{"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}, VersioningConfiguration={"Status": "Enabled"}, Tags=tags("sources")),
        "DocumentBucketPolicy": resource("S3::BucketPolicy", Bucket=ref("DocumentBucket"), PolicyDocument={"Version": "2012-10-17", "Statement": [{"Sid": "DenyUnencryptedTransport", "Effect": "Deny", "Principal": "*", "Action": "s3:*", "Resource": [attr("DocumentBucket", "Arn"), sub("${DocumentBucket.Arn}/*")], "Condition": {"Bool": {"aws:SecureTransport": "false"}}}]}),
    },
    "Outputs": {
        "RepositoryUri": {"Value": attr("Repository", "RepositoryUri")},
        "RepositoryName": {"Value": ref("Repository")},
        "DocumentBucketName": {"Value": ref("DocumentBucket")},
    },
}
for name in ("Repository", "DocumentBucket"):
    bootstrap["Resources"][name].update(DeletionPolicy="Retain", UpdateReplacePolicy="Retain")

parameters = {
    "ProjectName": project,
    "ImageUri": {"Type": "String", "AllowedPattern": "[0-9]{12}\\.dkr\\.ecr\\.[a-z0-9-]+\\.amazonaws\\.com(?:\\.cn)?/[a-z0-9/_-]+:[a-zA-Z0-9._-]+", "Description": "Existing image in the bootstrap ECR repository; build for linux/amd64."},
    "RepositoryName": {"Type": "String", "Default": "efoo-week2", "AllowedPattern": "[a-z0-9][a-z0-9/_-]+"},
    "DocumentBucketName": {"Type": "String", "AllowedPattern": "[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]"},
    "AppVersion": {"Type": "String", "MinLength": 1, "MaxLength": 128, "Description": "Git commit or SHA256 build fingerprint, displayed by the probe."},
    "TaskPlacement": {"Type": "String", "Default": "Public", "AllowedValues": ["Public", "Private"], "Description": "Public: no NAT, task public IP, inbound only from ALB SG. Private: private tasks via one NAT (lab, not multi-AZ NAT HA)."},
    "ExistingExecutionRoleArn": {"Type": "String", "Default": "", "AllowedPattern": "(^$|arn:aws:iam::[0-9]{12}:role/.+)", "Description": "Optional lab-provided execution role; must pull this ECR image and write this log group."},
    "ExistingTaskRoleArn": {"Type": "String", "Default": "", "AllowedPattern": "(^$|arn:aws:iam::[0-9]{12}:role/.+)", "Description": "Optional lab-provided task role; must GetObject only the bootstrap bucket samples prefix."},
}
resources = {
    "Vpc": resource("EC2::VPC", CidrBlock="10.42.0.0/16", EnableDnsSupport=True, EnableDnsHostnames=True, Tags=tags("vpc")),
    "InternetGateway": resource("EC2::InternetGateway", Tags=tags("igw")),
    "GatewayAttachment": resource("EC2::VPCGatewayAttachment", VpcId=ref("Vpc"), InternetGatewayId=ref("InternetGateway")),
    "PublicRouteTable": resource("EC2::RouteTable", VpcId=ref("Vpc"), Tags=tags("public-routes")),
    "PublicDefaultRoute": resource("EC2::Route", RouteTableId=ref("PublicRouteTable"), DestinationCidrBlock="0.0.0.0/0", GatewayId=ref("InternetGateway")),
    "AlbSecurityGroup": resource("EC2::SecurityGroup", GroupDescription="Public HTTP test endpoint; TLS termination is future work", VpcId=ref("Vpc"), SecurityGroupIngress=[{"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "CidrIp": "0.0.0.0/0"}], SecurityGroupEgress=[{"IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8000, "CidrIp": "10.42.0.0/16"}], Tags=tags("alb-sg")),
    "TaskSecurityGroup": resource("EC2::SecurityGroup", GroupDescription="Application ingress from ALB only; HTTPS egress for ECR S3 and logs", VpcId=ref("Vpc"), SecurityGroupIngress=[{"IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8000, "SourceSecurityGroupId": ref("AlbSecurityGroup")}], SecurityGroupEgress=[{"IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "CidrIp": "0.0.0.0/0"}], Tags=tags("task-sg")),
    "NatAddress": resource("EC2::EIP", Domain="vpc", Tags=tags("nat-ip")),
    "NatGateway": resource("EC2::NatGateway", AllocationId=attr("NatAddress", "AllocationId"), SubnetId=ref("PublicSubnetA"), Tags=tags("nat")),
    "Alb": resource("ElasticLoadBalancingV2::LoadBalancer", Scheme="internet-facing", Type="application", IpAddressType="ipv4", SecurityGroups=[ref("AlbSecurityGroup")], Subnets=[ref("PublicSubnetA"), ref("PublicSubnetB")], Tags=tags("alb")),
    "TargetGroup": resource("ElasticLoadBalancingV2::TargetGroup", VpcId=ref("Vpc"), Port=8000, Protocol="HTTP", TargetType="ip", HealthCheckPath="/healthz", HealthCheckProtocol="HTTP", HealthCheckIntervalSeconds=15, HealthCheckTimeoutSeconds=5, HealthyThresholdCount=2, UnhealthyThresholdCount=2, Matcher={"HttpCode": "200"}, TargetGroupAttributes=[{"Key": "deregistration_delay.timeout_seconds", "Value": "15"}], Tags=tags("targets")),
    "Listener": resource("ElasticLoadBalancingV2::Listener", LoadBalancerArn=ref("Alb"), Port=80, Protocol="HTTP", DefaultActions=[{"Type": "forward", "TargetGroupArn": ref("TargetGroup")}]),
    "Cluster": resource("ECS::Cluster", ClusterName=ref("ProjectName"), ClusterSettings=[{"Name": "containerInsights", "Value": "disabled"}], Tags=tags("cluster")),
    "LogGroup": resource("Logs::LogGroup", LogGroupName=sub("/ecs/${ProjectName}"), RetentionInDays=14, Tags=tags("logs")),
}
resources["PublicDefaultRoute"]["DependsOn"] = "GatewayAttachment"
for index, letter in enumerate(("A", "B")):
    for scope, third_octet in (("Public", index), ("Private", index + 10)):
        name = scope + "Subnet" + letter
        resources[name] = resource("EC2::Subnet", VpcId=ref("Vpc"), CidrBlock=f"10.42.{third_octet}.0/24", AvailabilityZone={"Fn::Select": [index, {"Fn::GetAZs": ""}]}, MapPublicIpOnLaunch=False, Tags=tags(scope.lower() + "-" + letter.lower()))
        table = "PublicRouteTable" if scope == "Public" else "PrivateRouteTable" + letter
        resources[name + "Routes"] = resource("EC2::SubnetRouteTableAssociation", SubnetId=ref(name), RouteTableId=ref(table))
        if scope == "Private":
            resources[name]["Condition"] = "PrivateTasks"
            resources[name + "Routes"]["Condition"] = "PrivateTasks"
            resources[table] = resource("EC2::RouteTable", VpcId=ref("Vpc"), Tags=tags("private-routes-" + letter.lower()))
            resources[table]["Condition"] = "PrivateTasks"
            resources["PrivateDefaultRoute" + letter] = resource("EC2::Route", RouteTableId=ref(table), DestinationCidrBlock="0.0.0.0/0", NatGatewayId=ref("NatGateway"))
            resources["PrivateDefaultRoute" + letter]["Condition"] = "PrivateTasks"
for name in ("NatAddress", "NatGateway"):
    resources[name]["Condition"] = "PrivateTasks"
resources["NatGateway"]["DependsOn"] = "GatewayAttachment"

assume = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": ref("AWS::AccountId")}, "ArnLike": {"aws:SourceArn": sub("arn:${AWS::Partition}:ecs:${AWS::Region}:${AWS::AccountId}:*")}}}]}
resources["ExecutionRole"] = resource("IAM::Role", AssumeRolePolicyDocument=assume, Policies=[{"PolicyName": "PullImageAndWriteLogs", "PolicyDocument": {"Version": "2012-10-17", "Statement": [
    {"Effect": "Allow", "Action": ["ecr:GetAuthorizationToken"], "Resource": "*"},
    {"Effect": "Allow", "Action": ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"], "Resource": sub("arn:${AWS::Partition}:ecr:${AWS::Region}:${AWS::AccountId}:repository/${RepositoryName}")},
    {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"], "Resource": attr("LogGroup", "Arn")},
]}}], Tags=tags("execution-role"))
resources["ExecutionRole"]["Condition"] = "CreateExecutionRole"
resources["TaskRole"] = resource("IAM::Role", AssumeRolePolicyDocument=assume, Policies=[{"PolicyName": "ReadSourceMetadata", "PolicyDocument": {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": sub("arn:${AWS::Partition}:s3:::${DocumentBucketName}/samples/*")}]}}], Tags=tags("task-role"))
resources["TaskRole"]["Condition"] = "CreateTaskRole"
task_role = choose("CreateTaskRole", attr("TaskRole", "Arn"), ref("ExistingTaskRoleArn"))
execution_role = choose("CreateExecutionRole", attr("ExecutionRole", "Arn"), ref("ExistingExecutionRoleArn"))
resources["TaskDefinition"] = resource("ECS::TaskDefinition", Family=ref("ProjectName"), RequiresCompatibilities=["FARGATE"], NetworkMode="awsvpc", Cpu="256", Memory="512", RuntimePlatform={"CpuArchitecture": "X86_64", "OperatingSystemFamily": "LINUX"}, ExecutionRoleArn=execution_role, TaskRoleArn=task_role, ContainerDefinitions=[{
    "Name": "environment", "Image": ref("ImageUri"), "Essential": True,
    "PortMappings": [{"ContainerPort": 8000, "Protocol": "tcp"}],
    "Environment": [{"Name": "AWS_REGION", "Value": ref("AWS::Region")}, {"Name": "DOCUMENT_BUCKET", "Value": ref("DocumentBucketName")}, {"Name": "DOCUMENT_KEY", "Value": "samples/manifest.json"}, {"Name": "APP_VERSION", "Value": ref("AppVersion")}],
    "LogConfiguration": {"LogDriver": "awslogs", "Options": {"awslogs-group": ref("LogGroup"), "awslogs-region": ref("AWS::Region"), "awslogs-stream-prefix": "environment"}},
    "HealthCheck": {"Command": ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()"], "Interval": 30, "Timeout": 5, "Retries": 3, "StartPeriod": 20},
}], Tags=tags("task"))
resources["Service"] = resource("ECS::Service", Cluster=ref("Cluster"), ServiceName=ref("ProjectName"), TaskDefinition=ref("TaskDefinition"), DesiredCount=1, LaunchType="FARGATE", PlatformVersion="LATEST", EnableECSManagedTags=True, PropagateTags="SERVICE", HealthCheckGracePeriodSeconds=60, DeploymentConfiguration={"MinimumHealthyPercent": 100, "MaximumPercent": 200, "DeploymentCircuitBreaker": {"Enable": True, "Rollback": True}}, NetworkConfiguration={"AwsvpcConfiguration": {"Subnets": choose("PrivateTasks", [ref("PrivateSubnetA"), ref("PrivateSubnetB")], [ref("PublicSubnetA"), ref("PublicSubnetB")]), "SecurityGroups": [ref("TaskSecurityGroup")], "AssignPublicIp": choose("PrivateTasks", "DISABLED", "ENABLED")}}, LoadBalancers=[{"ContainerName": "environment", "ContainerPort": 8000, "TargetGroupArn": ref("TargetGroup")}], Tags=tags("service"))
resources["Service"]["DependsOn"] = ["Listener", "PublicSubnetARoutes", "PublicSubnetBRoutes"]
# This intrinsic creates a dependency on the selected egress routes, including
# the slower NAT creation in Private mode, without a conditional DependsOn.
# EC2 Route Ref contains a pipe separator, which ECS tag values disallow.
def route_table_id(route_name):
    return {"Fn::Select": [0, {"Fn::Split": ["|", ref(route_name)]}]}


resources["Service"]["Properties"]["Tags"].append({"Key": "EgressRoutes", "Value": choose("PrivateTasks", {"Fn::Join": ["/", [route_table_id("PrivateDefaultRouteA"), route_table_id("PrivateDefaultRouteB")]]}, route_table_id("PublicDefaultRoute"))})
runtime = {
    "AWSTemplateFormatVersion": "2010-09-09",
    "Description": "EFOO Week2 HTTP environment probe on dedicated VPC ALB Fargate; no RAG or model deployment.",
    "Parameters": parameters,
    "Conditions": {"PrivateTasks": {"Fn::Equals": [ref("TaskPlacement"), "Private"]}, "CreateExecutionRole": {"Fn::Equals": [ref("ExistingExecutionRoleArn"), ""]}, "CreateTaskRole": {"Fn::Equals": [ref("ExistingTaskRoleArn"), ""]}},
    "Resources": resources,
    "Outputs": {
        "BaseUrl": {"Value": sub("http://${Alb.DNSName}")}, "DocumentBucketName": {"Value": ref("DocumentBucketName")},
        "ClusterName": {"Value": ref("Cluster")}, "ServiceName": {"Value": attr("Service", "Name")},
        "TargetGroupArn": {"Value": ref("TargetGroup")}, "LogGroupName": {"Value": ref("LogGroup")},
        "TaskRoleArn": {"Value": task_role}, "ExecutionRoleArn": {"Value": execution_role},
        "TaskSecurityGroupId": {"Value": ref("TaskSecurityGroup")}, "AlbSecurityGroupId": {"Value": ref("AlbSecurityGroup")},
        "TaskPlacement": {"Value": ref("TaskPlacement")}, "AppVersion": {"Value": ref("AppVersion")},
    },
}
for filename, template in (("bootstrap.json", bootstrap), ("environment.json", runtime)):
    ROOT.joinpath(filename).write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n")
