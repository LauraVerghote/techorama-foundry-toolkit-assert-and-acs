param(
    [string]$SubscriptionId = $env:AZURE_SUBSCRIPTION_ID,
    [string]$ResourceGroup = $env:AZURE_RESOURCE_GROUP,
    [string]$AccountName = $env:AZURE_AI_ACCOUNT_NAME,
    [string]$DeploymentName = $env:AZURE_AI_MODEL_DEPLOYMENT_NAME,
    [string]$PolicyName = "BankManager-DemoRelaxed"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

foreach ($value in @{
    SubscriptionId = $SubscriptionId
    ResourceGroup = $ResourceGroup
    AccountName = $AccountName
    DeploymentName = $DeploymentName
}.GetEnumerator()) {
    if ([string]::IsNullOrWhiteSpace($value.Value)) {
        throw "Missing required value: $($value.Key)"
    }
}

$apiVersion = "2024-10-01"
$accountUrl = "https://management.azure.com/subscriptions/$SubscriptionId/resourceGroups/$ResourceGroup/providers/Microsoft.CognitiveServices/accounts/$AccountName"

$contentFilters = foreach ($name in @("Hate", "Sexual", "Violence", "Selfharm")) {
    foreach ($source in @("Prompt", "Completion")) {
        @{
            name = $name
            source = $source
            enabled = $true
            blocking = $true
            severityThreshold = "High"
        }
    }
}
$contentFilters += @(
    @{
        name = "Jailbreak"
        source = "Prompt"
        enabled = $false
        blocking = $false
    },
    @{
        name = "Protected Material Text"
        source = "Completion"
        enabled = $true
        blocking = $true
    },
    @{
        name = "Protected Material Code"
        source = "Completion"
        enabled = $true
        blocking = $false
    }
)

$policyBody = @{
    properties = @{
        basePolicyName = "Microsoft.DefaultV2"
        mode = "Blocking"
        contentFilters = $contentFilters
    }
} | ConvertTo-Json -Depth 8 -Compress

function Invoke-AzRestPutJson {
    param(
        [Parameter(Mandatory)]
        [string]$Url,
        [Parameter(Mandatory)]
        [string]$Body
    )

    $bodyPath = New-TemporaryFile
    try {
        [System.IO.File]::WriteAllText(
            $bodyPath.FullName,
            $Body,
            [System.Text.UTF8Encoding]::new($false)
        )
        az rest --method PUT --url $Url --body "@$($bodyPath.FullName)" --output none
        if ($LASTEXITCODE -ne 0) {
            throw "Azure REST PUT failed for '$Url'."
        }
    }
    finally {
        Remove-Item $bodyPath.FullName -Force -ErrorAction SilentlyContinue
    }
}

Invoke-AzRestPutJson `
    -Url "$accountUrl/raiPolicies/${PolicyName}?api-version=$apiVersion" `
    -Body $policyBody

$deploymentUrl = "$accountUrl/deployments/${DeploymentName}?api-version=$apiVersion"
$deployment = az rest --method GET --url $deploymentUrl --output json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "Failed to read deployment '$DeploymentName'."
}

$deploymentBody = @{
    sku = @{
        name = $deployment.sku.name
        capacity = $deployment.sku.capacity
    }
    properties = @{
        model = @{
            format = $deployment.properties.model.format
            name = $deployment.properties.model.name
            version = $deployment.properties.model.version
        }
        raiPolicyName = $PolicyName
        versionUpgradeOption = $deployment.properties.versionUpgradeOption
    }
} | ConvertTo-Json -Depth 8 -Compress

Invoke-AzRestPutJson -Url $deploymentUrl -Body $deploymentBody

$attachedPolicy = az rest `
    --method GET `
    --url $deploymentUrl `
    --query properties.raiPolicyName `
    --output tsv
if ($LASTEXITCODE -ne 0 -or $attachedPolicy -ne $PolicyName) {
    throw "Deployment verification failed: expected '$PolicyName', found '$attachedPolicy'."
}

Write-Host "Attached RAI policy '$PolicyName' to deployment '$DeploymentName'."