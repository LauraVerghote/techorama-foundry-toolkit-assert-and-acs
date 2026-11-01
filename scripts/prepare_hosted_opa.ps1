param(
    [string]$Version = "1.8.0"
)

$ErrorActionPreference = "Stop"

if ($Version -ne "1.8.0") {
    throw "No trusted checksum is configured for OPA $Version."
}

$expectedSha256 = "1359b1bff233fc0a290066e864c75b8158e52756319757b6854df467fe7fc146"
$download = Join-Path $env:TEMP "bank-manager-opa-$Version.download"
$uri = "https://github.com/open-policy-agent/opa/releases/download/v$Version/opa_linux_amd64_static"
$destinationDirectories = @(
    (Join-Path $PSScriptRoot "..\bank_manager\agents\assert_optimized_instructions_acs\bin"),
    (Join-Path $PSScriptRoot "..\bank_manager\agents\cheaper_model\bin")
)

try {
    Invoke-WebRequest -Uri $uri -OutFile $download
    $actualSha256 = (Get-FileHash -Path $download -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) {
        throw "OPA checksum mismatch: expected $expectedSha256, got $actualSha256"
    }
    foreach ($destinationDirectory in $destinationDirectories) {
        New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
        Copy-Item -Force -Path $download -Destination (Join-Path $destinationDirectory "opa")
    }
}
finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $download
}

Write-Host "Prepared OPA $Version in both standalone controlled agent packages."