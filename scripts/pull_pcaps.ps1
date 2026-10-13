# 폰(PCAPdroid)에서 새 pcap만 받아 와서 수집 현황을 보여 준다.
#   준비: Android SDK Platform-Tools(adb) 설치, 폰에서 USB 디버깅 켜기, USB 연결
#   실행: .\scripts\pull_pcaps.ps1
#         .\scripts\pull_pcaps.ps1 -PhoneDir /sdcard/Download/PCAPdroid   # 폰 쪽 폴더가 다르면
param(
    [string]$PhoneDir = "/sdcard/Download/PCAPdroid",
    [string]$Dest = $(if ($env:CIPHERLENS_DATA) { "$env:CIPHERLENS_DATA/self_collected" } else { "D:/datasets/self_collected" })
)

if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
    Write-Error "adb를 찾을 수 없다. Android SDK Platform-Tools를 설치하고 PATH에 추가한다."
    exit 1
}

New-Item -ItemType Directory -Force $Dest | Out-Null
$onPhone = adb shell ls "$PhoneDir" | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '\.pcap(ng)?$' }
$new = $onPhone | Where-Object { -not (Test-Path (Join-Path $Dest $_)) }

foreach ($name in $new) {
    adb pull "$PhoneDir/$name" (Join-Path $Dest $name) | Out-Null
    Write-Output "받음: $name"
}
Write-Output "새 파일 $(@($new).Count)개 (폰에 있는 pcap $(@($onPhone).Count)개)`n"

Push-Location "$PSScriptRoot/.."
.\.venv\Scripts\python.exe -m src.preprocess --status --folder $Dest
Pop-Location
