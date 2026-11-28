# Clock sync of a Windows host that runs a WSL2 peer: show the time service status; with -Set, point it at one server and resync (administrator).
# Usage: powershell -ExecutionPolicy Bypass -File 17_timesync.ps1 [-Set] [-Server time.apple.com]   (then run 17_timesync.sh --set inside WSL)

param([switch]$Set, [string]$Server = "time.apple.com")

if ($Set) {
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) { Write-Error "-Set needs an administrator PowerShell"; exit 2 }
    Start-Service w32time -ErrorAction SilentlyContinue
    w32tm /config /manualpeerlist:"$Server" /syncfromflags:manual /update | Out-Host
    w32tm /resync /force | Out-Host
}
w32tm /query /status | Out-Host
w32tm /stripchart /computer:$Server /samples:4 /dataonly | Out-Host
$wsl = (wsl.exe -e date +%s.%N 2>$null)
if ($wsl) {
    $win = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    "{0}: WSL clock minus Windows clock = {1:N3} s (includes the wsl.exe start time)" -f $env:COMPUTERNAME, ([double]$wsl - $win) | Out-Host
}
