# Join a Windows + WSL2 machine to a nocturne run: Windows checks and fixes, then scripts/15_join.sh inside WSL.
# Usage: powershell -ExecutionPolicy Bypass -File 15_join.ps1 [-Key "<pubkey>"] [-Coordinator <ip>] [-Install] [-Check] [-Distro Ubuntu]   (docs/JOIN.md)

param(
  [string]$Key = "",
  [string]$Coordinator = "",
  [switch]$Install,
  [switch]$Check,
  [string]$Distro = "Ubuntu",
  [string]$RawBase = "https://raw.githubusercontent.com/OhadRose20596/nocturne/main/scripts"
)
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$VmId = '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
$Ports = @('2222', '50070-50079')

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin -and -not $Check) {
  Write-Host "Restarting as administrator (firewall and network profile need it)..."
  $a = @('-NoExit', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Distro', $Distro, '-RawBase', $RawBase)
  if ($Key) { $a += @('-Key', "`"$Key`"") }
  if ($Coordinator) { $a += @('-Coordinator', $Coordinator) }
  if ($Install) { $a += '-Install' }
  Start-Process powershell -Verb RunAs -ArgumentList $a
  exit
}

$work = Join-Path $env:TEMP "nocturne_join"
New-Item -ItemType Directory -Force -Path $work | Out-Null
$extra = Join-Path $work "win_facts.txt"
Set-Content -Path $extra -Value "" -Encoding ascii
function Fact($k, $v) { Add-Content -Path $extra -Value "fact $k=$v" -Encoding ascii }
function Chk($name, $status, $msg) {
  Add-Content -Path $extra -Value "check $name $status $msg" -Encoding ascii
  Write-Host ("  [{0,-5}] {1,-14} {2}" -f $status, $name, $msg)
}
function Try-Fix($name, $okMsg, [scriptblock]$fix) {
  if ($Check) { Chk $name FAIL "not set (check mode, nothing changed)"; return }
  try { & $fix; Chk $name FIXED $okMsg } catch { Chk $name FAIL $_.Exception.Message }
}

Write-Host "nocturne join: $env:COMPUTERNAME (Windows + WSL2)"
Fact win_host $env:COMPUTERNAME

# ---- network: address and profile
$cfg = Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' -and $_.InterfaceAlias -notmatch 'vEthernet|WSL|VPN|PANGP' } | Select-Object -First 1
$winIp = ""
if ($cfg) {
  $winIp = $cfg.IPv4Address[0].IPAddress
  $alias = $cfg.InterfaceAlias
  Fact win_ip $winIp; Fact win_if $alias; Fact win_mac $cfg.NetAdapter.MacAddress
  Chk win_network PASS "address $winIp on $alias (MAC $($cfg.NetAdapter.MacAddress), use it for a DHCP reservation)"
  $prof = Get-NetConnectionProfile -InterfaceAlias $alias
  Fact win_ssid $prof.Name
  if ($prof.NetworkCategory -eq 'Private') { Chk win_profile PASS "network '$($prof.Name)' is Private" }
  else { Try-Fix win_profile "network '$($prof.Name)' set to Private" { Set-NetConnectionProfile -InterfaceAlias $alias -NetworkCategory Private -ErrorAction Stop } }
} else { Chk win_network FAIL "no connected adapter with a default gateway" }

$vpn = Get-NetAdapter | Where-Object { $_.Status -eq 'Up' -and $_.InterfaceDescription -match 'PANGP|GlobalProtect|AnyConnect|VPN|TAP-Windows|WireGuard' }
if ($vpn) { Chk win_vpn FAIL "VPN adapter is up: $($vpn.InterfaceDescription -join ', '). Disconnect it by hand" } else { Chk win_vpn PASS "no VPN adapter up" }

# ---- WSL mirrored networking
$wslcfg = Join-Path $env:USERPROFILE ".wslconfig"
$needShutdown = $false
$txt = if (Test-Path $wslcfg) { Get-Content $wslcfg -Raw } else { "" }
if ($txt -match '(?im)^\s*networkingMode\s*=\s*mirrored') { Chk wsl_config PASS "networkingMode=mirrored" }
else {
  Try-Fix wsl_config "networkingMode=mirrored written to .wslconfig" {
    if ($txt -match '(?im)^\s*networkingMode\s*=') { $new = $txt -replace '(?im)^\s*networkingMode\s*=.*$', 'networkingMode=mirrored' }
    elseif ($txt -match '(?im)^\[wsl2\]') { $new = $txt -replace '(?im)^\[wsl2\]', "[wsl2]`r`nnetworkingMode=mirrored" }
    else { $new = $txt + "`r`n[wsl2]`r`nnetworkingMode=mirrored`r`n" }
    if (Test-Path $wslcfg) { Copy-Item $wslcfg "$wslcfg.bak_join" -Force }
    [IO.File]::WriteAllText($wslcfg, $new)
    $script:needShutdown = $true
  }
}

# ---- firewall
if (Get-NetFirewallRule -DisplayName nocturne -ErrorAction SilentlyContinue) { Chk win_firewall PASS "rule 'nocturne' exists" }
else { Try-Fix win_firewall "inbound TCP 2222,50070-50079 allowed" { New-NetFirewallRule -DisplayName "nocturne" -Direction Inbound -Protocol TCP -LocalPort $Ports -Action Allow -ErrorAction Stop | Out-Null } }
$hv = $null
try { $hv = Get-NetFirewallHyperVRule -Name nocturne-wsl -ErrorAction Stop } catch {}
if ($hv) { Chk win_fw_hyperv PASS "Hyper-V rule 'nocturne-wsl' exists" }
else { Try-Fix win_fw_hyperv "Hyper-V inbound rule added for WSL" { New-NetFirewallHyperVRule -Name nocturne-wsl -DisplayName "nocturne WSL" -Direction Inbound -VMCreatorId $VmId -Protocol TCP -LocalPorts $Ports -ErrorAction Stop | Out-Null } }

# ---- power
if (-not $Check) {
  powercfg /change standby-timeout-ac 0; powercfg /change hibernate-timeout-ac 0; powercfg /change monitor-timeout-ac 0
  Chk win_power FIXED "no sleep, hibernate or display-off on AC power"
}
$batt = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($batt -and $batt.BatteryStatus -eq 1) { Chk win_ac FAIL "running on battery. Plug in the charger" } else { Chk win_ac PASS "on AC power" }

# ---- GPU driver
$gpu = $null
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null | Select-Object -First 1) }
if ($gpu) { Fact win_gpu $gpu; Chk win_gpu PASS $gpu } else { Chk win_gpu FAIL "nvidia-smi not found or no GPU. Install the NVIDIA Windows driver" }

# ---- WSL
$distros = @()
if (Get-Command wsl -ErrorAction SilentlyContinue) { $distros = @((wsl -l -q) -replace "`0", "" | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
if ($distros -notcontains $Distro) {
  $u = $distros | Where-Object { $_ -match '^Ubuntu' } | Select-Object -First 1
  if ($u) { $Distro = $u }
}
if ($distros -notcontains $Distro) {
  Chk wsl_distro FAIL "no Ubuntu WSL distro found (have: $($distros -join ', ')). This tool needs WSL2: wsl --install -d Ubuntu, reboot, run again. Windows without WSL is not supported"
  Copy-Item $extra (Join-Path ([Environment]::GetFolderPath('Desktop')) "nocturne_join_$env:COMPUTERNAME.txt")
  Write-Host "Report (Windows part only) is on the Desktop."; exit 1
}
Chk wsl_distro PASS "$Distro"
if ($needShutdown) { Write-Host "  restarting WSL to apply mirrored networking. This stops every running WSL distro (old .wslconfig kept as .wslconfig.bak_join)"; wsl --shutdown; Start-Sleep 3 }

# keep WSL running with no Ubuntu window open
$alive = Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" | Where-Object { $_.CommandLine -match 'sleep infinity' }
if ($alive) { Chk wsl_keepalive PASS "background 'sleep infinity' keeps WSL up" }
elseif ($Check) { Chk wsl_keepalive FAIL "no keep-alive process. WSL stops when the last Ubuntu window closes" }
else { Start-Process wsl -ArgumentList '-d', $Distro, '--exec', 'sleep', 'infinity' -WindowStyle Hidden; Chk wsl_keepalive FIXED "started a hidden 'sleep infinity' that keeps WSL up until reboot" }

# ---- Linux side
$sh = Join-Path $PSScriptRoot "15_join.sh"
if (-not (Test-Path $sh)) {
  $sh = Join-Path $work "15_join.sh"
  try { Invoke-WebRequest -UseBasicParsing "$RawBase/15_join.sh" -OutFile $sh -ErrorAction Stop }
  catch { Write-Host "Could not download 15_join.sh: $($_.Exception.Message)"; Write-Host "Nothing was run inside WSL. No report was written."; exit 1 }
}
function WslPath($p) { (wsl -d $Distro -- wslpath -a ($p -replace '\\', '/')).Trim() }
$shW = WslPath $sh; $extraW = WslPath $extra
$args2 = "--extra '$extraW'"
if ($Key) { $kf = Join-Path $work "key.pub"; Set-Content -Path $kf -Value $Key -Encoding ascii; $args2 += " --key-file '$(WslPath $kf)'" }
if ($Coordinator) { $args2 += " --coordinator $Coordinator" }
if ($Install) { $args2 += " --install" }
if ($Check) { $args2 += " --check" }
wsl -d $Distro -- bash -lc "tr -d '\r' < '$shW' > ~/.nocturne_join.sh && NOCTURNE_WIN_IP=$winIp bash ~/.nocturne_join.sh $args2; test -s ~/nocturne_join_`$(hostname -s).txt"
if ($LASTEXITCODE -ne 0) {
  Write-Host "The Linux part did not write a report (exit $LASTEXITCODE). Copy the text of this window and send it instead."
  Copy-Item $extra (Join-Path ([Environment]::GetFolderPath('Desktop')) "nocturne_join_$env:COMPUTERNAME.txt")
  exit 1
}

$out = Join-Path ([Environment]::GetFolderPath('Desktop')) "nocturne_join_$env:COMPUTERNAME.txt"
wsl -d $Distro -- bash -lc 'cat ~/nocturne_join_$(hostname -s).txt' | Out-File -FilePath $out -Encoding utf8
Write-Host ""
Write-Host "Report copied to: $out"
Write-Host "If it was not sent automatically, mail this file to the coordinator machine."
