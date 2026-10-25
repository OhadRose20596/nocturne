# קישוריות: איך מחברים שני מחשבים, מכבל ועד אינטרנט

מדריך מעשי. כל אפשרות: מתי, מה צריך, איך מגדירים, איך בודקים, ואיך מריצים את הסקריפטים שלנו דרכה. נכתב 24.10.2026. אבטחה אינה שיקול בשלב זה (החלטת הצוות). llama.cpp מזהיר "never run the RPC server on an open network". אנחנו מודעים ודוחים את הטיפול בזה. ראו ADR-009.

## 0. התשובה הקצרה: כמה מזה נדרש בכלל

לא הרבה. היעדים (O1 עד O5) וההנחיה של חן מדברים על אינטרנט ועל רשת רחבה. T1 אומר שהרשת היא אילוץ נתון ולא יעד. לכן:

| דרגה | מה | האם נדרש | למה |
|---|---|---|---|
| 1 | Wi-Fi ביתי, כמו היום | כן. הבסיס | כל התוצאות עד כה נמדדו כך. ממשיכים כך |
| 2 | אינטרנט בין דירות, בלי VPN | כן. זה O3 | זה מה שהפרויקט טוען שהוא יודע לעשות |
| 3 | Thunderbolt Bridge בין שני המק-ים | אופציונלי. אחר צהריים אחד | עונה על שאלה אבחונית אחת: כמה מתוך 90 עד 150 ms לטוקן של ההופ הוא זנב Wi-Fi וכמה פרוטוקול. שימושי גם אם TP נטען אבל איטי |
| 4 | כבל בין מק ל-PC של ליאור | לא | המחשבים בדירות שונות. הקישור האמיתי ביניהם הוא אינטרנט |
| 5 | Ethernet דרך הראוטר | לא, אלא אם Wi-Fi לא יציב ביום מדידות | מתאם אחד למק. פתרון זול לרעש |

המלצה: להמשיך Wi-Fi ואינטרנט. לקנות כבל Thunderbolt אחד (או להשתמש בקיים) ולבצע את דרגה 3 פעם אחת, אחרי ניסוי ה-TP. לא לקנות מתאמי Ethernet עד שיש סיבה.

## 1. אותה רשת ביתית (LAN)

### 1.1 Wi-Fi, ברירת המחדל

כלום להגדיר. ה-IP מהראוטר. לבדוק שהוא לא השתנה:

```bash
ipconfig getifaddr en0        # macOS
hostname -I                   # Linux / WSL2
```

סקריפטים: `LINK=wifi OVERLAY=plain LOCATION=lan`. ה-worker ללא `BIND` (זיהוי אוטומטי).

### 1.2 Thunderbolt Bridge, מק אל מק

מה צריך: כבל Thunderbolt 3/4 או USB4 (סימן ברק). כבל הטעינה של Apple לא מתאים (USB 2.0 לנתונים).

הגדרה, בשני המק-ים: System Settings, Network, Thunderbolt Bridge. אם לא מופיע: לחצן "..." ואז Add Service ואז Thunderbolt Bridge. להגדיר IP ידני: M5 `10.10.0.1`, Air `10.10.0.2`, מסכה `255.255.255.0`, בלי router.

בדיקה:

```bash
ifconfig bridge0 | grep "inet "
ping -c 5 10.10.0.2
iperf3 -s                       # on the Air
iperf3 -c 10.10.0.2 -t 10       # on the M5
```

צפוי: ping בסביבות 0.1 עד 0.3 ms, iperf3 בסביבות 10 עד 20 Gbit/s.

הרצה:

```bash
# Air (worker)
BIND=10.10.0.2 bash scripts/08_r0.sh worker MTL0
# M5 (requester)
LINK=thunderbolt OVERLAY=plain LOCATION=lan bash scripts/08_r0.sh rpc 10.10.0.2
bash scripts/10_capacity.sh split ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf 10.10.0.2 1,1 P1
```

הערה: כש-Wi-Fi ו-Bridge פעילים יחד, macOS יבחר נתיב לפי ה-IP. כל עוד פונים ל-`10.10.0.x` התעבורה עוברת בכבל. לוודא ב-`net.txt` של הריצה (ping של 0.x ms).

### 1.3 Ethernet ישיר, מק אל PC

מה צריך: מתאם USB-C ל-RJ45 למק (1GbE או 2.5GbE). כבל רשת רגיל אל שקע ה-PC. אין צורך ב-switch או בכבל crossover.

הגדרה: במק, Network, USB 10/100/1000 LAN, IP ידני `10.20.0.1/24`. ב-Windows, Settings, Network, Ethernet, Edit IP, Manual, `10.20.0.2`, mask `255.255.255.0`. ב-WSL2 ראו סעיף 2.

בדיקה: `ping 10.20.0.2`, `iperf3` כמו למעלה. צפוי: 0.3 ms, 0.9 עד 2.3 Gbit/s.

סקריפטים: `LINK=ethernet OVERLAY=plain LOCATION=lan`.

### 1.4 Ethernet דרך הראוטר

אותו מתאם למק. שני כבלים אל הראוטר. ה-IP נשאר מהראוטר (DHCP). להעדיף אם הראוטר קרוב ולא רוצים IP ידני. `LINK=ethernet OVERLAY=plain LOCATION=lan`.

### 1.5 Thunderbolt Networking, מק אל PC

רק אם ל-PC יש שקע Thunderbolt 3/4 (ברק ליד השקע, או Thunderbolt Controller ב-Device Manager). ב-desktop עם כרטיס NVIDIA בדרך כלל אין. ב-Windows צריך את Intel Thunderbolt Control Center ולאשר את המק כ-trusted. לפעמים עובד מול מק, לפעמים לא. לא להשקיע בזה. אם עובד: `LINK=thunderbolt`.

## 2. WSL2 במחשב של ליאור

ה-worker רץ בתוך WSL2. כברירת מחדל ל-WSL2 יש רשת NAT פנימית (`172.x`), שאינה נגישה מבחוץ. חובה אחת משתי הדרכים.

### 2.1 Mirrored networking (מומלץ, Windows 11 22H2 ומעלה)

קובץ `%UserProfile%\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

ואז `wsl --shutdown` ופתיחה מחדש. WSL מקבל את ה-IP של Windows עצמו. עובד גם עם Tailscale שמותקן ב-Windows.

### 2.2 Port forward ידני

ב-PowerShell כמנהל, אחרי `wsl hostname -I`:

```powershell
netsh interface portproxy add v4tov4 listenport=50052 listenaddress=0.0.0.0 connectport=50052 connectaddress=<wsl-ip>
netsh advfirewall firewall add rule name="nocturne rpc" dir=in action=allow protocol=TCP localport=50052
```

ה-IP של WSL משתנה אחרי reboot. לבדוק כל פעם.

בשני המקרים ב-WSL: `BIND=0.0.0.0 bash scripts/08_r0.sh worker CUDA0` (בתוך WSL זה בסדר; החשיפה החוצה נקבעת ב-Windows). ה-requester פונה ל-IP של Windows.

## 3. אינטרנט בלי Tailscale

המטרה: משתמש פותח את האפליקציה ומצטרף, בלי להתקין VPN ובלי שנדרוש ממנו הגדרות ראוטר. הבעיה היחידה: ה-worker יושב מאחורי ראוטר ביתי עם NAT, ולא ניתן לפתוח אליו חיבור נכנס. חמש דרכים, מהפשוטה לנוחה ביותר.

### 3.1 Port forwarding בראוטר של ה-worker

בממשק הראוטר (בדרך כלל `192.168.1.1`): Port Forwarding, TCP `50052` אל ה-IP הפנימי של ה-worker. ה-requester פונה ל-IP הציבורי של הבית.

```bash
curl -4 ifconfig.me           # public IP, on the worker side
BIND=0.0.0.0 bash scripts/08_r0.sh worker MTL0
# requester, other home
LINK=internet OVERLAY=portforward LOCATION=wan bash scripts/08_r0.sh rpc <public-ip>
```

מגבלות: ה-IP הציבורי משתנה (פתרון: DDNS, סעיף 3.2). חלק מהספקים בישראל נותנים CGNAT (כתובת `100.64.x.x` עד `100.127.x.x` בצד ה-WAN של הראוטר), ואז port forwarding לא עובד כלל. לבדוק: אם `curl ifconfig.me` שונה מה-IP ב-WAN של הראוטר, יש CGNAT. במקרה כזה: סעיף 3.4 או 3.5.

### 3.2 DDNS: שם קבוע במקום IP משתנה

DuckDNS (חינם): `nocturne-lior.duckdns.org`. הראוטר או סקריפט cron מעדכן את ה-IP. ה-requester פונה לשם. אותו `OVERLAY=portforward`.

### 3.3 UPnP: הראוטר פותח את הפורט לבד

רוב הראוטרים הביתיים מאפשרים UPnP. ה-worker יכול לבקש מהראוטר לפתוח `50052` אוטומטית, בלי שהמשתמש ייגע בראוטר. זה מה שעושים לקוחות BitTorrent.

```bash
brew install miniupnpc        # macOS; apt install miniupnpc on Linux
upnpc -r 50052 TCP            # ask the router to map 50052 to this machine
upnpc -l                      # list mappings, see the public IP
```

זה המועמד ל-worker agent של O5: בהפעלה, לנסות UPnP; אם הצליח, להכריז ל-tracker את `public-ip:port`. אם לא, ליפול לסעיף 3.4. לא פותר CGNAT.

### 3.4 Relay דרך שרת ציבורי קטן (המועמד הראשי ל-O5)

ל-tracker נדרש ממילא מארח ציבורי קטן (VPS של כ-4 אירו בחודש, או Raspberry Pi עם port forwarding בבית אחד). אותו מארח יכול לשמש גם כ-relay: ה-worker פותח חיבור יוצא אליו (תמיד מותר דרך NAT), והמארח חושף פורט ציבורי שמפנה חזרה ל-worker. אפס הגדרות אצל המשתמש. עובד גם מאחורי CGNAT.

כלי: **frp** (frps בשרת, frpc ב-worker). קל, יציב, קוד פתוח.

```ini
# frps.toml on the public host
bindPort = 7000

# frpc.toml on the worker
serverAddr = "tracker.example.org"
serverPort = 7000
[[proxies]]
name = "rpc"
type = "tcp"
localIP = "127.0.0.1"
localPort = 50052
remotePort = 60052
```

ה-worker מריץ `ggml-rpc-server` על `127.0.0.1:50052` ו-`frpc`. ה-requester פונה ל-`tracker.example.org:60052`. ה-tracker יכול להקצות `remotePort` לכל worker ולפרסם אותו ב-announce. המחיר: כל בייט עובר דרך השרת, הזמן לטוקן בערך כפול, ורוחב הפס של ה-VPS הוא התקרה. לפי T1 זה מקובל.

חלופה בלי שרת משלנו, לבדיקה מהירה: **bore** (`bore local 50052 --to bore.pub`) נותן פורט ציבורי תוך שנייה. ללא הבטחת זמינות. טוב לניסוי, לא למוצר.

סקריפטים: `LINK=internet OVERLAY=relay LOCATION=wan`.

### 3.5 מנהרת SSH הפוכה

אם יש שרת עם SSH (אותו VPS):

```bash
# worker
ssh -N -R 0.0.0.0:60052:127.0.0.1:50052 user@tracker.example.org
```

דורש `GatewayPorts yes` ב-sshd של השרת. אותו רעיון כמו frp, בלי התקנה נוספת. פחות נוח לניהול אוטומטי. `OVERLAY=relay`.

### 3.6 Cloudflare Tunnel

`cloudflared` ב-worker חושף את 50052 דרך Cloudflare בחינם. אבל TCP גולמי דרך Cloudflare דורש `cloudflared access tcp` גם בצד ה-requester. שני צדדים מתקינים כלי, אז לא נוח יותר מ-Tailscale. לא מומלץ לנו. `OVERLAY=tunnel`.

### 3.7 IPv6 ישיר

אם לשני הבתים יש IPv6 מהספק (חלק מספקי הסיבים בישראל), אפשר לפנות ישירות לכתובת ה-IPv6 של ה-worker, רק עם פתיחת firewall בראוטר. אין NAT, אין relay. שתי בדיקות לפני: `curl -6 ifconfig.co` בשני הצדדים, והאם `ggml-rpc-server` ב-SHA שלנו מאזין ל-IPv6 (`-H` עם כתובת IPv6). לא נבדק. אם עובד, זה הפתרון הכי "פתוח" שיש.

### 3.8 מה עושה Tailscale, ולמה בכל זאת כדאי להשאיר אותו כגיבוי

Tailscale פותר NAT בשני הצדדים (hole punching) ונופל ל-relay (DERP) כשלא מצליח. זה בדיוק 3.3 ועוד 3.4 אוטומטיים, בתוספת הצפנה. המחיר: כל משתתף מתקין אפליקציה ונכנס לאותה רשת. לאפליקציה לציבור זה חיכוך. לניסויים בין שתי הדירות שלכם זה הכי מהיר להקים. לכן: ניסוי ראשון באינטרנט עם Tailscale (ADR-005), ובמקביל 3.1 או 3.4 כדי להראות שעובד גם בלי. חלופות דומות: ZeroTier, NetBird, WireGuard ידני.

## 4. ההמלצה לבניית O5 (מתכנן ואפליקציה)

סולם קישוריות ב-worker agent, אוטומטי, לפי סדר:
1. אותה רשת? IP מקומי ישיר.
2. UPnP הצליח? `public-ip:port` ישיר.
3. אחרת relay דרך מארח ה-tracker (frp). תמיד עובד.

ה-tracker מקבל ב-announce את הכתובת שהתקבלה. ה-requester לא צריך לדעת איך. המתכנן מקבל את סוג הנתיב (direct או relay) כנתון, לא כיעד.

## 5. טבלת סיכום

| אפשרות | צדדים | קושי למשתמש | מהירות טיפוסית | CGNAT | סקריפטים |
|---|---|---|---|---|---|
| Wi-Fi ביתי | מק, PC | אין | 50 עד 300 Mbit/s, 10 עד 20 ms | לא רלוונטי | `wifi plain lan` |
| Thunderbolt Bridge | מק, מק | כבל אחד, IP ידני | 10 עד 20 Gbit/s, 0.2 ms | לא רלוונטי | `thunderbolt plain lan` |
| Ethernet ישיר | מק, PC | מתאם, IP ידני | 1 עד 2.5 Gbit/s | לא רלוונטי | `ethernet plain lan` |
| Port forward + DDNS | כל | הגדרת ראוטר פעם אחת | לפי העלאה של הבית, 20 עד 100 Mbit/s, 10 עד 40 ms | נכשל | `internet portforward wan` |
| UPnP | כל | אפס | כנ"ל | נכשל | `internet portforward wan` |
| Relay (frp) דרך מארח ה-tracker | כל | אפס | מוגבל ב-VPS, זמן כפול | עובד | `internet relay wan` |
| SSH הפוך | כל | אפס, דורש שרת | כנ"ל | עובד | `internet relay wan` |
| IPv6 ישיר | כל | firewall בראוטר | כמו port forward | אין NAT | `internet plain wan` |
| Tailscale | כל | התקנה בכל צד | ישיר או DERP | עובד | `internet tailscale wan` |

הסקריפט `08_r0.sh` מקבל את התוויות האלה (`LINK`, `OVERLAY`, `LOCATION`) ורושם אותן במניפסט של כל ריצה.
