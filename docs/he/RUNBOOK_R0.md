# Runbook R0 — מה להריץ, על איזה מחשב, ומה נשמר איפה

**מצב (25.10.2026):** בתוקף. זהו המדריך הצעד-אחר-צעד לסקריפטים `08_r0.sh` ו-`09_r0_report.py`. השמות R0 וטבלאות A/B/C הם פנימיים. היעדים הרשמיים הם O1 עד O5 ב-GOAL_AND_OBJECTIVES.md v4. תוויות הרשת החדשות (`thunderbolt`, `internet`, `portforward`, `relay`) מתוארות ב-CONNECTIVITY.md.

**גרסה:** 2026-10-10 · מבוסס על `experiments/R0_PROTOCOL.md` (I005–I007). כל פקודה כאן שומרת את התוצאה שלה לקובץ בעצמה; אתה לא צריך להעתיק פלט ידנית.

> **מסגור (החלטת צוות T1, 10.10):** המספרים של R0 נקראים כ**יעילות החלוקה**, לא כהשהיה שצריך להוריד. הרשת האיטית היא נקודת המוצא. ההפרש בין מקומי למרוחק = חלק הרשת בזמן הטוקן; `0,1` = התקורה של הפרוטוקול עצמו; שניהם נכנסים ל-`E(N)`. אחרי טבלה B: **ניסוי היתכנות** — מודל שלא נכנס באף Mac לבד (Qwen2.5-32B Q4_K_M, ~19.9GB) על שניהם יחד.

## שני המחשבים

| סימון | מחשב | תפקיד ב-R0 |
|---|---|---|
| **M** | ה-Mac (Apple M3, 24GB, Metal) | requester בסבב 1, worker בסבב 2 |
| **N** | מכונת ה-NVIDIA (Linux או Windows+WSL2) | worker בסבב 1, requester בסבב 2 |

כל פקודה מתחילה **מתוך תיקיית הריפו** (`nocturne/`). ב-Mac:

```bash
cd "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Documents/אוניברסיטה/פרוייקט גמר/nocturne"
```

(כדאי alias: `alias noct='cd "...nocturne"'` ב-`~/.zshrc`.) ב-N — לאן ששכפלת את הריפו.

## מה נשמר, ואיפה

| מה | נשמר ב | על ידי |
|---|---|---|
| Inventory של כל מחשב (חומרה, SHA, דגלי build, rpath, רשת) | `experiments/inventory/<host>_<date>.md` | `07_preflight.sh` |
| כל ריצת מדידה: הפקודה המדויקת, פלט JSON גולמי של llama-bench, stderr מלא, manifest | `experiments/r0/runs/<run_id>/` | `08_r0.sh local` / `rpc` |
| לוג מיקום השכבות (אימות שהחלוקה באמת קרתה) | `experiments/r0/runs/<run_id>/placement.log` | `08_r0.sh rpc` / `placement` |
| מדידת רשת (ping, iperf3, tailscale) לכל סבב | `experiments/r0/runs/<run_id>/net.txt` | `08_r0.sh rpc` |
| לוג ה-worker | `experiments/r0/worker_logs/` | `08_r0.sh worker` |
| טבלאות A/B/C + בדיקות קבלה + CSV | `experiments/r0/REPORT.md`, `bench_r0.csv` | `09_r0_report.py` |

כל `run_id` נראה כך: `20261012-183000_ohads-MacBook-Air-2_bench_rpc_split_1_1`. ה-manifest שבתוכו רושם host, peer, split, link/overlay/location, SHA של llama.cpp, sha256 של המודל, PP/TG/REPS וקוד יציאה.

**שים לב:** אם הריפו ב-iCloud, הקבצים יסתנכרנו בין המחשבים לבד (ב-N צריך גישה לאותה תיקייה או `git pull`/העתקה ידנית של `experiments/r0/runs/` בסוף — ראה שלב 6).

---

## שלב 0 — פעם אחת, לפני הכל (M וגם N)

### 0.1 אותו llama.cpp, אותו מודל

```bash
git -C third_party/llama.cpp rev-parse --short HEAD      # חייב להיות זהה בשני המחשבים (כרגע 1e7bcf3)
grep MODEL .env                                          # אותו קובץ GGUF; ה-preflight ידפיס sha256
```

### 0.2 Preflight — מאמת שהבינארים נטענים ושומר inventory

```bash
# M (כתובת N כפרמטר):
scripts/07_preflight.sh <IP-של-N>
# N (כתובת M כפרמטר):
scripts/07_preflight.sh <IP-של-M>
```

מה לבדוק בפלט:

- **סעיף 1** — כל שורת `--version` חייבת להיות `OK`. אם `llama-cli --version: FAILED` עם `Library not loaded: @rpath/...` — זה E13 (rpath מוחלט). הסקריפט מדפיס שלוש דרכי תיקון; המהירה ב-Mac:
  ```bash
  B=third_party/llama.cpp/build/bin
  for b in llama-cli llama-bench llama-server ggml-rpc-server; do install_name_tool -add_rpath @executable_path "$B/$b"; done
  scripts/07_preflight.sh <IP>     # להריץ שוב
  ```
  אם זה לא עוזר (או במחשב שני בכלל) — לבנות מחדש **מחוץ ל-iCloud**:
  ```bash
  git clone https://github.com/ggml-org/llama.cpp ~/dev/llama.cpp && cd ~/dev/llama.cpp && git checkout 1e7bcf3
  cmake -B build -DGGML_RPC=ON -DGGML_METAL=ON -DCMAKE_BUILD_RPATH='@executable_path'   # ב-N עם NVIDIA: -DGGML_CUDA=ON -DCMAKE_BUILD_RPATH='$ORIGIN'
  cmake --build build --config Release -j
  ```
  `common.sh` מזהה `~/dev/llama.cpp` לבד. **לא** לכתוב `LLAMA_DIR` ב-`.env` — הוא משותף דרך iCloud לשני המק-ים; הגדרות פר-מחשב הולכות ל-`~/.nocturne.env` (מחוץ לריפו).
- **סעיף 2** — `--list-devices` מראה `MTL0` ב-M ו-`CUDA0` ב-N. את השמות האלה מעבירים ל-worker.
- **סעיף 4** — ping לעמית עובד. אם לא — אין טעם להמשיך לפני שהרשת מסודרת.
- **סעיף 5** — `ready for R0`.

שני קובצי ה-inventory (`experiments/inventory/`) הם התוצר הראשון. בלעדיהם אין טבלה.

### 0.3 iperf3 מותקן בשני המחשבים

```bash
brew install iperf3          # M
sudo apt install -y iperf3   # N (Ubuntu / WSL)
```

### 0.4 אם N הוא Windows + WSL2

ה-worker ב-WSL2 יושב מאחורי NAT. ב-PowerShell כמנהל, פעם אחת:

```powershell
netsh interface portproxy add v4tov4 listenport=50052 listenaddress=0.0.0.0 connectport=50052 connectaddress=$(wsl hostname -I)
netsh interface portproxy add v4tov4 listenport=5201  listenaddress=0.0.0.0 connectport=5201  connectaddress=$(wsl hostname -I)
```

ולאפשר 50052 ו-5201 ב-Windows Firewall. מ-M משתמשים אז ב-**IP של ה-Windows**, לא של ה-WSL. (פירוט: `docs/SETUP_GUIDE.md` §7.)

### 0.5 להחליט על הפרמטרים ולא לגעת בהם יותר

ברירת המחדל של הסקריפטים: `PP=256 TG=64 REPS=5 CTX=2048`. אם משנים — משנים **לכל** השורות, בשני המחשבים, דרך משתני סביבה (`PP=512 scripts/08_r0.sh ...`).

לפני כל סבב מדידה: לסגור דפדפנים, Ollama, LM Studio; לחבר לחשמל.

---

## שלב 1 — טבלה A: כל מחשב לבד (M וגם N, בלי קשר ביניהם)

```bash
scripts/08_r0.sh local
```

מריץ `llama-bench` פעמיים — `gpu_all_layers` (-ngl 99) ו-`cpu_only` (-ngl 0) — פעם אחת כל אחד, ושומר שתי תיקיות `runs/`. על המסך תראה:

```
    pp256:   208.30 ± 1.20 t/s
     tg64:    20.00 ± 0.30 t/s
```

✅ מצליח כאשר: שתי תיקיות חדשות ב-`experiments/r0/runs/` עם `bench.json` לא ריק, ו-std קטן מ-5% מהממוצע. אם ה-std גדול — להריץ שוב אחרי שהמחשב התקרר.

---

## שלב 2 — סבב 1: N הוא worker, M הוא requester — על **Ethernet**

חבר את שני המחשבים בכבל (או שניהם לאותו switch/router בכבל). ודא ש-Wi-Fi כבוי בשניהם, כדי שהתעבורה לא תתגנב דרכו.

### 2.1 על N — להרים worker

```bash
# טרמינל 1 ב-N:
iperf3 -s
# טרמינל 2 ב-N:
BIND=<IP-של-N-בכבל> scripts/08_r0.sh worker CUDA0
```

הפלט מסתיים ב-`on the requester use: --rpc <IP>:50052` ואז השרת ממתין. **להשאיר פתוח.** הלוג נשמר ב-`experiments/r0/worker_logs/`.

> `BIND` חייב להיות כתובת LAN (או Tailscale), לעולם לא `0.0.0.0` על ממשק ציבורי.

### 2.2 על M — להריץ את הסבב המלא

```bash
LINK=ethernet OVERLAY=plain LOCATION=lan scripts/08_r0.sh rpc <IP-של-N>
```

הסקריפט מסרב לרוץ בלי שלושת התיוגים — זה מכוון: הם נכנסים ל-manifest של כל שורה. הוא עושה, בסדר הזה:

1. **טבלה C** — ping ×20, iperf3 בשני הכיוונים, `tailscale status` אם קיים → `net.txt`.
2. **אימות מיקום** לכל split (`1,0`, `1,1`, `0,1`) — ריצת `llama-cli --verbose` קצרה (8 טוקנים) ששומרת את שורות `load_tensors` → `placement.log`. על המסך: `placement: buffers match split 1,1 (local=1 rpc=1)`. אם כתוב `do NOT match` — לעצור ולהסתכל בלוג לפני שמודדים.
3. **טבלה B** — `llama-bench` לכל split, פעם אחת → `bench.json`.

משמעות ה-splits — **הסדר הוא מרוחק קודם** (llama.cpp מעביר את מכשירי ה-RPC לראש הרשימה; אומת בלוגים ב-10.10): `1,0` = כל השכבות על ה-worker; `1,1` = חצי-חצי (שכבות 0–14 מרוחקות, 15–28 מקומיות); `0,1` = הכל מקומי **אבל דרך נתיב ה-RPC** — השורה הזו מבודדת את תקורת ה-RPC מהעברה ברשת, והיא צריכה להיות קרובה לטבלה A. (ב-llama-bench המפריד הוא `/` ולא `,` — הסקריפט מטפל בזה.)

⏱ עם REPS=5 ו-7B: בערך 10–20 דקות.

### 2.3 בדיקה מהירה שהכל נשמר

```bash
ls experiments/r0/runs | tail -7      # net + 3 placement + 3 bench
python3 scripts/09_r0_report.py        # טבלה B צריכה להראות 3 שורות עם ✅ בעמודת verified
```

---

## שלב 3 — סבב 2: M הוא worker, N הוא requester — עדיין על Ethernet

### 3.1 על M

```bash
# טרמינל 1:
iperf3 -s
# טרמינל 2:
BIND=<IP-של-M-בכבל> scripts/08_r0.sh worker MTL0
```

### 3.2 על N

```bash
LINK=ethernet OVERLAY=plain LOCATION=lan scripts/08_r0.sh rpc <IP-של-M>
```

אותו דבר כמו 2.2, בכיוון ההפוך. זה הסבב שבו CUDA הוא ה-requester ו-Metal הוא ה-worker — ההטרוגניות בשני הכיוונים.

---

## שלב 4 — סבבים 3 ו-4: אותו דבר על **Wi-Fi**

לנתק את הכבל, לחבר את שניהם לאותה רשת Wi-Fi, ולחזור על שלבים 2 ו-3 עם `LINK=wifi`:

```bash
# M כ-requester (N worker עם BIND=<IP-Wi-Fi של N>):
LINK=wifi OVERLAY=plain LOCATION=lan scripts/08_r0.sh rpc <IP-Wi-Fi-של-N>
# N כ-requester (M worker עם BIND=<IP-Wi-Fi של M>):
LINK=wifi OVERLAY=plain LOCATION=lan scripts/08_r0.sh rpc <IP-Wi-Fi-של-M>
```

**אופציונלי, סבב 5 — Tailscale:** אם שניהם ב-tailnet, אפשר להריץ עם `OVERLAY=tailscale` וכתובת ה-`100.x`. הסקריפט ישמור `tailscale status` ו-`tailscale ping` ב-`net.txt`, ומהם נדע אם החיבור היה direct או relay. זו תצורה שלישית בטבלה, לא תחליף לשתי הקודמות.

---

## שלב 5 — דוח

על המחשב שבו נמצאות כל תיקיות ה-`runs/` (ראה שלב 6 אם הן מפוצלות):

```bash
python3 scripts/09_r0_report.py
```

כותב `experiments/r0/REPORT.md` עם:

- **Consistency** — SHA אחד, hash מודל אחד, סט PP/TG/REPS אחד. אם יש ❌ כאן, השורות לא ברות השוואה.
- **טבלה A** — שורות local לכל host.
- **טבלה C** — שורה לכל מדידת רשת, בצירים נפרדים (location / link / overlay / path / RTT / iperf3 בשני הכיוונים).
- **טבלה B** — שורות rpc עם split, התוצאות, סיכום לוג המיקום ו-✅/❌.
- **Acceptance** — הבדיקות מ-`R0_PROTOCOL.md` §6: 2 מחשבים בטבלה A, 4 צירופים (2 כיוונים × 2 קישורים) עם שלושת ה-splits, שונות < 5%, מיקום מאומת לכל שורה.

כש-Acceptance כולו ✅ — R0 גמור. את `REPORT.md` מדביקים ל-`experiments/RESULTS.md` (או מקשרים אליו), ואת השורה "R0 הושלם + קישור" מוסיפים כ-input במסמך המשותף.

---

## שלב 6 — איסוף התוצאות משני המחשבים

הדוח צריך לראות את כל ה-`runs/` ביחד. שלוש אפשרויות:

- **iCloud / תיקייה משותפת** — אם N רואה את אותה תיקיית `nocturne/`, אין מה לעשות.
- **git** — בשני המחשבים: `git add experiments/r0 experiments/inventory && git commit -m "R0 <host>" && git push`, ואז `git pull` במחשב שמריץ את הדוח. התיקיות קטנות (JSON וטקסט).
- **העתקה ידנית** — מ-N ל-M: `scp -r <user>@<N>:<path>/nocturne/experiments/r0/runs/* experiments/r0/runs/` (וכנ"ל `inventory/`). שמות התיקיות כוללים hostname, אז אין התנגשויות.

---

## אם משהו נשבר

| תסמין | מה לעשות |
|---|---|
| `08_r0.sh rpc` נעצר מיד עם `set LINK=...` | זה מכוון. להוסיף `LINK= OVERLAY= LOCATION=` לפני הפקודה |
| `Failed to connect to <ip>:50052` ב-stderr.log | ה-worker לא רץ / BIND לא נכון / פיירוול / WSL2 בלי portproxy. לבדוק `nc -vz <ip> 50052` מה-requester |
| `placement: buffers do NOT match split` | לפתוח את `placement.log`. אם אין בכלל `RPC[` — ה-requester לא הגיע ל-worker; אם ב-`1,0` יש `RPC[` — llama.cpp הקצה בכל זאת buffer מרוחק, לתעד ולהמשיך |
| std גדול מ-5% | מחשב חם או משהו רץ ברקע. להמתין, לסגור, להריץ שוב את אותה שורה (תיקייה חדשה תיווצר; הדוח מציג את כולן, מוחקים ידנית את הישנה אם רוצים) |
| `exit code N` באדום | לקרוא `experiments/r0/runs/<run_id>/stderr.log`. ה-manifest שומר את קוד היציאה, והדוח מסמן את השורה |
| `--version` נכשל אחרי `install_name_tool` | לבנות מחדש מחוץ ל-iCloud (`07_preflight.sh` מדפיס את הפקודות) |
| ping עובד אבל iperf3 נכשל | `iperf3 -s` לא רץ בצד השני, או פורט 5201 חסום. בלי iperf3 הטבלה עדיין נבנית, רק עמודת ה-BW ריקה |

## מה לשלוח בסוף למסמך המשותף

1. שני קובצי `experiments/inventory/*.md`.
2. `experiments/r0/REPORT.md` (ו-`bench_r0.csv`).
3. שורה אחת: מה הפתיע. למשל אם `1,0` רחוק מטבלה A — זו תקורת RPC, לא רשת; אם `0,1` על Ethernet קרוב ל-`1,0` — ההופ זול; אם Wi-Fi הרבה יותר גרוע מ-Ethernet — זה הכיוון של H1.
