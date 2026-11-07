# מדריך הרצה: nocturne-bench על שני המקים

נכתב 2026-10-26 אחרי בניית `nocturne/bench/`. משלים את I021 במסמך המשותף. הפירוט הטכני של ה-bench נמצא ב-`nocturne/bench/README.md` (אנגלית). כאן רק מה שצריך כדי להריץ, בסדר, עם הפקודה המדויקת לכל מכונה.

מוסכמות: `[M5]` פקודה שמריצים על ה-Mac M5 (24GB, `orosentswaig@MMF4FWRG4R`). `[Air]` פקודה שמריצים על ה-Air (16GB, `ohadrosentswaig@Mac`). `[both]` על שתיהן. לפני כל פקודה לבדוק את `user@host` בשורת הפקודה. כתובות ה-LAN מהמדידות הקודמות: M5 `192.168.1.242`, Air `192.168.1.219`. אם הראוטר חילק כתובות אחרות, `ipconfig getifaddr en0` נותן את הנוכחית.

## 1. מה ה-bench עושה בפקודה אחת

`python -m nocturne.bench.run --cell A1 --network wifi --model qwen2.5-7b-q4km --runs 3` עושה את כל זה בלי התערבות: מודד RTT ורוחב פס ל-worker, מרים `ggml-rpc-server` על ה-Air דרך ssh, מרים `llama-server` על ה-M5 עם דגלים דטרמיניסטיים, מפעיל דוגמי ניצול ו-tcpdump, שולח 33 פרומפטים, עוצר הכול, מחשב S ו-B וניצול ונכונות מול BASELINE, כותב רשומת JSON ל-`nocturne/bench/results/`, ומוסיף שורה ל-`experiments/RESULTS.md`. חוזר על זה 3 פעמים.

| תא | מה רץ | מה חייב להיות מוכן |
|---|---|---|
| BASELINE | מודל על מכונה אחת, בלי `--rpc` | המודל על אותה מכונה |
| A1 | PP: requester M5, worker Air | ssh ל-Air, `iperf3 -s` על ה-Air, BASELINE של M5 קיים |
| A2 | A1 ועוד מודל טיוטה על ה-requester (8 ניחושים לצעד) | כמו A1, ועוד קובץ הטיוטה על ה-M5 |
| A2b | כמו A2 עם טיוטה קצרה וסלקטיבית (4 ניחושים, p-min 0.75) | כמו A2 |
| A4 | A1 עם 3 בקשות ארוכות במקביל | כמו A1 |
| B1 | TP בתוך llama.cpp עם `--split-mode tensor` מעל RPC (שאלה Q7) | כמו A1 |
| B1t | TP במנוע PyTorch שלנו, שני ranks | venv של torch על שתי המכונות, ראו סעיף שורה B |
| B1d | TP ב-distributed-llama (אושר ב-27.10, עדיפות נמוכה) | ראו 2.5 ו-README |

### 1.1 סבב 2: למה מריצים את A1 עד A4 מחדש

ב-26.10 בערב הרשומה עברה לסכמה גרסה 2 בעקבות הביקורת של Codex (I022). הרשומות הישנות (גרסה 1) נשארות כראיה ונקראות על ידי הדוח, אבל חסרים בהן שדות שנמדדים רק מעכשיו, ולכן ההשוואות הבאות דורשות סבב חדש של BASELINE, A1, A2, A2b ו-A4 על כל רשת. מה חדש ברשומה:

| היה | עכשיו | למה |
|---|---|---|
| `msgs_per_token_median` | `comm.rpc_commands_per_step`, `comm.tokens_per_step`, `comm.rpc_commands_per_token` | צעד אחד מייצר כמה tokens ב-A2 וב-A4 |
| `bytes_per_token_median` מ-`tcpdump -i any` | `comm.bytes_per_step`, `comm.bytes_per_token` על הממשק של הרשת (`comm.capture_iface`), איחוד חלונות | `-i any` ספר כפול ב-A4 |
| `memory_peak_gb` = סכום buffers מהלוג | `memory_buffers_gb` (מהלוג) ו-`memory_peak_gb` אמיתי, נדגם כל 500ms, לכל מכונה | זיכרון הוא תנאי הצלחה, אז הוא חייב להימדד |
| `weight_transfer_s` = זמן עד `/health` | `setup.load_total_s` ו-`setup.transfer_s` (מהודעת set_tensor הראשונה עד האחרונה, לוג ה-worker מקבל חותמות זמן) | להפריד העברה מטעינה |
| `ttft_s` מ-`prompt_ms` | `ttft_s` מהשליחה עד ה-token הראשון בזרם, ו-`ttft_server_s` הישן | כפי שהוגדר בתכנית |
| סף איכות בתוך `success` | `success` תפעולי בלבד, `quality_pass` נפרד | ריצה יכולה להצליח ולהיכשל באיכות |
| baseline לפי מודל, host, commit | גם sha256, קוונטיזציה, גרסת סולם ודגלי generation; ציון ה-baseline מחושב על אותם פרומפטים שרצו | A4 רץ רק על 3 פרומפטים |

**תנאי חדש להצלחה**: זיכרון חייב להימדד על כל מכונה. על מק המדידה היא `ps -o rss` של התהליך ואינה דורשת sudo. אם משהו מונע אותה, הריצה תיכשל; אפשר לעקוף ב-`--allow-unmeasured-memory`, וזה נרשם בהערות.

**סדר סבב 2** (לכל רשת, Wi-Fi קודם ואז hotspot):

1. `git pull` או סנכרון iCloud, ואז `python -m pytest tests/bench -q` בשתי המכונות.
2. כיול מונה הבייטים פעם אחת (סעיף 2.6).
3. BASELINE מחדש על כל מק, ריצה אחת מספיקה (סעיף 4).
4. A1, A2, A2b, A4 עם 3 חזרות (סעיפים 5 עד 7). `--live` אופציונלי ולא משפיע על המדידה (סעיף 8.1).
5. `python -m nocturne.bench.report`, ו-`python -m nocturne.bench.recompute --id <id>` על רשומה ישנה אחת כדי לראות מה אפשר היה לשחזר ממנה.

## 2. הכנה חד פעמית

### 2.1 סביבת פייתון

הריפו יושב ב-iCloud. Claude Code יצר היום `.venv` בתוך הריפו. זה יעבוד על המכונה שבה נוצר, אבל iCloud יסנכרן אלפי קבצים ו-venv אינו עובר בין מכונות. לכן ההוראות כאן יוצרות venv מחוץ ל-iCloud ב-`~/dev/nocturne-venv` על כל מכונה, ואפשר למחוק את `.venv` שבריפו (`rm -rf .venv`, הוא gitignored).

```bash
# [both]
cd "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Documents/אוניברסיטה/פרוייקט גמר/nocturne"
brew install uv iperf3
uv venv --python 3.12 "$HOME/dev/nocturne-venv"
source "$HOME/dev/nocturne-venv/bin/activate"
uv pip install -e ".[dev]"
python -m pytest tests/bench -q
```

הבדיקות חייבות לעבור לפני כל דבר אחר. אם `uv` לא מותקן ואין Homebrew, `python3 -m venv ~/dev/nocturne-venv` ואז `pip install -e ".[dev]"` עובד גם כן.

### 2.2 ssh מה-M5 ל-Air בלי סיסמה

ב-Air: System Settings, General, Sharing, Remote Login מופעל.

```bash
# [M5]
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 2>/dev/null || true
ssh-copy-id ohadrosentswaig@192.168.1.219
ssh ohadrosentswaig@192.168.1.219 'hostname; ls ~/dev/llama.cpp/build/bin/ggml-rpc-server'
```

הפקודה האחרונה חייבת להדפיס את שם המכונה ואת נתיב הבינארי. אם הבינארי ב-Air נמצא ב-`third_party/llama.cpp/build/bin` ולא ב-`~/dev/llama.cpp/build/bin`, לתקן את `bin` של ה-Air ב-`peers.yaml` בסעיף 2.4.

### 2.3 sudo בלי סיסמה ל-powermetrics ול-tcpdump

בלי זה הריצה עובדת, אבל השדות `utilization` ו-`bytes_per_token` יהיו null וההערה תסביר למה. כדאי להגדיר פעם אחת על שתי המכונות. להחליף את שם המשתמש בהתאם למכונה.

```bash
# [M5]
echo "orosentswaig ALL=(root) NOPASSWD: /usr/bin/powermetrics, /usr/sbin/tcpdump" | sudo tee /etc/sudoers.d/nocturne
sudo chmod 440 /etc/sudoers.d/nocturne
sudo -n powermetrics --samplers gpu_power -i 100 -n 1 | head -3
```

```bash
# [Air]
echo "ohadrosentswaig ALL=(root) NOPASSWD: /usr/bin/powermetrics, /usr/sbin/tcpdump" | sudo tee /etc/sudoers.d/nocturne
sudo chmod 440 /etc/sudoers.d/nocturne
sudo -n powermetrics --samplers gpu_power -i 100 -n 1 | head -3
```

### 2.4 קובץ peers לכל מכונה, מחוץ ל-iCloud

`nocturne/bench/peers.yaml` נמצא בתוך הריפו, כלומר בתוך iCloud, ולכן שתי המכונות יראו את אותו קובץ. זה שגוי כי בכל מכונה `ssh: null` צריך להצביע על המכונה המקומית. לכן כל מכונה מחזיקה קובץ משלה ב-`~/.nocturne/peers.yaml`, ותמיד מעבירים `--peers`. הקובץ שבריפו ישמש רק כדוגמה.

שימו לב: הקובץ שנוצר היום בריפו מכיל ל-M5 כתובת `132.73.206.61`. זו כתובת ציבורית, כנראה מרשת האוניברסיטה. לפי כלל 7 ב-CLAUDE.md ו-ADR-009, ה-worker נקשר רק לכתובת LAN או Tailscale שנבחרה בכוונה. בבית הכתובת היא `192.168.1.242`.

```bash
# [M5]
mkdir -p ~/.nocturne
cat > ~/.nocturne/peers.yaml <<'EOF'
models:
  qwen2.5-7b-q4km:  {path: "~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",  quant: Q4_K_M, draft: "~/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf"}
  qwen2.5-14b-q4km: {path: "~/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf", quant: Q4_K_M, draft: "~/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf"}
  qwen2.5-32b-q4km: {path: "~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf", quant: Q4_K_M, draft: "~/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf"}
peers:
  m5:
    host: MMF4FWRG4R
    ssh: null
    backend: metal
    device: MTL0
    mem_gb: 24
    os: macos
    bin: "~/dev/llama.cpp/build/bin"
    bind: 192.168.1.242
    networks: {wifi: 192.168.1.242, wired: 10.0.0.2, tailscale: 100.64.0.2, hotspot: 172.20.10.3}
  air:
    host: Mac
    ssh: ohadrosentswaig@192.168.1.219
    backend: metal
    device: MTL0
    mem_gb: 16
    os: macos
    bin: "~/dev/llama.cpp/build/bin"
    bind: 192.168.1.219
    networks: {wifi: 192.168.1.219, wired: 10.0.0.1, tailscale: 100.64.0.1, hotspot: 172.20.10.2}
roles: {requester: m5, workers: [air]}
tensor_split: null
rpc_port: 50052
server_port: 8089
EOF
```

```bash
# [Air]
mkdir -p ~/.nocturne
cat > ~/.nocturne/peers.yaml <<'EOF'
models:
  qwen2.5-7b-q4km:  {path: "~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",  quant: Q4_K_M, draft: "~/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf"}
peers:
  air:
    host: Mac
    ssh: null
    backend: metal
    device: MTL0
    mem_gb: 16
    os: macos
    bin: "~/dev/llama.cpp/build/bin"
    bind: 192.168.1.219
    networks: {wifi: 192.168.1.219, wired: 10.0.0.1, tailscale: 100.64.0.1, hotspot: 172.20.10.2}
roles: {requester: air, workers: []}
tensor_split: null
rpc_port: 50052
server_port: 8089
EOF
```

הכתובות של `wired` ו-`tailscale` הן placeholders. לפני ריצה על רשת כזו למלא את הכתובות האמיתיות (`tailscale ip -4` נותן את כתובת ה-Tailscale).

### 2.5 מודלים

ה-requester דוחף משקולות ל-worker, ולכן ל-A1 מספיק שהמודל יהיה על ה-M5. ל-BASELINE של ה-Air המודל חייב להיות גם על ה-Air. ל-A2 נדרש מודל הטיוטה על ה-M5. שם הקובץ ב-Hugging Face הוא באותיות קטנות, ולכן משנים שם אחרי ההורדה כדי להתאים ל-peers.yaml.

```bash
# [M5]
ls -la ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q8_0.gguf --local-dir ~/models
mv ~/models/qwen2.5-0.5b-instruct-q8_0.gguf ~/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf
shasum -a 256 ~/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf
```

```bash
# [Air]
ls -la ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

את ה-sha256 של הטיוטה לרשום ב-`experiments/RESULTS.md` תחת "Model hashes", כמו שאר המודלים.

#### זוג המודלים לשורה B1d (distributed-llama, עדיפות נמוכה)

שורה B משווה מערכת מול מערכת: llama.cpp עם GGUF מול distributed-llama עם פורמט q40 משלו. לכן צריך את אותה משפחת מודל בשני המנועים, Qwen3-8B, וכל אחד מהם מקבל BASELINE משלו. ההורדה כמה GB לכל קובץ, ולכן עושים אותה בבית. התיקון ל-ADR-007 אושר ב-27.10 (amendment 1 ו-2). התא נקרא עכשיו B1d.

```bash
# [M5]
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir ~/models
shasum -a 256 ~/models/Qwen3-8B-Q4_K_M.gguf
```

```bash
# [both]
git clone https://github.com/b4rtaz/distributed-llama.git ~/dev/distributed-llama
cd ~/dev/distributed-llama && make dllama && make dllama-api
```

```bash
# [M5]
cd ~/dev/distributed-llama && python3 launch.py qwen3_8b_q40 -skip-run
ls -la models/qwen3_8b_q40/
```

המודל המומר נדרש רק על ה-root (M5). על ה-Air מספיק הבינארי `dllama`. אחרי ההורדה מוסיפים ל-`peers.yaml` את `dllama_bin` ו-`dllama_nthreads` לכל מכונה ואת שני המודלים `qwen3-8b-q4km` ו-`qwen3-8b-dllama` כמו ב-`peers.example.yaml`. בדיקה בלי להריץ כלום:

```bash
# [M5]
python -m nocturne.bench.run --cell B1d --network wifi --model qwen3-8b-dllama --peers ~/.nocturne/peers.yaml --dry-run
```

### 2.6 כיול מונה הבייטים (פעם אחת)

B נמדד ב-tcpdump על הממשק שמוביל ל-worker. לפני שמצטטים אותו, בודקים שהמונה לא סופר כפול: הפקודה שולחת 10MB ידועים דרך אותו מסלול לכידה ומדפיסה נמדד מול צפוי. קודם על loopback בכל מכונה, ואז מה-M5 ל-Air על הרשת.

```bash
# [M5]
python -m nocturne.bench.calibrate_capture --local
python -m nocturne.bench.calibrate_capture --network wifi --peers ~/.nocturne/peers.yaml
```

```bash
# [Air]
python -m nocturne.bench.calibrate_capture --local
```

יחס של 1.0 עד 1.1 תקין (כותרות TCP). יחס קרוב ל-2 פירושו ספירה כפולה, ואז B לא נכנס לשום השוואה עד שזה מתוקן. את השורה מעתיקים לטבלת "Capture calibration" ב-`nocturne/bench/README.md`.

## 3. ריצה ראשונה: בדיקת עשן בלי להריץ כלום

```bash
# [M5]
cd "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Documents/אוניברסיטה/פרוייקט גמר/nocturne"
source "$HOME/dev/nocturne-venv/bin/activate"
python -m nocturne.bench.run --cell A1 --network wifi --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --dry-run
```

הפלט הוא רשימת כל הפקודות שהיו רצות, כולל פקודת ה-ssh ל-Air ושורת `llama-server` המלאה. לקרוא אותה פעם אחת. לבדוק: `--rpc 192.168.1.219:50052`, `-ngl 99`, `--temp 0`, `--seed 42`, ו-`-ts` לא מופיע (כי `tensor_split: null`). אם משהו לא נראה נכון, לתקן ב-peers.yaml ולא בקוד.

## 4. BASELINE על כל מכונה

BASELINE הוא הייחוס לנכונות. בלי רשומת BASELINE של ה-M5, כל תא אחר ייכשל בשלב הנכונות. מתחילים בתת-קבוצה של 5 פרומפטים כדי לראות שהכול עובד, ואז ריצה מלאה.

```bash
# [M5]
python -m nocturne.bench.run --cell BASELINE --host m5 --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 1 --subset 5 --no-results-md
python -m nocturne.bench.run --cell BASELINE --host m5 --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 3
```

```bash
# [Air]
cd "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Documents/אוניברסיטה/פרוייקט גמר/nocturne"
source "$HOME/dev/nocturne-venv/bin/activate"
python -m nocturne.bench.run --cell BASELINE --host air --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 3
```

הריצה המלאה על 7B לוקחת בערך 10 עד 15 דקות לחזרה על ה-M5, בעיקר בגלל שלושת הפרומפטים הארוכים. הרשומות נכתבות ל-`nocturne/bench/results/`. התיקייה בתוך iCloud, ולכן רשומה שנכתבה על ה-Air תופיע גם על ה-M5 אחרי סנכרון. זה רצוי.

מה לבדוק ברשומה: `success: true`, `correctness.task_accuracy` (על 7B צפוי 0.7 עד 0.9), `timing.decode_tps_median`, ו-`notes` ריק או עם הסברים שמובנים לכם.

## 5. A1: PP על Wi-Fi

לפני כל ריצה ברשת, ה-Air מריץ שרת iperf3. בלי זה `bw_mbps` יהיה null וההערה תגיד שהשרת לא רץ.

```bash
# [Air]
iperf3 -s -D
```

```bash
# [M5]
python -m nocturne.bench.run --cell A1 --network wifi --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 1 --subset 5 --no-results-md
python -m nocturne.bench.run --cell A1 --network wifi --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 3
```

ה-bench מרים את `ggml-rpc-server` על ה-Air בעצמו ועוצר אותו בסוף. אם ריצה קודמת נקטעה והפורט תפוס, על ה-Air: `pkill -f ggml-rpc-server`.

מה לבדוק ברשומה של A1 (סכמה v2), לעומת BASELINE של ה-M5:

| שדה | ערך צפוי | אם לא |
|---|---|---|
| `success` | true | לקרוא `failure_class` ו-`notes` |
| `quality_pass` | true | null פירושו שלא נמצא BASELINE תואם (sha256, קוונטיזציה, גרסת סולם, host) |
| `comm.rpc_commands_per_step` | 6 עד 7 (גבול אחד; 5 set_tensor קטנים, graph_recompute, get_tensor) | אם הרבה יותר, לבדוק את `worker_air.log` |
| `comm.tokens_per_step` | 1.0 ב-A1, כ-3 ב-A2, 3 ב-A4 | |
| `comm.bytes_per_step` | 15 עד 45 אלף, תלוי בכיול מסעיף 2.6 | null פירושו tcpdump לא רץ, ראו 2.3. `comm.capture_iface` אומר על איזה ממשק נלכד |
| `utilization` | 0.3 עד 0.5 לכל מכונה, כי ב-PP רק אחת מחשבת בכל רגע | null פירושו powermetrics לא רץ |
| `memory_buffers_gb` / `memory_peak_gb` | שניהם לכל מכונה. 7B: כ-3.4 ל-M5 ו-1.6 ל-Air ב-buffers | חסר פירושו שהריצה נכשלה על זיכרון לא מדוד, אלא אם עבר `--allow-unmeasured-memory` |
| `correctness.exact_match_ratio` | 1.0 בין שני Metal (נמדד) | `first_divergence_median` נמוך ושיטתי הוא באג |
| `correctness.task_accuracy` | שווה ל-`baseline_task_accuracy` על אותם פרומפטים | הפער מופיע ב-`quality_pass` |
| `setup.load_total_s` / `setup.transfer_s` | קר: כ-55s ו-1.67GB. חם: כ-10s ו-0.23GB | `cache_state` אומר אם ה-cache פעל |
| `timing.ttft_s` / `timing.ttft_server_s` | 0.3s בערך, הראשון מעט גדול מהשני | |
| `network.rtt_ms_median` | 2 עד 20 ב-Wi-Fi ביתי | נרשם, לא קריטריון |

## 6. A2: טיוטה על ה-requester

```bash
# [M5]
python -m nocturne.bench.run --cell A2 --network wifi --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 3
```

בהגדרות greedy הפלט של A2 חייב להיות זהה לפלט של A1 האחרון שהצליח. אם לא, הרשומה תסומן `wrong_output`. זה כשל אמיתי ולא רעש, ואז מדווחים עליו במסמך המשותף. ברשומה יש גם `draft_n` ו-`draft_n_accepted` לכל פרומפט מתוך `timings`. היחס ביניהם הוא שיעור הקבלה של הטיוטה. שימו לב שהתא הזה עדיין לא מעסיק את ה-Air יותר מ-A1, כי הטיוטה רצה על ה-M5. הגרסה עם טיוטה על worker (`--device-draft RPC[...]`) היא ה-session הבא.

### 6.1 A2b: טיוטה קצרה וסלקטיבית

```bash
# [M5]
python -m nocturne.bench.run --cell A2b --network wifi --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 3
```

ב-A2 על Wi-Fi הטיוטה ניחשה 8 tokens בכל צעד והתקבלו 39 אחוז מהם. A2b מנחשת עד 4 ועוצרת כשהביטחון יורד מ-0.75. ההשוואה המעניינת היא `tokens_per_step` ו-decode של A2 מול A2b על אותה רשת, והפלט בשניהם חייב להישאר זהה ל-A1.

## 7. A4: שלוש בקשות ארוכות במקביל

```bash
# [M5]
python -m nocturne.bench.run --cell A4 --network wifi --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 3
```

k ברירת המחדל הוא מספר הפרומפטים הארוכים, כלומר 3. זה התא הראשון שבו `utilization` של שתי המכונות אמור לעלות מעל 0.5, כי llama-server מעבד 3 רצפים בצינור. זו הבדיקה הישירה של טענת "ממלאים את הבטלה בלי TP".

## 7.5 שורה B: שלושה מנועים ל-TP

ההחלטה (ADR-007, amendment 2, 27.10): שורה B נמדדת בשלושה תאים, לכל אחד מנוע משלו ו-BASELINE משלו. השוואה בין B1t ל-A1 היא השוואת מערכות (bf16 מול Q4_K_M, ליבות ATen מול ggml), לא TP מול PP בתוך מנוע אחד. B1 הוא היחיד שמשווה TP ל-PP בתוך llama.cpp.

| תא | מנוע | מה צריך |
|---|---|---|
| B1 | llama.cpp `--split-mode tensor -fit off --rpc`, אותו build `1e7bcf3` | כמו A1. `-fit off` חובה: בלעדיו `fit.cpp` זורק שגיאה ב-tensor mode |
| B1t | המנוע שלנו ב-PyTorch, שני ranks, TCP משלנו | venv של torch על שתי המכונות, המודל בפורמט safetensors על שתיהן |
| B1d | distributed-llama | ראו 2.5 |

### 7.5.1 B1: מה נמדד עד כה

בדיקת מנגנון על ה-M5 לבד (29.10, worker על 127.0.0.1): ה-build בונה "Meta device" מ-MTL0 ומ-RPC0, מחלק את משקלי ה-7B לשניים (2084 MiB לכל מכשיר), מדליק flash attention בכוח, ומפענח עם פלט זהה ל-BASELINE ב-15.5 tok/s. `--split-mode row` נכשל בטעינה: "device RPC0 does not support split buffers". הריצה על שני המקים ב-Wi-Fi עדיין פתוחה. הפירוט ב-`experiments/RESULTS.md` תחת "Q7 mechanism check".

```bash
# [Air]
iperf3 -s
```

```bash
# [M5] בדיקת עשן ואז הריצה המלאה
python -m nocturne.bench.run --cell B1 --network wifi --model qwen2.5-7b-q4km --runs 1 --subset 5 --peers ~/.nocturne/peers.yaml
python -m nocturne.bench.run --cell B1 --network wifi --model qwen2.5-7b-q4km --runs 3 --peers ~/.nocturne/peers.yaml
```

```bash
# [M5] נקודת נתונים שנייה, --split-mode row, נרשמת כהערה ולא כתא
python -m nocturne.bench.run --cell B1 --network wifi --model qwen2.5-7b-q4km --runs 1 --peers ~/.nocturne/peers.yaml "--override-flag=--split-mode=row" --tag row --no-results-md
```

### 7.5.2 B1t: הכנה לכל מכונה

ה-venv של torch נפרד מזה של ה-bench ויושב מחוץ ל-iCloud. הנתיב המלא שלו נכנס ל-`peers.yaml` תחת `python` של כל מכונה.

```bash
# [both]
cd "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Documents/אוניברסיטה/פרוייקט גמר/nocturne"
uv venv --python 3.12 "$HOME/dev/nocturne-torch"
uv pip install --python "$HOME/dev/nocturne-torch/bin/python" -e ".[dev,bench-torch]"
"$HOME/dev/nocturne-torch/bin/python" -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
"$HOME/dev/nocturne-torch/bin/python" -m pytest tests/bench/test_torch_tp.py -q
```

המודלים בפורמט safetensors, על שתי המכונות באותו נתיב. ה-1.5B הוא כ-3GB, ה-7B כ-15GB.

```bash
# [both]
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/hf/Qwen2.5-1.5B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/hf/Qwen2.5-7B-Instruct
```

ב-`peers.yaml`: לכל מכונה `python: /Users/<user>/dev/nocturne-torch/bin/python`; למודלים `qwen2.5-1.5b-bf16` ו-`qwen2.5-7b-bf16` עם `engine: torch_tp` ו-`torch_model` (ראו `peers.example.yaml`); ברמה העליונה `torch_port: 50053` ו-`torch_api_port: 8092`.

### 7.5.3 B1t: בדיקות נכונות לפני ה-bench

שלב 1, rank אחד על ה-M5: הפלט החמדני של המנוע שלנו זהה ל-`generate` של transformers על 20 פרומפטים.

```bash
# [M5]
"$HOME/dev/nocturne-torch/bin/python" -m nocturne.engines.torch_tp compare --model ~/models/hf/Qwen2.5-1.5B-Instruct --world 1 --n 20
```

שלב 2, שני ranks. קודם שניהם על ה-M5 (loopback), אחר כך rank 1 על ה-Air. ה-worker נקשר לכתובת שנבחרה במכוון, אף פעם לא 0.0.0.0; הפורט מקבל כל rank 0 (אבטחה נדחתה לפי ADR-009).

```bash
# [M5] חלון 1
"$HOME/dev/nocturne-torch/bin/python" -m nocturne.engines.torch_tp worker --bind 127.0.0.1 --port 50053 --model ~/models/hf/Qwen2.5-1.5B-Instruct --rank 1 --world 2
```

```bash
# [M5] חלון 2
"$HOME/dev/nocturne-torch/bin/python" -m nocturne.engines.torch_tp compare --model ~/models/hf/Qwen2.5-1.5B-Instruct --world 2 --peer 127.0.0.1:50053 --n 20
```

```bash
# [Air] rank 1 על הרשת הביתית (הכתובת של ה-Air)
"$HOME/dev/nocturne-torch/bin/python" -m nocturne.engines.torch_tp worker --bind 192.168.1.219 --port 50053 --model ~/models/hf/Qwen2.5-1.5B-Instruct --rank 1 --world 2
```

```bash
# [M5]
"$HOME/dev/nocturne-torch/bin/python" -m nocturne.engines.torch_tp compare --model ~/models/hf/Qwen2.5-1.5B-Instruct --world 2 --peer 192.168.1.219:50053 --n 20
```

המונים צריכים להראות 56 מסגרות all-reduce לכל טוקן (28 שכבות, שתי all-reduce לשכבה) גם ב-1.5B וגם ב-7B.

### 7.5.4 B1t: ה-bench עצמו

BASELINE של המנוע הזה הוא `--world 1` על ה-M5. ה-7B ב-bf16 הוא 15GB ולכן ה-BASELINE שלו רץ רק על ה-M5; על ה-Air נכנס רק החצי של B1t (כ-7.6GB).

```bash
# [M5]
python -m nocturne.bench.run --cell BASELINE --host m5 --model qwen2.5-7b-bf16 --peers ~/.nocturne/peers.yaml --runs 3
python -m nocturne.bench.run --cell B1t --network wifi --model qwen2.5-7b-bf16 --peers ~/.nocturne/peers.yaml --runs 3
```

מה לצפות: 1 עד 2 tok/s על Wi-Fi ל-7B, כי כל טוקן הוא 56 סבבי הלוך ושוב חוסמים ב-6 עד 20 ms RTT. המספר הזה הוא המדידה, לא באג. ברשומה: `comm.rpc_commands_per_step` מגיע ממוני המנוע (מסגרות all-reduce), `bytes_per_token` מ-tcpdump על פורט 50053, `memory_buffers_gb` מ-`/metrics` של המנוע (הקצאות Metal של torch, כי `ps -o rss` מחסיר).

### 7.5.5 שורה B: מה נמדד ב-31.10.2026

שני המנועים רצו על אותו זוג מכונות (M5 מבקש, Air עובד) ועל אותו Wi-Fi ביתי (RTT של 8 ms), שלוש ריצות של 33 פרומפטים כל אחת. הרשומות ב-`nocturne/bench/results/` והשורות ב-`experiments/RESULTS.md`.

| מנוע | תא | decode tok/s | סבבים לטוקן | בייטים לטוקן | זיכרון לכל צד | פלט זהה ל-BASELINE |
|---|---|---|---|---|---|---|
| llama.cpp `--split-mode tensor` | B1 | 0.74 עד 0.77 | 170 פקודות RPC | 17 MB | 2.4 GB / 2.1 GB | 33 מתוך 33 |
| torch_tp (המנוע שלנו) | B1t | 1.32 עד 1.47 | 56 all-reduce | 0.78 MB | 9.75 GB / 7.5 GB | 31 מתוך 33 |
| llama.cpp `--split-mode layer` (להשוואה) | A2 | כ-14 | 2 | 80 KB | 4.2 GB / 2.1 GB | 33 מתוך 33 |

שני מנועי ה-TP חסומים על סבבי הלוך ושוב: 56 סבבים ב-8 ms הם כבר 0.45 שניות לטוקן. `--split-mode row` נכשל בטעינה בשני המקרים (loopback ו-Wi-Fi) עם `device RPC0 does not support split buffers`. שתי הסטיות של torch_tp הן עיגול bf16 של שני סכומים חלקיים בפרומפטים הארוכים, לא באג (הדיוק במשימה 1.0). B1d (distributed-llama) מוכן ולא רץ.

## 8. דוח

```bash
# [M5]
python -m nocturne.bench.report
```

מדפיס טבלת 4x4 ב-Markdown לכל רביעייה של מודל, רשת, מכונת requester ומנוע, עם הצלחה תפעולית ו-quality_pass בנפרד, task_accuracy, S, B, זיכרון ו-TPS חציוני. תאים כמו A2b ו-B1t מופיעים כשורה נפרדת מתחת לטבלה. רשומות שמופיעות ב-`results/EXCLUDE.yaml` (ריצות עשן, ריצות A2 בלי `--spec-type`, בדיקות loopback) לא נכנסות לטבלה ומופיעות ברשימה בסוף עם הסיבה; הקבצים עצמם נשארים. את הטבלה מדביקים ב-I-entry הבא במסמך המשותף יחד עם מזהי הרשומות.

### 8.1 תצוגה חיה של ריצה

בזמן שתא רץ אפשר לראות בדפדפן את שתי המכונות עובדות ואת ההודעות ביניהן. הדף רץ על ה-M5 בלבד, על 127.0.0.1, בלי שום ספרייה חיצונית, ולא כותב כלום לרשומה.

```bash
# [M5]
python -m nocturne.bench.run --cell A1 --network wifi --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 1 --subset 5 --no-results-md --live
```

הפקודה מדפיסה שורה `[live] http://127.0.0.1:8091/`. פותחים אותה בדפדפן. למעלה פס שלבים: probe, load (עם cold או warm), prefill, decode, done. משמאל כרטיס ה-requester, מימין כרטיס לכל worker: מצב (loading, computing, waiting, idle), זיכרון buffers מהטעינה, ועומס GPU אם יש sudo ל-powermetrics (אחרת `not measured`). באמצע זרם ההודעות: כל פקודת RPC כשהיא מגיעה, עם סימון לכל צעד decode ומונים של פקודות לצעד, פקודות בסך הכול ובייטים. למטה מונה טוקנים, tok/s, ובקשות בתעופה.

ריצה שהסתיימה אפשר להריץ מחדש מהלוגים שלה, במהירות אמיתית:

```bash
# [M5]
python -m nocturne.bench.live --logs nocturne/bench/results/logs/A1_wifi_qwen2.5-7b-q4km_m5_r2
```

בלי `--logs` הפקודה נצמדת לתיקיית הלוגים החדשה ביותר. `--speed 2` מכפיל את המהירות. בריצות שנרשמו לפני 2026-10-26 אין חותמות זמן בלוג של ה-worker, ולכן ההודעות משובצות לפי ספירת צעדים, בקירוב.

## 9. רשתות נוספות

אחרי ש-A1 עובד על Wi-Fi, אותן פקודות עם `--network wired`, `--network hotspot` או `--network tailscale`. ה-bench לוקח את הכתובת של ה-worker מתוך `networks` לפי סוג הרשת, מקשר אליה את `ggml-rpc-server`, ומודד אליה RTT ורוחב פס. ה-ssh ממשיך לרוץ על הכתובת שב-`ssh:`, וזה בסדר: בקרה על Wi-Fi, נתונים על הרשת הנמדדת. לפני כל רשת לוודא ש-`iperf3 -s` רץ על ה-Air.

### 9.1 כבל Thunderbolt בין המקים (זה `wired`)

כבל USB-C רגיל לא יוצר רשת בין שני מקים. נדרש כבל Thunderbolt או USB4 (כבל טעינה בלבד לא עובד, כבל USB 3 גם לא). macOS יוצר ממשק Thunderbolt Bridge. זו הרשת המהירה ביותר שיש לכם: RTT של עשיריות מילישנייה ורוחב פס של עשרות Gbps. לפי ההחלטה הקיימת `wired` הוא אבחון בלבד, והוא הגבול התחתון של `S*RTT`. זה גם המקום היחיד שבו ל-TP (שורה B) יש סיכוי להראות רווח.

הגדרה חד פעמית בשתי המכונות: System Settings, Network, Thunderbolt Bridge, Details, TCP/IP, Configure IPv4: Manually. ב-Air כתובת `10.0.0.1`, ב-M5 כתובת `10.0.0.2`, מסכה `255.255.255.0`, בלי router. Wi-Fi נשאר דלוק.

```bash
# [M5]
ping -c 5 10.0.0.1
iperf3 -c 10.0.0.1 -t 5
```

RTT צפוי מתחת ל-0.5 מילישנייה ורוחב פס של כמה Gbps. אם `ping` נכשל, הכבל אינו Thunderbolt או שהממשק לא קיבל כתובת. `ifconfig bridge0` מראה את הממשק. אחרי זה `networks.wired` ב-peers.yaml כבר נכון (10.0.0.1 ל-Air, 10.0.0.2 ל-M5) ואפשר להריץ:

```bash
# [M5]
python -m nocturne.bench.run --cell A1 --network wired --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 3
```

### 9.2 נקודה חמה מהטלפון (זה `hotspot`)

נקודה חמה היא רשת Wi-Fi שבה הטלפון הוא הנתב. התעבורה בין המקים עוברת מק, טלפון, מק, פעמיים באוויר, ולכן ה-RTT בדרך כלל גבוה מ-Wi-Fi ביתי (5 עד 30 מילישניות) ורוחב הפס נמוך ממנו. היא אינה תחליף ל-Wi-Fi הביתי בטבלה ולכן יש לה סוג רשת משלה, `hotspot`, כדי שהרשומות לא יתערבבו. התעבורה בין המקים לא עוברת בסלולר, אבל כל הורדה מהאינטרנט כן.

מתי זה שימושי: כשאתם לא בבית, למשל באוניברסיטה, שם רשת הקמפוס בדרך כלל מבודדת בין לקוחות ושני המקים לא רואים זה את זה. השימוש השני, החשוב יותר, מופיע ב-9.3.

כתובות הנקודה החמה משתנות בכל חיבור. ב-iPhone הן בדרך כלל `172.20.10.x`. לפני כל ריצה:

```bash
# [Air]
ipconfig getifaddr en0
```

```bash
# [M5]
ipconfig getifaddr en0
```

ולעדכן `networks.hotspot` של שתי המכונות ב-`~/.nocturne/peers.yaml` של ה-M5. אם גם ה-ssh צריך לעבור על הנקודה החמה, לעדכן גם `ssh:`. חלופה שחוסכת את זה: `ssh: ohadrosentswaig@Mac.local`, כי שם Bonjour עובד בכל רשת מקומית, גם על Thunderbolt Bridge. ל-`networks` עדיין נדרשות כתובות IP, כי `ggml-rpc-server -H` מקבל כתובת.

```bash
# [M5]
python -m nocturne.bench.run --cell A1 --network hotspot --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 3
```

ב-Android יש נקודות חמות שמבודדות בין לקוחות (AP isolation). אם `ping` בין המקים נכשל על הנקודה החמה, זו הסיבה, ולא תקלה ב-bench.

### 9.3 אינטרנט אמיתי בבית אחד: Air על הנקודה החמה, M5 על ה-Wi-Fi הביתי, Tailscale ביניהם

זה השימוש הכי שימושי בנקודה החמה. ה-Air מחובר לסלולר דרך הטלפון, ה-M5 לרשת הביתית, ו-Tailscale מחבר ביניהם. התעבורה יוצאת לאינטרנט באמת ועוברת דרך שני ספקים, בלי לחכות למחשב של ליאור. ה-RTT הצפוי הוא 30 עד 80 מילישניות, וגבוה יותר אם Tailscale לא מצליח לפתוח חיבור ישיר דרך ה-CGNAT של הסלולר ועובר דרך relay. שני המקרים הם תנאי אינטרנט אמיתיים, והרשומה שומרת את מה שנמדד.

```bash
# [both]
tailscale ip -4
```

לעדכן `networks.tailscale` בשתי המכונות ב-peers.yaml של ה-M5, ואז:

```bash
# [M5]
tailscale ping 100.x.y.z
python -m nocturne.bench.run --cell A1 --network tailscale --model qwen2.5-7b-q4km --peers ~/.nocturne/peers.yaml --runs 3
```

`tailscale ping` מדפיס אם החיבור ישיר או דרך relay. לרשום את זה ב-`notes` של הרשומה. כששתי המכונות באותה רשת ביתית, Tailscale עושה קיצור ישיר ב-LAN וה-RTT יהיה דומה ל-Wi-Fi. זה לא ניסוי אינטרנט, וההרכב מ-9.3 הוא הדרך לקבל אחד.

### 9.4 מה לצפות

| רשת | RTT טיפוסי | רוחב פס טיפוסי | תפקיד במטריצה |
|---|---|---|---|
| Wi-Fi ביתי | 2 עד 20 מילישניות | 50 עד 300 Mbps | הבסיס |
| Thunderbolt Bridge | 0.1 עד 0.5 | 10 עד 40 Gbps | אבחון, גבול תחתון, הסיכוי היחיד של TP |
| נקודה חמה, שני המקים עליה | 5 עד 30 | 50 עד 200 Mbps | תחליף ל-LAN מחוץ לבית |
| נקודה חמה ועוד Wi-Fi ביתי דרך Tailscale | 30 עד 80, יותר דרך relay | 10 עד 50 Mbps בכיוון העלאה של הסלולר | אינטרנט אמיתי בבית אחד |
| שני בתים דרך Tailscale | 15 עד 40 | 20 עד 100 Mbps העלאה | יעד T2, עם ליאור |

## 10. תקלות נפוצות

| תסמין | סיבה | פתרון |
|---|---|---|
| `ssh: connect to host ... refused` | Remote Login כבוי ב-Air או כתובת השתנתה | סעיף 2.2, ואז `ipconfig getifaddr en0` ב-Air |
| הריצה מבקשת סיסמת sudo באמצע | sudoers לא הוגדר או שם משתמש שגוי | סעיף 2.3. לבדוק `whoami` |
| `bw_mbps: null` | iperf3 לא רץ ב-Air | `iperf3 -s -D` ב-Air |
| `address already in use` על 50052 או 8089 | ריצה קודמת נקטעה | `pkill -f ggml-rpc-server` ב-Air, `pkill -f llama-server` ב-M5 |
| macOS שואל אם לאפשר ל-llama-server לקבל חיבורים | חומת האש | לאשר. llama-server נקשר ל-127.0.0.1, ה-rpc-server לכתובת ה-LAN שנבחרה |
| `dyld: Library not loaded` או שגיאת LC_RPATH | הבינארי נבנה בתוך iCloud | לבנות ב-`~/dev/llama.cpp` כמו ב-M5 |
| `weight_transfer_s` ארוך מאוד בכל ריצה | ה-cache של rpc-server לא פעיל | ה-bench מעביר `-c`. לבדוק שיש מקום ב-`~/Library/Caches/llama.cpp/rpc/` ב-Air |
| `task_accuracy` נמוך גם ב-BASELINE | פרומפט או grader בעייתי, לא המערכת | לפתוח את `results/logs/<id>/` ולראות איזה פרומפט נכשל. לא לתקן את הסולם באמצע סבב. לרשום ולתקן בגרסה 2 של הסולם |
| רשומה נכתבה אך `success: false` עם `oom` | מודל גדול מדי לתצורה | 7B נכנס בשתי המכונות. 14B ו-32B רק אחרי ש-7B עבר את כל התאים |

## 11. אחרי כל סבב

1. `git status` ב-M5. הרשומות ב-`results/` והשורות החדשות ב-`experiments/RESULTS.md` נכנסות ל-commit. הלוגים ב-`results/logs/` לא (gitignored).
2. I-entry במסמך המשותף: מזהי הרשומות, טבלת הדוח, מה הפתיע, ומה לא נבדק.
3. לא לשנות את `eval_set.jsonl` בין תאים של אותו סבב. שינוי בסולם מתחיל סבב חדש עם BASELINE חדש.
4. רשומות ישנות לא עורכים. `python -m nocturne.bench.recompute --id <record>` מדפיס מה אפשר לשחזר מהלוגים של רשומה v1 בלי לכתוב לתוכה.
5. בדוח, S הוא `rpc_commands_per_token` ו-B הוא `bytes_per_token`. ברשומות v1 של A4 שני הערכים היו לכל צעד ועם ספירה כפולה, וההערה על כך כבר ב-RESULTS.md.
