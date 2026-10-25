# מדריך המשך: מהמצב הנוכחי עד סיום R3

**מצב (25.10.2026):** הכלים והפקודות בתוקף. השלבים R0 עד R3 הם שמות פנימיים ומקבילים ל-O2 ול-O4 ב-GOAL_AND_OBJECTIVES.md v4. R1 ו-R2 (מסלול PyTorch) מוקפאים עד הכרעת ADR-007. ניסוי ה-TP מתבצע עכשיו עם `SPLIT_MODE=tensor scripts/10_capacity.sh` ולא בפקודה הידנית שבסעיף R3d. המשך העבודה: NEXT_STEPS_2026-10-17.md.

**גרסה:** 2026-10-10 (אחרי T1, I010–I011). משלים את `RUNBOOK_R0.md` (שמכסה את R0 צעד-צעד) ומוביל עד שיש לכם את שלוש הטבלאות של R3. כל שלב: מה השאלה, על איזה מחשב, איזו פקודה, מה נשמר, מה "גמור", ומה כותבים במסמך המשותף.

**נקודת המוצא (T1):** אנחנו מודדים היתכנות ויעילות של חלוקת מודל על מחשבים נוספים, כשהרשת היא צוואר הבקבוק הנתון. השהיה נרשמת כתנאי, לא כיעד. שלושת הצירים (I010): **קיבולת** תחת מדיניות זיכרון מוצהרת · **בקשה יחידה** · **תפוקה תחת עומס מול עותקים עצמאיים**.

## איפה אנחנו

| | מצב | ראיה |
|---|---|---|
| סביבה | Air M3 (16GB, `192.168.1.219`, user `ohadrosentswaig`) + M5 (24GB, `192.168.1.242`, user `orosentswaig`), Wi-Fi ביתי ~12ms RTT, 250+ Mbit/s. אותו llama.cpp `1e7bcf3`, אותו hash מודל | `experiments/inventory/`, `experiments/r0/REPORT.md` |
| טבלה A | ✅ מלאה (M3: 208/20.0; M5: 683/26.3; CPU בשניהם) | REPORT.md |
| טבלה C | ✅ 6 שורות Wi-Fi | REPORT.md |
| טבלה B | ⏳ באיסוף מחדש אחרי תיקון שני באגים (סדר מרוחק-קודם; תחביר `-ts` של llama-bench). 18 ריצות ישנות שמורות ב-`runs_superseded_ts_syntax/` | I008, I010 §5 |
| כלים | `07_preflight`, `08_r0`, `09_r0_report` מתוקנים; חדשים: `10_capacity.sh`, `11_load.py`, `12_rpc_trace.py` (נבדקו מול כלים מזויפים, לא על החומרה) | `scripts/` |
| החלטות | T1 בלבד. מנוע (ADR-007) פתוח, מוכרע לפי כלל סימטרי אחרי מדידות | `SHARED_RESEARCH_AND_DECISIONS.md` §9.2 |

## מפת הדרך

```
R0  ──► R3a קיבולת ──► R3b עומס ──► R3c trace ──► R3d Ethernet (נקודה) ──► R3e TP (פעם אחת) ──► סיכום R3
טבלה B                                                                                              │
                                                   R1/R2 (PyTorch) ◄── רק אם R3c/R3e מראים פער ◄────┘
```

כל חץ הוא שער: לא עוברים הלאה בלי התוצר של השלב הקודם ב-`experiments/`. זמן כולל משוער: 5–7 ערבי עבודה לשני המק-ים, בלי R1/R2.

---

## שלב 1 — סגירת R0: טבלה B מתוקנת (ערב אחד)

**שאלה:** מה קורה ל-pp/tg כשמחלקים 7B בין שני המק-ים, בשלושת ה-splits, בשני הכיוונים, על ה-Wi-Fi הקיים — עם הוכחת מיקום מתוך הריצה הנמדדת.

לפי `RUNBOOK_R0.md` שלבים 2–3, בסדר הזה ובלי לרוץ משני הצדדים בו-זמנית:

```bash
# M5 (worker):  טרמינל 1: iperf3 -s      טרמינל 2: BIND=192.168.1.242 scripts/08_r0.sh worker MTL0
# Air (requester):
LINK=wifi OVERLAY=plain LOCATION=lan scripts/08_r0.sh rpc 192.168.1.242
# ואז הפוך: Air worker (BIND=192.168.1.219 ... MTL0), M5 requester:
LINK=wifi OVERLAY=plain LOCATION=lan scripts/08_r0.sh rpc 192.168.1.219
python3 scripts/09_r0_report.py
```

ה-splits רצים `1,0 → 1,1 → 0,1` (**מרוחק קודם**): `1,0` = הכל על ה-worker כולל שכבת הפלט (608KB logits חוזרים לכל טוקן); `1,1` = 15 שכבות מרוחקות, 14 + פלט מקומי; `0,1` = הכל מקומי דרך נתיב ה-RPC (בקרת רישום backend).

**נשמר:** `experiments/r0/runs/<run_id>/` עם `bench.json`, `placement.log` **מהריצה הנמדדת** (`-v`), `stderr.log`, `manifest.json`.
**גמור כש:** `REPORT.md` → Acceptance: B = 2 צירופים (2 כיוונים × Wi-Fi) ✅, "placement verified" ✅ לכל שורה (הדוח מדפיס `layers: RPC0=15, MTL0=14` וכד'), ⚠ שונות מסומן אם יש.
**למסמך המשותף:** `REPORT.md` + משפט אחד: היחס `1,1`/`1,0` (output מקומי מול מרוחק) — זו הבדיקה הראשונה של כלל "שלב הפלט על ה-requester".

**קורא נכון את המספרים:** ms/טוקן = `avg_ns/64` מתוך bench.json, לא היפוך הממוצע. אין להסיק מההפרש "עלות רשת" — זה הפרש *תצורה*; ההפרדה באה ב-R3c.

---

## שלב 2 — R3a: קיבולת (ערב אחד + זמן הורדה)

**שאלה:** האם צירוף שני המחשבים מריץ מודל שאף אחד מהם לא מריץ לבד — **תחת מדיניות זיכרון מוצהרת**?

### 2.1 מודלים (אותו קובץ בשני המחשבים; ה-preflight מאמת hash)

```bash
brew install hf   # אם אין
hf download bartowski/Qwen2.5-14B-Instruct-GGUF Qwen2.5-14B-Instruct-Q4_K_M.gguf --local-dir ~/models   # ~9.0GB — מדרגת כיול
hf download bartowski/Qwen2.5-32B-Instruct-GGUF Qwen2.5-32B-Instruct-Q4_K_M.gguf --local-dir ~/models   # ~19.9GB — המועמד
```

להוריד **פעם אחת** ולהעביר ל-M5 ב-scp/AirDrop (סעיף 0 ב-RUNBOOK), כדי שה-hash יהיה זהה בוודאות. אחרי ההעתקה, בכל מחשב: `shasum -a 256 ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf`.

### 2.2 מדיניות — לקבוע לפני הריצה

- **P1** "כולו על המאיצים בתקציב": `-ngl 99 -fit off`. "נכנס" = כל השכבות offloaded + generation של 32 טוקנים הסתיימה + בלי `failed to allocate`. זו הטענה המדידה והצנועה, ואיתה מתחילים.
- **P2** "רץ בכלל": `-fit on`, llama.cpp רשאי להשאיר שכבות על CPU; נרשם כמה הגיעו ל-GPU.

12.1/18.2GB של Metal הם המלצות **רכות**; יכול להיות ש-32B רץ על ה-M5 תחת P2 ואפילו P1 — וזו תוצאה תקפה, לא כישלון של הניסוי. timeout = "לא הסתיים בזמן", לא הוכחת חוסר זיכרון.

### 2.3 הריצות

```bash
# על כל מחשב בנפרד (מקומי, בלי worker):
scripts/10_capacity.sh local ~/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf P1
scripts/10_capacity.sh local ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf P1
scripts/10_capacity.sh local ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf P2
# בשניים (worker על ה-M5 כמו בשלב 1; requester על ה-Air):
scripts/10_capacity.sh split ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf 192.168.1.242          # ts אוטומטי = פרופורציונלי לזיכרון
scripts/10_capacity.sh split ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf 192.168.1.242 3,2      # ~60% על ה-M5, 40% + פלט על ה-Air
# ואז גם בכיוון ההפוך (worker Air, requester M5) — פעם אחת, לתעד שההיתכנות לא תלויה בכיוון
```

כל ריצה מדפיסה `verdict: fits_under_policy=yes|no|timeout` ושומרת `experiments/r3/capacity/<run_id>/` (פקודה, stderr מלא, שורות `load_tensors`, זיכרון שיא של ה-requester, swap לפני/אחרי). ב-32B מקומי על ה-Air צפו ל-`failed to allocate` או לדשדוש ב-swap — התיקייה תראה את זה.

**גמור כש:** יש manifest לכל שורה בטבלה: {14B, 32B} × {Air, M5, split} × {P1, P2 היכן שרלוונטי}, ועבור כל ריצת split — `placement.log` עם שכבות בשני הצדדים ו-tok/s.
**תנאי הצלחה לטענת קיבולת (I010):** אותו מודל נכשל בכל מחשב יחיד *לפי המדיניות*, ומסיים חלוקה תקינה בשניים, עם זיכרון ותפוקה מדווחים. בלי reference ל-32B — `smoke-test`, לא "נכונות עברה".
**למסמך המשותף:** טבלת קיבולת + המשפט "המודל הגדול ביותר שרץ על מחשב יחיד תחת P1 היה X; על שניים — Y".

---

## שלב 3 — R3b: תפוקה תחת עומס, חלוקה מול עותקים עצמאיים (ערב אחד)

**שאלה:** למודל שנכנס בכל מחשב (7B), מתי כדאי לפצל ומתי עדיף שהמחשב השני ישרת משתמש נוסף? זו ההשוואה ש-I010 הגדיר כחיונית, והיא עונה ישירות על "יעילות השימוש במחשבים נוספים".

### 3.1 שתי תצורות, אותם דגלים, אותו מודל 7B

**A — חלוקה:** worker על ה-M5 (`BIND=192.168.1.242 scripts/08_r0.sh worker MTL0`, כמו תמיד), ועל ה-Air:

```bash
scripts/13_serve.sh split 192.168.1.242          # ts ברירת מחדל 1,1 (מרוחק קודם); MODEL מ-.env
```

**B — עותקים עצמאיים:** על **שני** המחשבים, בלי worker:

```bash
scripts/13_serve.sh solo                          # פורט 8080 בכל מחשב
```

`13_serve.sh` מוצא את הבינארי דרך `common.sh` (ב-Air: `third_party/`, ב-M5: `~/dev/llama.cpp`) ומריץ בכל תצורה בדיוק את אותם דגלים: `-ngl 99 -c 4096 -np 8 --metrics`. `-np 8` = עד 8 slots במקביל; `-c 4096` מתחלק ביניהם (512 לכל slot — מספיק ל-128+64). הלוג נשמר ב-`experiments/r3/load/server_logs/`. בדיקת חיים: `curl -s http://<ip>:8080/health`.

### 3.2 העומס (מ-Air, או ממחשב שלישי אם יש — ה-requester עמוס גם ככה)

```bash
for C in 1 4 8; do
  python3 scripts/11_load.py --url http://192.168.1.219:8080 --concurrency $C --requests $((8*C)) \
      --prompt-tokens 128 --max-tokens 64 --label split_1,1_wifi
done
# תצורה B — שני URL-ים, הסקריפט מנתב round-robin:
for C in 1 4 8; do
  python3 scripts/11_load.py --url http://192.168.1.219:8080 --url http://192.168.1.242:8080 --concurrency $C --requests $((8*C)) \
      --prompt-tokens 128 --max-tokens 64 --label independent_wifi
done
```

כל ריצה שומרת `experiments/r3/load/<run_id>/{requests.jsonl,summary.json,summary.md}` ומדפיסה שורת טבלה: `Q` (טוקני פלט של בקשות שהושלמו / חלון משותף), req/s, שיעור השלמה, p50/p95.

**גמור כש:** 6 שורות (2 תצורות × 3 רמות עומס), כולן עם completion rate 100%, ואפשר לחשב `Q_split / Q_independent` לכל C.
**איך לקרוא:** אם היחס < 1 בכל C — למודל שנכנס בכל מחשב, עדיף עותקים; החלוקה מוצדקת רק לקיבולת (שלב 2). אם היחס עולה עם C — ה-pipeline מתחיל למלא bubbles; זו תוצאה מעניינת, אבל רק trace (לא `--parallel` כשלעצמו) מוכיח חפיפת שלבים.
**אופציונלי אם הזמן מאפשר:** אותו דבר ל-14B, שנכנס רק ב-M5 — אז "עותקים עצמאיים" = עותק אחד בלבד, והחלוקה מתחרה במחשב יחיד.

---

## שלב 4 — R3c: trace של פקודות RPC לטוקן (שעה)

**שאלה:** כמה אינטראקציות ובתים התשתית מוציאה לכל טוקן — ה-`k` ב-`k × עלות`, מתוך מדידה ולא מספירת שמות פקודות (I010 §4).

```bash
# worker (M5) עם debug — אותו סקריפט, משתנה סביבה אחד:
GGML_RPC_DEBUG=1 BIND=192.168.1.242 scripts/08_r0.sh worker MTL0
# requester (Air): ריצה אחת של 64 טוקנים לכל split, בלי bench:
for s in 1,0 1,1; do scripts/08_r0.sh placement 192.168.1.242 $s; done     # 8 טוקנים כל אחת — מספיק לדפוס
# אחרי Ctrl-C ל-worker, על אותו מחשב:
python3 scripts/12_rpc_trace.py experiments/r0/worker_logs/<הלוג-החדש>.log --tail 8 --label wifi_1,0_then_1,1
```

הפלט: פקודות לצעד-חישוב לפי סוג, בתים נכנסים (`set_tensor`) ויוצאים (`get_tensor`) לצעד, ושיעור `graph_recompute` מול `graph_compute` מלא. בתצורת `1,0` צפו ל-`get_tensor` של ~608KB לצעד (logits); ב-`1,1` ל-~7KB (hidden state). זה ה-"בתים לגבול" של H1, שחור על גבי לבן.

**מה זה לא:** לא RTT. `set_tensor`/`graph_compute` נשלחים בלי המתנה; רק `get_tensor` מחכה. המתנות בצד ה-requester דורשות trace נפרד (R1, אם יגיע).
**גמור כש:** `experiments/r3/trace/<run_id>/summary.md` לשני ה-splits.
**למסמך המשותף:** שורת הטבלה + משפט: "לכל טוקן ב-`1,0`: X פקודות, Y KB החוצה, Z% recompute".

---

## שלב 5 — R3d: נקודת רגישות אחת — Ethernet (ערב, אם יש כבל)

**שאלה:** כמה מהפער מקומי↔מרוחק נעלם כשהקישור מתנהג יפה? נקודה אחת, לא סריקה. **לא** מפרידה "רשת" מ"פרוטוקול" (RTT, jitter ו-BW משתנים יחד) — אבל נותנת את הגבול התחתון של עלות ההופ על חומרה צרכנית.

כבל USB-C↔Ethernet לשניהם דרך הראוטר, או Thunderbolt ישיר בין המק-ים. Wi-Fi כבוי בשניהם. ואז **בדיוק שלב 1** עם `LINK=ethernet` (שני הכיוונים), ואם הזמן מאפשר — חזרה על שלב 3 בתצורת A בלבד.

**גמור כש:** 2 צירופים נוספים בטבלה B (Acceptance: 4/4 ✅), ושורת Table C עם RTT < 1ms.

---

## שלב 6 — R3e: אפיון TP, פעם אחת (ערב)

**שאלה:** האם llama.cpp מסוגל להריץ `--split-mode tensor` מעל RPC בצירוף שלנו, ומה היעילות שלו תחת הרשת האיטית. תוצאה פתוחה; זה גם מסלול A של ADR-007.

```bash
# worker כרגיל. requester (Air), אותו 7B:
"$BIN/llama-cli" -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf -ngl 99 --rpc 192.168.1.242:50052 \
    --split-mode tensor -ts 1,1 -fit off -c 2048 -n 32 -p "Explain in two sentences what a transformer is." --temp 0 -st --simple-io --verbose \
    2> experiments/r3/tp_attempt_$(date -u +%Y%m%d-%H%M%S).log
```

שלוש תוצאות אפשריות, כולן תקפות: (א) נופל בטעינה/ב-assert (כמו issue #22307 במקרה אחר) → "לא נתמך בצירוף הזה", עם הלוג; (ב) רץ אבל הטקסט זבל → נכונות נכשלת; (ג) רץ ונכון → למדוד עם `llama-bench ... --split-mode tensor -ts 1/1` (5 חזרות) ולהשוות ל-`1,1` של PP. אם (ג): לוודא ב-`load_tensors` שהמשקולות באמת מפוצלות (buffer בשני הצדדים).

**גמור כש:** יש לוג + verdict אחד לכל קישור שבדקתם (Wi-Fi, ו-Ethernet אם היה). לא יותר מערב אחד — "אפיון אחד" הוא לא פרויקט.

---

## שלב 7 — R1/R2 (PyTorch): רק אם יש סיבה נמדדת

R1 (תעבורה משלנו: TCP מול Gloo-על-CPU) ו-R2 (בלוק TP אחד ב-PyTorch על Mac+NVIDIA) מתוארים ב-`RESEARCH_PLAN.md` §6. **תנאי הכניסה**, לפי כלל ההכרעה הסימטרי ב-ADR-007:

- R3c מראה ש-llama.cpp RPC מבזבז על כל טוקן הרבה יותר מגבול אחד (בתים או פקודות) **וגם** אין דרך לתקן זאת במיקום (למשל שלב פלט מקומי) — פער יעילות של התשתית.
- או R3e מראה ש-TP מעל RPC לא רץ כלל, והצוות מאשר ש"יכולת TP חוקי" היא דרישה.
- או שלב 2 מראה שחלוקה מעל 2 מחשבים נדרשת ו-llama.cpp לא נותן שליטה מספקת בהקצאה.

אם אף אחד מהתנאים לא מתקיים — ADR-007 נסגר על llama.cpp כמנוע ל-PoC-2, ו-R1/R2 נשארים מתועדים כאפשרות. אם מתקיים — R1 לפי התיאור בתכנית, עם ה-mandatory requirements לשני המנועים.

---

## שלב 8 — סיכום R3 (שעתיים)

לרכז ל-`experiments/r3/R3_SUMMARY.md` שלוש טבלאות + שורת trace + verdict של TP:

| ציר | מקור | מה בטבלה |
|---|---|---|
| קיבולת | `experiments/r3/capacity/*/manifest.json` | מודל × מחשב/ים × מדיניות → נכנס/לא/timeout, שכבות ל-GPU, זיכרון שיא, tok/s |
| בקשה יחידה | `experiments/r0/REPORT.md` טבלאות A+B | tok/s ו-`T_best_single/T_plan` לכל split; `1,1` מול `1,0` |
| עומס | `experiments/r3/load/*/summary.md` | `Q`, req/s, p50/p95 ב-1/4/8; `Q_split/Q_independent` |
| trace | `experiments/r3/trace/*/summary.md` | פקודות ובתים לטוקן לכל split |
| TP | `experiments/r3/tp_attempt_*.log` | verdict |

ואז במסמך המשותף: input עם הטבלאות, **ההכרעה ב-ADR-007** לפי הכלל הסימטרי (או "ראיות חסרות — הבדיקה החסרה היא …"), ושלוש שורות של "מה הופתענו ממנו". זה החומר לפגישה הבאה עם המנחה — כולל התשובה לנקודה 3 שלו: המערכת יודעת לחתוך לאורך, נבדק לרוחב, ויש מספרים למה לאורך.

---

## איפה המחשב השלישי (NVIDIA) נכנס

אחרי שלב 3, בכל נקודה: אותו zip (`nocturne_2026-10-10.zip`), `SETUP_GUIDE` §1b/1c, build ב-`~/dev/llama.cpp` עם `-DGGML_CUDA=ON -DCMAKE_BUILD_RPATH='$ORIGIN'`, אותו SHA, אותם קבצי מודל, `07_preflight.sh`. ואז: שלב 1 כלפיו (זוג הטרוגני — העמודות בטבלה מבדילות לבד לפי host/backend), שלב 2 עם N=3 (`-ts a,b,c` — שני ה-RPC קודם, המקומי אחרון), ושלב 3 עם שלושה עותקים. אם זה WSL2 — portproxy ל-50052 ול-8080/5201 (RUNBOOK §0.4).

## תקלות צפויות

| תסמין | מה לעשות |
|---|---|
| `cannot connect` מיד ב-`08_r0.sh rpc` | ה-worker לא רץ / BIND שגוי / שני הצדדים requester בו-זמנית. מחשב אחד בכל פעם |
| `recv failed ... Remote RPC server crashed` | להרים את ה-worker מחדש עם מכשיר מפורש (`MTL0`), לא `all`; לבדוק `--list-devices --rpc` מה-requester |
| 32B מקומי "נתקע" דקות | זה swap. `10_capacity.sh` יהרוג אחרי `TIMEOUT_S` (900s) וירשום timeout — לא "לא נכנס" |
| `11_load.py` מדווח 0 completed | ה-server לא על 0.0.0.0, או `-np` קטן מ-C, או timeout. לבדוק `curl http://<ip>:8080/health` |
| `12_rpc_trace.py`: "no [command] lines" | ה-worker רץ בלי `GGML_RPC_DEBUG=1`, או שהלוג הוא stdout בלבד (הסקריפט החדש שומר דרך `script`, אז אמור להיות שם) |
| TP נופל ב-`GGML_ASSERT` | תוצאה (א) בשלב 6 — לתעד את הלוג ולהמשיך; לא לנסות לתקן את llama.cpp |
