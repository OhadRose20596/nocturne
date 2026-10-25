# מדריך המשך, 17.10.2026

מאיפה ממשיכים אחרי הפגישה עם חן. כל צעד: על איזה מחשב, מה להריץ, מה לשמור. משפטים קצרים. בלי עברית בתוך בלוקי פקודות.

## 0. איפה עצרנו ומה השתנה

**נקודת העצירה (11.10):** 32B רץ על שני המק-ים ב-PP בכ-4 tok/s. ל-7B עותקים עצמאיים ניצחו חלוקה. שלוש שורות Air→M5 בטבלה B פסולות וצריך לאסוף מחדש. נכתב STATUS ו-ONE_PAGER לקראת הפגישה. ניסיון TP על 32B נכשל בלי לוג.

**מה חן ביקש (11.10):** Git. משפטים קצרים בלי מקף ארוך. GOAL של P2P בלבד. יעדים קצרים בתחילת המסמך: מכונה יחידה; חלוקה לפי שכבות; יותר משני מחשבים ואינטרנט ומודלים גנריים; חקר ביצועים; אפליקציית P2P עם liveness. Flow diagram מוסכם. מסמך חומרה ותוכנה. Formal background and design. PP מול TP מעשי. ולמה TP על 32B נכשל. הפירוט ב-[MEETING_CHEN_2026-10-11.md](MEETING_CHEN_2026-10-11.md).

**מה כבר נעשה (17.10, בלי להריץ כלום על המחשבים):**

| קובץ | מה |
|---|---|
| `docs/GOAL_AND_OBJECTIVES.md` | v4. מטרה של P2P, חמישה יעדים. הישן ב-`docs/archive/` |
| `docs/FLOW.md` | דיאגרמת זרימה (Mermaid) ושלבים במילים. טיוטה לאישור חן |
| `docs/PLATFORM.md` | מכונות, תוכנה, GPU מול CPU, Metal מול CUDA כ-review |
| `docs/BACKGROUND_AND_DESIGN.md` | הגדרות, PP מול TP, פורמט מודל ומניפסט, השאלה על TP |
| `docs/he/MEETING_CHEN_2026-10-11.md` | סיכום הפגישה וטבלת מעקב |
| `docs/he/SHARED_RESEARCH_AND_DECISIONS.md` | T3 ו-I019 |
| `docs/CURRENT_CONTEXT.md`, `README.md`, `CLAUDE.md` | מסונכרנים |
| `scripts/10_capacity.sh` | תומך ב-`SPLIT_MODE=tensor` לניסוי ה-TP עם לוגים |
| `nocturne/.git` | אותחל. ה-commit הראשון עליך (סעיף 1) |

## 1. Git: commit ראשון (על ה-M5, 10 דקות)

ה-repo סודר ל-git ב-25.10 (הערות מינימליות בסקריפטים, מסמכים מסונכרנים, המסמך המשותף של העוזרים מחוץ ל-git). הפקודות המלאות, ה-remote ב-GitHub וכללי העבודה: [GIT_HOWTO.md](GIT_HOWTO.md). בקצרה, מתוך תיקיית `nocturne`:

```bash
rm -f .git/index.lock .git/tvniLyM
git add -A
git commit -m "Nocturne: PoC-1 results, supervisor meeting 2026-10-11, docs v4, scripts cleanup"
gh repo create nocturne --private --source=. --remote=origin --push
```

## 2. HTML של המסמכים בעברית (על כל מק, דקה)

```bash
bash scripts/14_build_he_html.sh
open docs/he/html/index.html
```

## 3. ניסוי Q7: למה TP על 32B נכשל (שני המק-ים, 30 דקות)

המטרה: לוג. לא מסקנה. מריצים TP ומיד אחריו PP באותם תנאים, בשני הכיוונים.

**על ה-Air (worker, 192.168.1.219):**

```bash
bash scripts/08_r0.sh worker MTL0
```

**על ה-M5 (requester):**

```bash
SPLIT_MODE=tensor bash scripts/10_capacity.sh split ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf 192.168.1.219 1,1 P1
bash scripts/10_capacity.sh split ~/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf 192.168.1.219 1,1 P1
```

ואז להחליף: worker על ה-M5 (`192.168.1.242`), requester על ה-Air, אותן שתי פקודות עם ה-IP של ה-M5.

מה לבדוק בכל תיקיית ריצה תחת `experiments/r3/capacity/` (שם עם `_tp`):

```bash
cat placement.log
grep -m5 -iE "error|abort|assert|not implemented|failed" stderr.log
tail -3 manifest.json
```

| מה רואים | פירוש |
|---|---|
| `Meta device` עם שני התקנים, `offloaded 65/65`, יש tok/s | TP נטען ורץ. חן צדק לגבי הזיכרון. להשוות tok/s ונכונות הטקסט מול ריצת ה-PP |
| `not implemented` או `unsupported mul_mat split states` | חסר מימוש ב-meta backend ל-Metal או ל-RPC. לא בעיית זיכרון |
| `failed to allocate` או OOM | זיכרון. לרשום את גודל ה-buffer בכל צד מתוך `load_tensors` |
| שגיאה ב-50 השורות הראשונות | דגל או תצורה. לרשום את השורה |

לשלוח לי את שתי התיקיות (או להדביק את `placement.log` ואת השגיאה הראשונה). אני אכתוב את הניתוח ל-RESULTS.md ול-I020, ואתן גם ל-Codex.

## 3א. קישוריות: כבלים ואינטרנט (נוסף 24.10)

כל האפשרויות, עם פקודות, ב-[CONNECTIVITY.md](CONNECTIVITY.md). השורה התחתונה:

- ממשיכים על Wi-Fi. זה הבסיס של כל המדידות.
- Thunderbolt Bridge בין שני המק-ים: אופציונלי, אחר צהריים אחד, אחרי ניסוי ה-TP. עונה על שאלה אחת: Wi-Fi או פרוטוקול בעלות ההופ. `LINK=thunderbolt`.
- כבל בין מק ל-PC של ליאור: לא. הם בדירות שונות.
- אינטרנט בלי Tailscale: ניסוי ראשון עם port forwarding בראוטר של צד אחד (`OVERLAY=portforward`). אם יש CGNAT, relay דרך frp על מארח ציבורי (`OVERLAY=relay`). Tailscale נשאר כגיבוי ולהשוואה. נרשם כ-ADR-009 (proposed). לבדוק עם חן.
- WSL2 אצל ליאור: חובה `networkingMode=mirrored` או port forward, אחרת ה-worker לא נגיש.

## 4. איסוף מחדש של טבלה B בכיוון Air→M5 (15 דקות)

**על ה-M5 (worker):**

```bash
bash scripts/08_r0.sh worker MTL0
```

**על ה-Air (requester):**

```bash
LINK=wifi OVERLAY=plain LOCATION=lan bash scripts/08_r0.sh rpc 192.168.1.242
python3 scripts/09_r0_report.py
```

לבדוק שב-REPORT.md השורות החדשות מסומנות תקינות ושה-placement מופיע מתוך הריצה עצמה.

## 5. לשלוח לחן (אחרי סעיף 1)

לשלוח קישור ל-GitHub ואת ארבעת המסמכים: GOAL_AND_OBJECTIVES, FLOW, PLATFORM, BACKGROUND_AND_DESIGN. לבקש שלושה דברים: אישור חמשת היעדים, אישור הדיאגרמה ב-FLOW.md או תיקונים, ותשובה לשלוש השאלות הפתוחות בסוף FLOW.md. אם תרצה גרסת Word, תגיד לי.

## 6. Codex

להדביק ל-Codex: "נוסף I019 במסמך המשותף. בקר את GOAL_AND_OBJECTIVES v4 ואת FLOW.md מול הערות המנחה ב-MEETING_CHEN_2026-10-11.md. ענה ב-I020." את תוצאות סעיף 3 לצרף כשיהיו.

## 7. השבועיים הבאים, לפי סדר

1. **מכונת ה-NVIDIA של ליאור.** לשלוח לו את ה-repo (clone מ-GitHub, לא zip). על המכונה, ב-WSL2:

```bash
bash scripts/07_preflight.sh 192.168.1.242
BACKEND=cuda bash scripts/00_build_llamacpp.sh
bash scripts/01_list_devices.sh
bash scripts/02_run_local.sh CUDA0
bash scripts/05_bench.sh
```

   לרשום את ה-GPU, ה-VRAM והדרייבר ב-`experiments/inventory/`. זה סוגר את O1 על שלוש מכונות.

2. **אינטרנט ולא Wi-Fi.** אותו ניסוי 32B כשמחשב אחד בדירה אחרת. פעם ראשונה דרך Tailscale (הכי מהיר להקים): worker `BIND=<tailscale-ip> bash scripts/08_r0.sh worker MTL0`; requester `LINK=internet OVERLAY=tailscale LOCATION=wan bash scripts/08_r0.sh rpc <tailscale-ip>` ואז `10_capacity.sh split ... <tailscale-ip>`. פעם שנייה בלי Tailscale, לפי CONNECTIVITY.md סעיף 3 (`OVERLAY=portforward` או `relay`). זה O3 ו-ADR-009.

3. **סט משימות לאיכות.** 10 עד 20 שאלות הסקה קבועות. להריץ על 32B מפוצל ועל 7B מקומי. לשמור את התשובות. זה O4 שאלה 1 בכיוון האיכות.

4. **ניסוי אמינות ראשון.** להרדים את ה-worker באמצע ריצה ולתעד מה קורה. היום: הריצה מתה. זה ה-baseline ל-O5 liveness.

5. **ADR-007 עד 15.11.** לאחר ניסוי Q7 יש ראיה על TP ב-llama.cpp. אם TP לא רץ מעל RPC, זה נרשם כמגבלה של מסלול A, לא כהחלטה אוטומטית למסלול B.

## 8. מה לא לעשות

- לא לערוך קבצי ריצה ישנים. פסולים עוברים ל-`runs_superseded_*`.
- לא להסיק מהירות מהשוואה בין כלים שונים או SHA שונה.
- לא להכניס עברית לתוך פקודות.
- לא להפוך את ההנחיות של חן ל-ADR בלי שהוא יאשר את v4.
