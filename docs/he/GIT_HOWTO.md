# Git לפרויקט: מה הוכן, מה אתה מריץ, ואיך עובדים יום-יום

נכתב 25.10.2026. בלי עברית בתוך בלוקי פקודות.

## 1. מה כבר מוכן ב-repo

| פריט | מצב |
|---|---|
| `nocturne/.git` | אותחל. ענף `main`. שם ואימייל הוגדרו. אין עדיין commit |
| `.gitignore` | מודלים, build, `.env`, לוגים רועשים (worker, smoke, server) בחוץ. ראיות הריצות (`stderr.log`, `placement.log`, `net.txt`) תחת `experiments/r0/runs` ו-`experiments/r3` בפנים |
| המסמך המשותף שלי ושל Codex | `docs/he/SHARED_RESEARCH_AND_DECISIONS.md` וה-HTML שלו ב-`.gitignore`. לא יעלו. ההחלטות T1 עד T3 הועתקו ל-`docs/DECISIONS.md` כדי שה-repo יהיה שלם בלעדיו |
| סקריפטים | הערות מינימליות: שתי שורות כותרת (מטרה, שימוש) בכל קובץ |
| מסמכים | מסונכרנים ל-v4. מסמכים ישנים קיבלו שורת מצב. RESEARCH_PLAN עבר ל-`docs/archive/` |

## 2. commit ראשון (על ה-M5, פעם אחת)

ה-commit חייב לרוץ מהמק: חלק מקבצי הניסויים נמצאים רק בענן iCloud ו-git צריך לקרוא אותם.

פתח Terminal. הקלד `cd ` ורווח, גרור את התיקייה `nocturne` מ-Finder, Enter. ואז:

```bash
rm -f .git/index.lock .git/tvniLyM
git status --short | wc -l
git add -A
git status --short | grep -c SHARED_RESEARCH || echo "private doc excluded: ok"
git commit -m "Nocturne: PoC-1 results, supervisor meeting 2026-10-11, docs v4, scripts cleanup"
git log --oneline
```

אם `git add` איטי: iCloud מוריד קבצים. לחכות, או `brctl download .` ולנסות שוב.

לבדוק מה נכנס לפני ה-push:

```bash
git ls-files | wc -l
git ls-files | grep -E "SHARED|\.env$|\.gguf$|third_party" || echo "nothing private tracked"
du -sh .git
```

## 3. remote ב-GitHub (פרטי)

עם `gh` מותקן (`brew install gh`, ואז `gh auth login`):

```bash
gh repo create nocturne --private --source=. --remote=origin --push
```

בלי `gh`: ליצור repo ריק בשם `nocturne` באתר GitHub, בלי README, ואז:

```bash
git remote add origin git@github.com:<user>/nocturne.git
git push -u origin main
```

להוסיף את ליאור ואת חן כ-collaborators דרך Settings של ה-repo.

## 4. Git בתוך iCloud: אזהרה והמלצה

iCloud מסנכרן גם את `.git`. זה עובד, אבל שני מק-ים שכותבים לאותה תיקייה עלולים ליצור קבצי conflict בתוך `.git` ולהשחית את ה-repo. הכלל: **רק ה-M5 עושה commit ו-push מתוך תיקיית iCloud.** ה-Air וליאור עובדים מ-clone רגיל מחוץ ל-iCloud:

```bash
mkdir -p ~/dev && cd ~/dev
git clone git@github.com:<user>/nocturne.git
cd nocturne
cp env.example .env
```

ואז בכל מכונה `~/.nocturne.env` עם `LLAMA_DIR` ו-`BIND` שלה. בהמשך כדאי שגם ה-M5 יעבור ל-`~/dev/nocturne` וה-iCloud יישאר למסמכים בלבד.

## 5. עבודה יום-יומית

| מתי | מה |
|---|---|
| לפני ריצה | `git pull` |
| אחרי ריצת ניסוי | `git add experiments/ && git commit -m "r3: tp probe 32B M5->Air"` ואז `git push`. תיקיית הריצה היא הראיה. לא עורכים אותה |
| שינוי בסקריפט | `bash -n scripts/*.sh` לפני commit. אם דגל של llama.cpp השתנה, לעדכן גם את SETUP_GUIDE באותו commit |
| שינוי במסמך | משפטים קצרים, בלי מקף ארוך. בדיקה: `grep -nP '[\x{2013}\x{2014}]' docs/*.md` |
| ריצה שהתבררה פסולה | לא למחוק. להעביר ל-`experiments/.../runs_superseded_<reason>/` ולרשום ב-RESULTS.md |
| מסמך בעברית | `bash scripts/14_build_he_html.sh` לפני commit כדי שה-HTML יהיה מעודכן |

הודעות commit קצרות, באנגלית, עם תחילית: `scripts:`, `docs:`, `r0:`, `r3:`, `adr:`.

## 6. מה לא נכנס ל-git, ולמה

| מה | למה |
|---|---|
| `docs/he/SHARED_RESEARCH_AND_DECISIONS.md` | יומן עבודה פנימי של שני העוזרים. ההחלטות שבו נמצאות ב-DECISIONS.md |
| `.env`, `~/.nocturne.env` | נתיבים וכתובות של מכונה ספציפית |
| `third_party/llama.cpp` | נבנה מחדש בכל מכונה מה-SHA המוצמד `1e7bcf3` |
| `*.gguf` | מודלים. מורידים מ-Hugging Face, hash ב-RESULTS.md |
| `experiments/r0/worker_logs/`, `experiments/logs/`, `experiments/r3/load/server_logs/` | לוגים רועשים של worker ושרת. הראיות הנדרשות כבר בתיקיות הריצה |

## 7. אם משהו השתבש

```bash
git status
git stash            # put local changes aside
git pull --rebase
git stash pop
```

repo שבור בתוך iCloud: לא לתקן ידנית. `git clone` טרי ל-`~/dev/nocturne` מ-GitHub, ולהעתיק רק את הקבצים ששונו.
