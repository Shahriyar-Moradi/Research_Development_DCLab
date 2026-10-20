# Testing the DCLab R&D studio with trusted friends

Packages 25.11, 32.15 to 32.17. Each friend clones this repository and runs DCLab on their own machine, with their own
Anthropic key, to solve a problem of their own. Everything they do is recorded in their workspace folder. `make share`
sends it to the owner's private records repository, on **a branch of their own** (`tester/<their id>`). The owner's
`make collect` reads every tester's branch and rebuilds each tester's workspace; the store (the storing phase,
`STORING_PHASE_PLAN.md`) keeps everything from there.

## For a friend: step by step

### 1. Get the studio (once)

```bash
git clone git@github.com:Shahriyar-Moradi/Research_Development_DCLab.git
cd Research_Development_DCLab
python -m venv .venv && .venv/bin/pip install -r requirements/base.txt pyarrow
python -m dclab_rnd.testers notice
```

Read the notice. It says what is recorded and sent, and what is never sent.

### 2. Set yourself up (once)

```bash
make tester-setup NAME="Ali Rezaei" EMAIL=ali@example.com REPO=git@github.com:Shahriyar-Moradi/dclab-records.git AGREE=yes
```

Add `CLAUDE_CODE=yes` only if you also want to share your own Claude Code sessions on this repository (with anything
shaped like a key removed). Without it, they never leave your machine.

What setup does:

- gives you an id;
- turns your workspace's consent on (to improve DCLab and to train its models);
- turns Claude beside DCLab on;
- caps your month at 15 (`DCLAB_WORKSPACE_MONTHLY_EUR` in your `.env`);
- clones the records repository into your workspace and puts it on your branch, `tester/<your id>`.

Your branch starts empty: nobody else's records come with it.

Then add your own key to `.env` (`ANTHROPIC_API_KEY=…`, `DCLAB_LLM_PROVIDER=anthropic`). Keep `DCLAB_DATABASE_URL` empty:
a tester's workspace keeps its records in files, and a share refuses otherwise.

### 3. Solve your problem

```bash
make notebook
```

Open the page it prints. Describe your problem in your own words on Home, answer DCLab's questions, upload your table
(or ask for an invented one), and carry it to a result. Say when something is unclear or wrong: your corrections are the
most useful thing you can give.

### 4. Share (now and then, and before you stop for the day)

```bash
make share                      # asks once per table (again when its content changes)
make share ARGS="--tables no"   # never send a table
make share ARGS=--forget-tables # decide every table again
```

**What is sent.** Only the files that changed since your last share, whole:

- your sessions and the decisions DCLab made;
- the audit;
- Claude beside DCLab, step by step;
- the intern's sessions, every saved version of them;
- the teams' boards and walks;
- the students DCLab trains, and the judgments made of them;
- every model request whole (its prompt, its reply and the model's thinking);
- the calls an MCP client made;
- your projects' and drafts' records;
- the tables you said yes to;
- with `CLAUDE_CODE=yes`, your Claude Code sessions on this repository.

**What is never sent.**

- **Your `.env`, a key, or a file over 45 MB.**
- **A table you said no to.** The values of its cells are taken out of the records that quote them (a column's examples,
  previews, top values).
- **Screenshots.** No redaction can read them.

**If you are offline,** the bundle is committed and the next `make share` sends it.

### 5. Stop, or ask for removal

- To stop sharing, stop running `make share`, or turn your workspace's consent off.
- Deleting a project in the app does **not** remove what you already shared.
- To have something removed, ask the owner. The owner's `store forget` removes it from the store, and the removal is
  recorded.

## Testing together

- **A weekly problem.** The owner proposes one problem a week, or you bring your own. Each of you solves it and shares.
- **A shared checklist** for each run:
  - did DCLab understand the problem?
  - did it ask the right questions?
  - did it catch the leaking column?
  - did the number it gave hold up?
  - what did you have to correct?
- **A bug.** File it with your share's id: the bundle folder `tester/<your id>` → `<id>/<time>` that `make share` printed.
  The owner can then read exactly what you did.
- **After each round,** the owner runs `make collect` and reads the Admin pages: Sessions, Decisions and Evaluation.

## For the owner: once

1. The private records repository is `dclab-records`. The owner's own store lives on its `owner/main` branch, and each
   tester gets `tester/<id>`.
2. Invite each friend with write access (Settings → Collaborators), and give them read access to this code repository.
3. Protect the branches that are not theirs (Settings → Branches → a rule for `owner/*`, and for each `tester/<id>`
   allow only that tester, or protect `tester/*` and allow pushes from the collaborators who own each). The tool itself
   only ever pushes a tester's own branch.
4. Clone the records repository and put its path in `.env`: `DCLAB_TESTER_RECORDS=/path/to/dclab-records`.

## For the owner: now and then

```bash
make collect   # pulls, reads every tester/* branch, rebuilds each tester under agent_runs/testers/<id>/
```

How it works:

- **Each branch is read in a worktree of its own;** none is checked out in place.
- **A branch must hold its own tester's folder:** a branch holding another tester's folder is skipped, and the log says so.
- **Bad files are never applied:** a file that does not match its manifest's hash, a link, or a path outside the tester's
  folder.
- **Testers set up before branches** keep their folders on the shared branch, and are still collected.

`make collect` ends with a table per tester: sessions, drafts, projects, Claude's turns, decisions and model spend. The
store's keepers then keep each tester's workspace, under that tester's own consent.

---

## راهنمای فارسی

**برای دوستانی که DCLab را آزمایش می‌کنند**

۱. **یک بار:**
   - مخزن را clone کنید، محیط را بسازید و متن اطلاعیه را بخوانید (`python -m dclab_rnd.testers notice`).
   - بعد `make tester-setup` را با نام، ایمیل، آدرس مخزن ثبت‌ها و `AGREE=yes` اجرا کنید.
   - اگر می‌خواهید جلسه‌های Claude Code شما روی همین مخزن هم فرستاده شود، `CLAUDE_CODE=yes` را هم اضافه کنید. در این حالت هر
     چیزی که شبیه کلید باشد پاک می‌شود.
   - این کار شاخهٔ خودتان، `tester/<شناسهٔ شما>`، را می‌سازد. شاخه خالی شروع می‌شود و ثبت‌های هیچ‌کس دیگری با آن نمی‌آید.

۲. **کلید:** کلید Anthropic خودتان را در `.env` بگذارید و `DCLAB_DATABASE_URL` را خالی بگذارید.

۳. **حل مسئله:** با `make notebook` مسئلهٔ خودتان را حل کنید. هر جا DCLab اشتباه کرد بگویید؛ اصلاح‌های شما ارزشمندترین داده
   است.

۴. **ارسال:** گاه‌به‌گاه `make share` را اجرا کنید.
   - فقط چیزهایی که تغییر کرده فرستاده می‌شود، و فقط به شاخهٔ خودتان.
   - جدول‌ها فقط با «بله» شما فرستاده می‌شوند.
   - `.env`، کلیدها، فایل‌های بزرگ‌تر از ۴۵ مگابایت و تصویرها هرگز فرستاده نمی‌شوند.

۵. **حذف:** پاک کردن یک پروژه در برنامه چیزی را که قبلاً فرستاده‌اید پاک نمی‌کند. برای حذف از مالک بخواهید.

**آزمایش با هم:**

- هر هفته یک مسئله؛
- یک فهرست بررسی مشترک؛
- هر اشکال با شناسهٔ بستهٔ ارسالی‌تان گزارش شود.

**برای مالک:**

- هر آزمایش‌کننده را با دسترسی نوشتن به مخزن `dclab-records` دعوت کنید.
- شاخه‌های دیگران را محافظت کنید.
- `make collect` را اجرا کنید. این فرمان همهٔ شاخه‌های `tester/*` را می‌خواند و فضای کار هر آزمایش‌کننده را می‌سازد.
