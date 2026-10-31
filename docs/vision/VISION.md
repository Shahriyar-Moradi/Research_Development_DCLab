# DCLab vision

The owner's vision, 1 November 2026: **the goal DCLab must go for.** Every plan, phase and package should say how it
moves DCLab toward it. The full page, English and Persian side by side, with examples, today's state and the order of
work, is [`vision.html`](vision.html).

## English

DCLab is an AI-powered platform that takes companies from raw data to trusted models, and from models to better
decisions.

For data scientists and ML engineers, DCLab is a smart assistant. An AI agent first understands the problem, then
designs and runs the right ML pipeline. Before a model leaves DCLab, it is checked: no leakage, an honest score, and
the code that rebuilds it. Developers work from their own notebooks and spend their time on the real problem instead
of repetitive work.

For businesses, DCLab goes one step further. Soon every company will run many models, each making its own
predictions. But predictions alone are not enough. Companies need help turning them into the best decisions. DCLab
will give each company its own decision model, built on its own data, that grows together with the company and its
business.

Every business decision workflow starts with a data science and ML team. DCLab supports that team and the
decision-makers they serve, as one assistant for both.

## فارسی

<div dir="rtl" lang="fa">

DCLab یک پلتفرم مبتنی بر هوش مصنوعی است که شرکت‌ها را از داده‌ی خام به مدل‌های قابل اعتماد می‌رساند، و از مدل‌ها به
تصمیم‌های بهتر.

برای دانشمندان داده و مهندسان یادگیری ماشین، DCLab یک دستیار هوشمند است. یک عامل هوش مصنوعی اول مسئله را می‌فهمد، بعد
خط لوله‌ی درست یادگیری ماشین را طراحی و اجرا می‌کند. قبل از این‌که مدلی از DCLab بیرون برود، بررسی می‌شود: بدون نشت
داده، با نمره‌ی صادقانه، و همراه با کدی که دوباره آن را می‌سازد. برنامه‌نویس‌ها از همان نوت‌بوک خودشان کار می‌کنند و
وقتشان را روی مسئله‌ی اصلی می‌گذارند، نه روی کارهای تکراری.

برای کسب‌وکارها، DCLab یک قدم جلوتر می‌رود. به‌زودی هر شرکت مدل‌های زیادی خواهد داشت که هر کدام پیش‌بینی خودش را
می‌کند. اما پیش‌بینی به‌تنهایی کافی نیست. شرکت‌ها کمک لازم دارند تا این پیش‌بینی‌ها را به بهترین تصمیم تبدیل کنند.
DCLab به هر شرکت مدل تصمیم‌گیری مخصوص خودش را می‌دهد؛ ساخته‌شده روی داده‌ی خود شرکت، که همراه با شرکت و کسب‌وکارش رشد
می‌کند.

هر روند تصمیم‌گیری در کسب‌وکار با یک تیم علم داده و یادگیری ماشین شروع می‌شود. DCLab از این تیم، و از تصمیم‌گیرانی که
این تیم برایشان کار می‌کند، پشتیبانی می‌کند؛ یک دستیار برای هر دو.

</div>

## Words

- **The company's decision model** (in this vision) turns many models' predictions, plus the company's costs, budgets
  and rules, into one best action. It is not the same thing as the small *decision models* of the ledger
  (`dclab_core/decision_models.py`), which answer DCLab's own technical choices, such as a column's role.
- **The order of work:** the data team first (the assistant that builds models right), then the decision layer on
  top of those models, then each company's decision model learning from its decisions and their outcomes.
