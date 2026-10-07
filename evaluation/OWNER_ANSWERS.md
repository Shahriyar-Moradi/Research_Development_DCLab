# The owner's answers (package 14.1)

Write each answer in plain words under `ANSWER:`. Leave a line empty when you do not know: **an open question stays open;
nothing is filled in for you.** `python -m dclab_rnd.agent_eval.owner_inputs status` lists what is still open and checks the
contracts. Never paste an API key here.

The questions come from `evaluation/OWNER_INPUTS.md`. The columns listed are what the files in `data/project/` contain,
so you can answer in their words.

### hyperack.event
At what exact event does the model make its decision (order creation, rider assignment, dispatch, or another event)?
Columns: created_date, deliverey_category_id, weekday, first_created_at, time_bucket, total_distance, sum_product,
source_latitude, source_longitude, destination_latitude, destination_longitude, final_customer_fare, final_biker_fare,
first_customer_fare, hyper_ack.

ANSWER:

### hyperack.outcome
What outcome does it predict (the file's target is `hyper_ack`), and over what time window?

ANSWER:

### hyperack.known_at_decision
Which columns are guaranteed to exist at that moment? In particular: are `final_customer_fare`, `final_biker_fare`,
timestamps and coordinates known then?

ANSWER:

### hyperack.split
Should validation split by order, customer or rider, geography, or later time? Which entities could appear both in training and in use?

ANSWER:

### churn.snapshot
What is the customer snapshot date and the prediction horizon (for example "given the data at month-end, churn in the next 30 days")?
Columns of the Telco file: customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn.

ANSWER:

### churn.available_when
If a cancellation date or a contract length exists in your own churn data: is it known before the decision, or only after a
customer asks to cancel? Is the contract length contracted or observed until churn? (For the Telco sample: is `tenure` the
months until the snapshot, or until churn?)

ANSWER:

### churn.action
What action should the prediction support, and which error costs more: missing a future churner, or contacting someone who would have stayed?

ANSWER:

### review.reviewers
Who can independently review the benchmark's labels (package 14.4)? Two people, if possible, who did not write them. Names, or "none yet".

ANSWER:

### study.tasks
Two or three real tasks you expect the assistant to help with, each with a success criterion and a time budget (package 14.6).

ANSWER:

### study.participants
Three to five data scientists for the user study, and whether their names may be kept (package 14.6). Names, or "none yet".

ANSWER:

### provider.check
Has the OpenAI account's billing, quota and model access been checked from your dashboard? (The live checks of package 14.2 ran
on `gpt-6-luna` on 2026-10-07; say if anything changed.)

ANSWER: Recorded from the live runs, not from you, so please confirm: on 2026-10-07 the key in `.env` served `gpt-6-luna` and the live checks of package 14.2 ran without a failed request.
