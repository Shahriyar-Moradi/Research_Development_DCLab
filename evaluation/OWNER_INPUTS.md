# Inputs needed for a credible DCLab agent evaluation

Please answer in plain language; no code or API secret is needed in chat.

## HyperAck

- At what exact event does the model make its decision (order creation, rider
  assignment, dispatch, or another event)?
- What outcome is it predicting, and over what time window?
- Which columns are guaranteed to exist **at that moment**? Are final fares,
  acceptance/completion outcomes, timestamps and coordinates known then?
- Is the desired validation split by order, customer/rider, geography, and/or
  later time? Which entities could appear in both training and deployment?

## Churn

- What is the customer snapshot date and prediction horizon (for example,
  "given data at month-end, churn in the next 30 days")?
- Does `cancellation_date` exist before the decision, or only after a customer
  requests cancellation? Is `contract_duration` contracted tenure or observed
  tenure ending at churn?
- What actions should this prediction support, and what error is more costly:
  missing a future churner or contacting someone who will stay?

## Agent quality review

- Who can independently review 20–30 anonymized notebook comments and agent
  decisions (ideally a data scientist other than the author)? Two reviewers
  help reveal ambiguous labels.
- For each reviewed suggestion, mark: correct / partially correct / incorrect;
  actionable / not actionable; evidence citation relevant / irrelevant; and
  would you use it in a real notebook?
- Provide two or three real tasks you expect the assistant to help complete,
  with a success criterion and a time budget. HyperAck and churn are already
  selected as the first sources; there is no need to reselect them.

## Provider access

The configured model currently raises `InternalServerError` before returning
an answer. Check the OpenAI account's API access, billing/quota and model
availability from your own dashboard. Do **not** paste the API key into chat.
The key belongs only in the local `.env` or environment; a working connection
is needed before live decision-quality scores can be reported.

These inputs turn the current development pilot into a defensible evaluation
against your actual workflow. Until then, `overall_release_ready` remains false.
