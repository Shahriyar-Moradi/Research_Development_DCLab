# DCLab renewal workspace guide

[Open the team room](index.html) to decide which of 120 customers approaching renewal should receive a call and a €10 credit. Everything runs locally. Open the HTML file in a desktop browser, keeping its adjacent JavaScript and CSS files in the same folder. The demo uses invented accounts and sends no messages to customers.

## First visit

1. Choose **Watch the teams work**. Business sends three requests. Data Science checks the prediction target while Sales and Marketing work on call capacity and budget.
2. Follow the desk conversation as the simulation switches between teammates. **Pause and take over** lets you make choices. **Step** advances one simulated team decision. Selecting a desk or switching views also pauses playback.
3. Watch mode stops when Business has a proposal. Open **Decision brief**. Under the default limits, 18 call slots and €120 of credit allow **12 offers**, with another 12 customers assigned to comparison.
4. Choose **Open Business desk**, then **Approve and reveal renewals**. Read the closing handoffs from Sales, Marketing and Data Science. Return to the brief for group outcomes and **Download report** for a Markdown record. Restart opens a confirmation before clearing this case.

## Find your way around

| View | What to use it for |
| --- | --- |
| Team room | Choose a team desk, read its conversation, inspect the current task and act as its human owner. The shared campaign brief summarizes inputs returned so far. Filter agent handoffs by team to follow a particular exchange. |
| Model evidence | Inspect the prediction contract, rows available for training and validation, each model run, and the risk distribution among eligible customers. |
| Decision brief | Inspect committed limits, try a different budget, request revisions, compare proposal versions, filter the customer list and download a report. |
| Calculation walkthrough | Open a separate step-by-step scenario to explore the arithmetic. Its controls and saved run are independent of the team room. |

On a narrow screen, scroll the team strip horizontally to reach all five desks. Opening a desk shows the task for that person. **Agent report** summarizes the desk’s last handoff and the next decision. An optional note attaches to the next manual action and stays in that desk’s transcript. Use the action buttons to make decisions.

## Run it yourself

| Team | Human decision | Agent action |
| --- | --- | --- |
| Business · Mina | Start the renewal campaign. | Ask Data Science to rank customers, Sales to confirm capacity and Marketing to confirm funds. |
| Data Science · Niloufar | Confirm non-renewal by expiry as the outcome, or ask Business about explicit cancellations. | Check whether the recorded label answers the question and send the agreed task to ML. |
| Machine Learning · Kaveh | Choose usage only or usage plus past support tickets. Review the score and send it to Data Science, or change inputs and retrain. | Fit logistic regression, compare it with a constant prediction and use the lower-error method for ranking. |
| Data Science · Niloufar | Review the interpretation and send the findings to Business. | Rank customers by risk and propose matched groups. |
| Sales · Raha | Confirm how many customers can be called, from 0 to 30. | Return the weekly call limit. The call list includes customers with contact consent and offer eligibility. |
| Marketing · Ava | Confirm the budget, from €0 to €300. | Return the credit available at €10 per offer. |
| Business · Mina | Approve the current proposal or ask for a revision. | Assemble the list after all inputs arrive, record the decision, then reveal simulated outcomes. |

Sales and Marketing can respond as soon as Business dispatches their requests. ML waits for Data Science to confirm the target. Approval requires a current proposal with at least one funded call.

## Three experiments to try

### Challenge the definition of churn

From the Data Science desk, choose **Ask about explicit cancellations**. The records tell you whether a customer renewed. An explicit cancellation would need its own event record. Business can confirm non-renewal and return the task to Data Science, or put the case on hold. On a held case, **Reopen with non-renewal as the target** sends the revised question back for review. It does not add cancellation data.

### Compare two models

At the ML desk, train with usage only. Review **Model evidence**, return to ML and choose **Change inputs and retrain**. Train again with support tickets. The two rows show the model and baseline Brier scores on the same 300 validation contracts. A lower Brier score means the predicted probabilities were closer to the observed outcomes on average. The chart counts eligible accounts in each risk band for the current ranking method. Only the evaluation you release advances the case.

The first 900 synthetic contracts provide training examples. The following 300 provide validation examples. The campaign scores 120 separate accounts. Neither customer identifiers nor renewal outcomes are model inputs. Validation results help select the model. They are not independent estimates of its future performance.

### Revise a proposal before approving it

1. Let the teams produce the default proposal. Open **Decision brief**.
2. Set **Sales capacity** to 4 and **Offer budget** to €80. The preview shows 4 planned calls, 4 comparison customers and €40 reserved. Moving the sliders only changes the preview.
3. Choose **Send revised limits to teams**. Both desks receive requests. Visit Sales and Marketing to confirm or edit their proposed values. The current list is withdrawn while either response is outstanding.
4. After both replies, inspect **Proposal history**. With the proposed inputs accepted, the old version has 12 calls and the new version has 4. The model is reused.
5. Try €0 before approval. The resulting proposal is blocked. Restore a budget of at least €10 and at least one call slot to continue.

The number of calls is the smallest of three limits: available call slots, the number of whole €10 credits the budget covers, and the number of eligible customer pairs. Customers are paired by estimated risk. A fixed random draw assigns one member of each pair to the offer group.

## Read and save the result

The customer list shows the account, group, estimated risk and model inputs. Outcomes stay hidden until Business approves. The simulator assumes an offer reduces the recipient’s underlying non-renewal risk by 24% relative to their original risk. The resulting counts are simulated observations from small groups. They cannot establish the real effect of a retention offer.

**Download report** works before or after approval. It exports the current status, recorded model scores, proposal versions, currently assigned synthetic customers and inter-team handoffs. A completed report also includes renewal outcomes. Desk notes are excluded. Reloading in the same browser restores submitted decisions when browser storage is available. Unsent notes and slider previews are temporary. Older saved sessions may contain only their latest model or proposal because earlier versions of the demo did not record structured histories.
