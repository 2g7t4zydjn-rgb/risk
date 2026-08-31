# PasaBank Risk and AML Policy (Sample Document)

## Section 1: Transaction Amount Rules
**1.1** A single transaction that is more than 5x the customer's average
transaction amount over the last 90 days is classified as "High Risk" and
requires manual review.

**1.2** Transactions between 8x-15x the customer's average amount fall
into the "Critical Risk" category and must be blocked immediately, until
the customer is contacted.

**1.3** Transactions exceeding 15x the average are automatically blocked
and a notification is sent to the Financial Monitoring Service (FMS).

## Section 2: Time and Hour Rules
**2.1** Transactions conducted between 01:00-06:00, especially cash
withdrawals, require additional scrutiny.

**2.2** Transactions outside the customer's typical active hours, but
still during daytime, are logged as "Low Risk" - no automatic block is
required.

## Section 3: Geographic Rules
**3.1** A transaction in a country different from the customer's home
country, if the amount is above average, is classified as "Medium-High
Risk".

**3.2** Transactions from countries on the high-risk list (FATF grey/black
list) automatically receive "Critical Risk" status, regardless of the
country's specific name.

**3.3** Transactions in two different countries on the same day (a
physically impossible travel distance) must be logged as an "Impossible
Travel" fraud indicator and reported to the FMS immediately.

## Section 4: Frequency Rules
**4.1** 3 or more transactions within a 15-minute window are logged as
"Structuring Activity" suspicion - this may indicate an attempt to split
amounts to evade limits.

**4.2** More than 10 transactions within 24 hours, regardless of total
amount, are flagged for manual review.

## Section 5: Recommended Actions
- **Low Risk**: log it, no action required
- **Medium Risk**: send an SMS/push notification to the customer, monitor the transaction
- **High Risk**: send to a risk analyst for manual review, temporarily hold the transaction
- **Critical Risk**: block the transaction immediately, require a phone call with the customer, notify the FMS
