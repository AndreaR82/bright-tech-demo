# What counts as financial advice

Source of truth for `prompts.advice_check` in [config.yaml](config.yaml). Edit this
file first, then carry the changes into the prompt — the two are meant to agree.

That prompt is not just a runtime guardrail. `scripts/label_drafts.py` sends it to
Claude as the labelling instruction, and `scripts/build_train_data.py` bakes it into
every training example as the system message. **Whatever is written here ends up
being what the fine-tuned adapter learns.**

> Not legal advice. This is a classifier spec for a fictional bank in a booth demo.
> Verified against the sources listed at the end on 12 September 2026; see
> [§5 Currency](#5-currency) for what was checked and what is still moving.

## Sign-off

| | |
|---|---|
| **Status** | Approved |
| **Approved by** | Andrea |
| **Date** | 14 September 2026 |
| **What was approved** | This document as it stood on that date, and `prompts.advice_check` in `app/config.yaml` (3,062 characters, SHA-256 prefix `c5913ac7f94949c2`) |
| **Scope** | The two-label classifier spec for the fictional-bank booth demo. Not legal advice, and not a sign-off for use with real customers |

Any change to `prompts.advice_check` changes that fingerprint and needs a fresh
sign-off, because the prompt is also the labelling instruction and the system message
the adapter was trained on. Check it with:

```bash
uv run python -c "import yaml,hashlib; t=yaml.safe_load(open('app/config.yaml'))['prompts']['advice_check']; print(hashlib.sha256(t.encode()).hexdigest()[:16])"
```

**Open at sign-off:** a product verdict that names neither him nor a class of people
("This product would be best for managing debt while building savings", given in answer
to his own question) is not settled by §4. Labellers split on it; 11 of 1,460 training
rows turn on it.

---

## 1. The statutory tests

`Corporations Act 2001` (Cth) Ch 7. Three questions, in order.

**Is it advice at all?** (`s766B`) Financial product advice is a recommendation or
statement of opinion, or a report of either, that is intended to influence a person
in making a decision about a particular financial product — or that could reasonably
be regarded as being intended to have that influence.

ASIC glosses this two ways that matter here:

- Advice "generally involves a qualitative judgement about — or an evaluation,
  assessment or comparison of — some or all of the features of a financial product"
  (RG 244.26). Description alone is not advice.
- Factual information "may be likely to be advice if it is presented in a way that is
  intended to, or can reasonably suggest or imply an intention to, **make a
  recommendation about what a client should do**" (RG 244.29). The facts need not be
  wrong or slanted; it is the recommendation riding on them that crosses over.

Factual information itself is "objectively ascertainable information, the truth or
accuracy of which cannot reasonably be questioned" (RG 244.24).

**If it is advice, is it personal or general?** (`s766B(3)`) Personal advice is
advice given or directed to a person in circumstances where:

- the person giving the advice **has considered** one or more of the client's
  objectives, financial situation and needs; **or**
- **a reasonable person might expect** the person giving the advice to have
  considered one or more of those matters.

Everything else is general advice (`s766B(4)`). Note the limbs are **disjunctive** —
either one is enough — and note the second says *might* expect, not *would*. That is
a deliberately low bar, and courts have read "might expect" as wider than "would
expect".

In `Westpac Securities Administration Ltd v ASIC` [2021] HCA 3 the High Court held
that "considered" simply means "took account of" — no active process of evaluation
and reflection is required — and that the second limb is an objective test assessed
against all the circumstances in which the advice was given. Taking account of even
one aspect of one of those three matters is enough.

**What does that trigger?** General advice to a retail client needs a general advice
warning (`s949A(2)`). Personal advice to a retail client pulls in the best interests
duty (`s961B`) and a Statement of Advice (`s946A`), and providing it at all requires
an AFS licence or an authorisation from a licensee. That gap is why this demo blocks
one and merely flags the other.

## 2. Possessing data is not the same as using it

This is the subtlety the demo kept getting wrong, and it cuts both ways.

**Merely holding a customer's information does not make a reply personal advice.**
RG 244.36 is blunt: "The test is not whether you merely possess information about the
client's relevant circumstances." You may even *use* what you know to choose or shape
general advice so it is relevant, and ASIC will not treat that as personal advice
provided you do not in fact consider the client's circumstances when preparing it,
and it is unlikely the client would expect the advice to reflect such consideration
(RG 244.46–.49).

Example B3 makes the point for factual answers: an adviser told a client's situation
who responds with the objectively ascertainable rules on early super access has given
factual information, not advice, because it "offers no recommendation or statement of
opinion intended to influence the client".

**But expectation does the work, and a warning does not cure it.** Example C4 is the
closest thing ASIC publishes to this demo: an interactive web application, carrying a
general advice warning, whose recommendations are all pre-written, where the user's
answers only decide which page they land on. ASIC's verdict is that this **is**
personal advice — "because it is likely that the client would expect, from being
asked for detailed personal information, that the provider is seeking information
about all their relevant circumstances in order to make a recommendation that is
appropriate for them. **The presence of a general advice warning does not alter
this.**" RG 244.47 says the same directly: if you do in fact consider the client's
circumstances, "you cannot avoid this by giving a general advice warning".

**So the rule for this assistant.** It answers one to one, it visibly holds his
accounts and eighteen months of transactions, and the specialists query that data
before the writer drafts. If it answers "should I fix or go variable?" with a steer,
a reasonable person in his position **might** expect his circumstances to have been
considered — limb (b) is satisfied and it is personal advice, even though the reply
names none of his numbers. The justification is the expectation, not the mere
possession of the data, and slapping a warning on it fixes nothing.

## 3. The two labels

The pipeline's labels are not a straight copy of the statutory categories:

| Label | ASIC category | App behaviour |
|---|---|---|
| `factual_information` | Factual information, **plus** general advice | pass (+ warning on product answers) |
| `personalised_recommendation` | Personal advice | 🚫 block → rewrite |

**Why two labels and not three.** The detector was three-class until three labelling
passes over the same 50 drafts showed the middle class was not learnable. It split on
verb choice over identical products — "lets your savings **reduce** the interest" was
general, "**includes** an offset account" was factual — a distinction a 4B model would
approximate as *contains the word "reduce"*. Collapsed to a binary block / don't-block
decision, the same three passes agree on 49 of 50 drafts (98%, against 84% three-way),
and every one of the unstable drafts was a middle-class boundary case.

General advice did not disappear from the demo, it stopped being a model judgement:
every `product_advisor` answer now carries the general-advice warning as a
deterministic rule in `pipeline.py`. That is the conservative direction — an
unnecessary warning is harmless, a missing one is not — and it matches RG 244.47 and
Example C4, which say a warning cannot cure advice that *is* personal anyway. The
model is left deciding only the thing it can actually decide, and the thing that
matters: whether the draft steers this customer.

**The deliberate divergence.** Strictly, explaining a product feature — "fixed rates
keep repayments the same for the term" — is factual information, not general advice,
since nothing is evaluated or recommended; RG 244.30 would let it stay factual if we
clarify as much and imply no recommendation. The demo routes it to
`factual_information` and attaches the warning by route instead. Attaching a warning that
was not needed is harmless; omitting one that was is not.

**The carve-out the other way.** Reporting the customer's own money back to him is
always factual, however detailed, and however much he spent — his transaction history
is objectively ascertainable and presenting it evaluates nothing. Arithmetic over that
data stays factual too: "halving your delivery spend would free up about $180 a month"
states a consequence, it does not tell him to do it. RG 244.33–.34 and Examples B2/B3
support this directly. Without the carve-out the spending analyst would be blocked on
every honest answer it gives.

## 4. Decision procedure

One question decides it.

**Does the draft point this customer at a particular financial product, or at a course
of action about one** — saying it suits him, would benefit him, or is what he should
do? Either resting on his circumstances, or put so that a reasonable person **might**
expect his circumstances were taken into account.

- yes → `personalised_recommendation` — blocked, then rewritten
- no → `factual_information` — released

Everything that is not a steer lands in `factual_information`: rates, fees, minimums,
eligibility rules, what a product's features are and how they work, several products
set side by side, and his own numbers.

Three things do **not** cross the line, and each exists because a labelling pass got
them wrong:

- **Referring him to a person.** "You should speak to a lender" recommends a human,
  not a product. `s766B` requires advice to concern a financial product, and
  `prompts.writer_careful` is explicitly instructed to say this, so it saturates the
  careful drafts.
- **Budget guidance.** "You would need to cut back on takeaway" is directive but names
  no product, so it falls outside Ch 7 entirely.
- **Who a product suits in general.** "Offsets suit people who keep a large balance"
  describes a class of person, not this customer. Only steering *him* crosses.

The one distinction that needed stating outright, because two labelling passes split
on it silently: **"Yes, you might qualify for the First Home Starter" is personalised;
"the First Home Starter is for first home buyers only" is factual.** Same eligibility
facts — what differs is whether the draft says it about him.

> **The two changelog sections below are history, not current guidance.** They record
> the three-class era and the reasoning that ended it. Where they say "step 2" or
> "`general_information`", read them as describing a taxonomy that no longer exists;
> §4 above is the live rule. They are kept because the audit trail is the reason the
> binary decision is trusted.

### Four corrections from the first labelling sample

The first version of this procedure asked about evaluation *first* and put the
personal check second. A 50-draft sample labelled against it produced two structurally
identical drafts — three products with their rates and minimums, no opinion in either
— with two different labels, and flipped a third to `general_information` on the
single word "lower". That is noise, not a line. Four fixes, each grounded in a source:

1. **Juxtaposition is not comparison.** Example B1 says listing "objectively
   ascertainable factual information about specific product characteristics" is not
   advice, since it involves no qualitative judgement. Example B2 goes further: an
   operator who "explains the difference" between two products has given **factual
   information**, because they have "not given an opinion or made a recommendation". RG
   244.26's word "comparison" means evaluative comparison, not setting facts side by
   side. Bare `compare` has been dropped from the test; the trigger is judgement.
2. **The personal check must run first.** Under stop-at-first-match, asking about
   opinion first let anything aimed at him escape to `factual_information` whenever it
   was not phrased as an opinion — "Yes, you might qualify for the First Home Starter"
   never reached the personal test at all. The order is now reversed.
3. **Referring him to a human is not product advice.** `prompts.writer_careful`
   explicitly instructs the model to "suggest speaking to a lender", so that phrasing
   saturates the careful drafts. Left uncarved, "I suggest" would appear throughout
   the factual training examples while the prompt lists it as a personal cue — a
   directly contradictory signal. `s766B` requires advice to concern a *financial product*;
   a person to talk to is not one.
4. **Budget guidance is outside Ch 7.** "You would need to adjust your spending in
   those areas" tells him to act, but not about any financial product, so it is not
   financial product advice. The carve-out is now scoped to products explicitly.

### Four more from the second sample

Re-labelling the same 50 drafts against those fixes swung the counts to 42 / 1 / 7 —
the middle class all but disappeared. The cause was a self-contradiction introduced by
fix 1 above: **`general_information`'s own examples stopped routing to
`general_information`.** "Fixed rates keep repayments the same for the term, while
variable rates can move up or down" is *explaining how they differ*, which the new
carve-out called factual; "An offset account reduces the interest charged" carries no
judgement at all and fell through to factual. Only the "suits people who keep a large
balance" example still qualified. A prompt whose worked examples fail its own
procedure cannot produce coherent training data, and a demo that never emits
`general_information` never shows the warning path SPEC.md promises.

5. **The line is data versus mechanism, not juxtaposition versus comparison.**
   Listing rates, fees, minimums and eligibility side by side stays factual — that is
   ASIC's Example B1. Explaining how a product *behaves*, or what effect a feature
   has, now belongs to step 2. This keeps the structurally identical drafts together
   while restoring the middle class.
6. **Step 1 was unscoped.** It said "product **or course of action**", which made any
   directive aimed at him personal — including budget guidance that carve-out 3 had
   just declared outside Ch 7. Step 1 is now "a course of action *about a financial
   product*", so the two stop fighting.
7. **The lender carve-out needed to override the surface cue explicitly.** "You should
   speak to a lender" contains both "you should" and a referral; the prompt listed the
   first as a personal cue and the second as factual, leaving the reader to guess
   which wins. It now says the carve-out applies *even phrased* that way.
8. **Conditionals needed a rule.** "If you prefer no fees, the Basic Variable has a
   lower rate" and "If you are a first home buyer, the Deposit Saver allows balances
   up to $150,000" share a surface shape but differ: the first matches a *preference*
   (a suitability match, step 2), the second states *eligibility* (factual). Three
   labels in the sample turned on a distinction the prompt never drew; it now does.

### The shortcut problem — unresolved

An earlier draft of this document waved this away: drafts like "Since you are looking
for a card, X is best if you pay monthly" hinge on a four-word prefix, and that was
called "the genuine distinction", so it stayed. **Deleting the middle class changed
the stakes of that decision and nobody re-read it.** Stripping the prefix used to
demote a draft to a warned class; now it demotes it to a full release.

Measured on the 50-draft sample, the damage is exact. Predicting
`personalised_recommendation` iff the draft opens with "Since you are", "Given your"
or "Yes," gives **7 true positives, 0 false positives, 0 false negatives** — perfect
separation with no understanding of advice at all. A second free rule sits alongside
it: all 11 drafts mentioning a lender or specialist are factual.

A model trained on this learns the templates. Worse, `advice_test.jsonl` is drawn from
the same generator as the training split, so `eval_advice.py` would score that model
well and the adapter would ship. **A good eval number here would measure prefix
detection.** That is a worse outcome than a failed training run, because a failed run
is visible.

The fix is counterfactual data, not more volume: prefix-stripped drafts keeping their
true label so the cue stops being *sufficient*; prefixed-but-factual drafts so it
stops being *necessary*; personalised drafts that open neutrally and steer mid-body;
and lender referrals attached to genuinely personalised drafts so the referral stops
being a free negative. Until at least the first two exist, the eval cannot tell a
guardrail from a template matcher.

### Two traps

- **Aiming is what decides, not personal detail.** "Go fixed" said to him is
  personalised even when it cites none of his numbers. The same opinion aimed at
  nobody in particular — "offsets suit people who keep a large balance" — is factual.
  Whether he is the target is the whole question.
- **Numbers alone never make a draft advice.** A page of his own figures, to the cent,
  is factual. The label turns on the recommendation, never on the precision.

### Soft forms that still count as personal

"for someone in your position" · "you'd be better off" · "I'd suggest" · "in your
case" · "given your deposit" · "that would work well for you" · "go fixed". Tentative
framing is a tone, not a legal distinction.

## 5. Currency

Checked 12 September 2026:

- The `Corporations Act 2001` compilation on the Federal Register was current to
  **27 August 2026**. `s766B`, `s946A`, `s949A` and `s961B` are all in force; ASIC was
  still issuing warnings for `s946A(1)` contraventions in 2026.
- **RG 244** (issued December 2012) and **RG 36** (issued June 2016) are both still
  listed as current ASIC guidance, with no withdrawal or supersession notice.
- **DBFO Tranche 1** — `Treasury Laws Amendment (Delivering Better Financial Outcomes
  and Other Measures) Act 2024` (No. 67, 2024), assent 9 July 2024 — covered ongoing
  fee arrangements, Financial Services Guides, conflicted remuneration and insurance
  commissions. It did **not** touch `s766B`, `s946A`, `s949A` or `s961B`.
- **DBFO Tranche 2 has not been legislated.** Draft components were released
  21 March 2025 and the package remained at announcement/consultation stage as at
  September 2026. Watch three items, none of them yet law: replacing the Statement of
  Advice with a shorter *client advice record*; simplifying the best interests duty by
  **retaining** the `s961B(2)` safe harbour steps but removing paragraph (g); and a
  New Class of Adviser regime. A proposal to rename "general advice" has been floated
  since the Quality of Advice Review but has not been enacted.

If Tranche 2 passes, revisit §1 and §3 — the SOA trigger and possibly the general
advice label would need rewording. Nothing in it currently changes the `s766B(3)`
test this classifier turns on.

### Sources

- [RG 244 *Giving information, general advice and scaled advice*](https://download.asic.gov.au/media/tkqi11il/rg244-published-13-december-2012-20211208.pdf) (PDF) — RG 244.24, .26, .29, .30, .33–.36, .43, .46–.49, Examples B2, B3, C1–C4, and the glossary's verbatim `s766B(3)` definition
- [ASIC — RG 244 landing page](https://www.asic.gov.au/regulatory-resources/find-a-document/regulatory-guides/rg-244-giving-information-general-advice-and-scaled-advice)
- [ASIC — RG 36 *Licensing: Financial product advice and dealing*](https://www.asic.gov.au/regulatory-resources/find-a-document/regulatory-guides/rg-36-licensing-financial-product-advice-and-dealing/)
- [ASIC — Giving financial product advice](https://www.asic.gov.au/regulatory-resources/financial-services/giving-financial-product-advice)
- [ASIC — Delivering Better Financial Outcomes (DBFO) package](https://www.asic.gov.au/regulatory-resources/financial-services/regulatory-reforms/delivering-better-financial-outcomes-dbfo-package) (page itself last updated November 2024)
- [Federal Register — Corporations Act 2001](https://www.legislation.gov.au/C2004A00818/latest/text) · [Act No. 67 of 2024](https://www.legislation.gov.au/C2024A00067/latest/text)
- [Hall & Wilcox — High Court clarifies the test for personal financial advice](https://hallandwilcox.com.au/news/dont-take-it-personally-high-court-clarifies-the-test-for-personal-financial-advice/) and [Bright Law case note](https://www.brightlaw.com.au/case-note-high-court-defines-personal-advice/) on *Westpac* [2021] HCA 3
- [FSC Policy Update, Issue 91, September 2026](https://fsc.org.au/news/fsc-policy-updates/fsc-policy-update-issue-91) — DBFO Tranche 2 status

---

## Appendix — the original source extracts

Kept verbatim as supplied, for provenance. These match ASIC's published wording; the
personal advice paragraph below reproduces only the **first** limb of `s766B(3)` and
stops, which is what sent the first draft of this document wrong.

> Factual information is more likely to be advice if it makes a recommendation about
> what a client should do.
>
> A recommendation or a statement of opinion constitutes financial product advice if
> it is intended to influence a person in making a decision about a particular
> financial product.
>
> Financial product advice generally involves an evaluation, assessment or comparison
> of, some or all of the features of a financial product.
>
> Under the Corporations Act, all financial product advice is either 'personal advice'
> or 'general advice'.
>
> Personal advice is financial product advice given or directed to a person (including
> by electronic means) in circumstances where the person giving or directing the
> advice has considered one or more of the client's objectives, financial situation
> and needs.
>
> All other financial product advice is general advice.

**Financial product advice and dealing**

> To determine your obligations under the licensing provisions you first need to
> consider whether you provide a 'financial service'.
>
> You provide a financial service if (among other things) you: 'provide financial
> product advice', or 'deal in a financial product'.
>
> Arranging for a person to engage in certain conduct, such as applying for or
> acquiring a financial product, will constitute dealing unless it amounts to
> providing financial product advice or is exempt.
>
> If you provide a financial service you need to consider whether you must hold an
> Australian financial services (AFS) licence or hold an authorisation from a
> licensee.
>
> For more information, see Regulatory Guide 36 Licensing: Financial product advice
> and dealing (RG 36).

**Giving information, general advice and scaled advice**

> Regulatory Guide 244 Giving information, general advice and scaled advice (RG 244)
> explains: the differences between giving factual information, general advice and
> personal advice, and how to meet the advice obligations in Ch 7 of the Corporations
> Act 2001, including the best interests duty and related obligations, when giving
> 'scaled' advice (i.e. personal advice that is limited in scope).
>
> Our guidance aims to facilitate access for retail clients to good quality
> information and advice about all financial products.

**Example statement of advice**

> Our example SOA is based on a hypothetical and limited financial advice scenario
> developed in consultation with stakeholders. The financial advice scenario deals
> with personal advice about investing in managed funds and basic deposit products
> and personal insurance, given to a new client (i.e. not in an ongoing advisory
> relationship).
>
> The advice we developed is one of a number of possible outcomes. The purpose of
> this example SOA is to illustrate clear, concise and effective disclosure and not
> to illustrate the giving of suitable or best advice.

*(The SOA material is background on disclosure obligations. It does not feed the
classifier — the demo never produces a Statement of Advice, it blocks the drafts that
would require one.)*
