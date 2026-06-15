# Training a model

This guide explains what training does, the two ways goblas-ai can train a linear
model, and how to read the quality scores afterwards. It assumes you have clean,
loaded data (see [data-cleaning.md](data-cleaning.md)).

## What a linear-regression model actually is

A linear-regression model predicts the target as a **weighted sum** of the
features, plus a constant. With features for size, bedrooms, and age, it looks
like:

```
predicted_price = base
                + w_size     * size
                + w_bedrooms * bedrooms
                + w_age      * age
```

- Each `w_...` is a **coefficient**: one number per feature. It says how much the
  prediction moves when that feature goes up by one unit, holding the others
  fixed. A coefficient of `100` for `size` means "each extra square foot adds
  100 to the predicted price."
- `base` is the **intercept**: the starting value before any feature is taken
  into account.

Training is simply the search for the coefficients and intercept that make the
predictions as close as possible to the true targets across your training rows.

## Training in three lines

```go
lr := linear.NewRegression()           // sensible defaults
err := lr.Fit(context.Background(), train)
preds, _ := lr.PredictBatch(test.Features())
```

`Fit` does all the work. The defaults are: fit an intercept, standardize the
features (see [data-cleaning.md](data-cleaning.md)), and choose the training
method automatically.

## The two training methods (solvers)

A "solver" is the method used to find the coefficients. goblas-ai has two, and by
default picks one for you. You can override the choice if you want.

### 1. Closed-form (the exact method)

There is a mathematical formula — the *normal equation* — that computes the
single best set of coefficients in one shot, no trial and error. goblas-ai
streams your data through this calculation, so it works even on files larger than
memory, and it needs no settings to tune.

```go
lr := linear.NewRegression(linear.WithSolver(linear.ClosedForm))
```

Use it when you have a modest number of feature columns (up to a few hundred). It
is exact and fast there.

### 2. SGD (the iterative method)

SGD stands for *stochastic gradient descent*. Instead of solving in one shot, it
starts with a guess and **nudges** the coefficients in the right direction over
many small steps, looking at a batch of rows at a time. It is the method that
scales to very wide data (very many feature columns) and is what powers
[online learning](online-learning.md).

```go
lr := linear.NewRegression(
	linear.WithSolver(linear.SGD),
	linear.WithEpochs(50),         // how many passes over the data
	linear.WithLearningRate(0.05), // how big each nudge is
)
```

Two settings matter for SGD:

- **Epochs**: one epoch is one full pass over the training data. More epochs give
  the method more chances to improve, at the cost of time. Start around 10–50.
- **Learning rate**: the size of each nudge. Too large and the steps overshoot
  and the model never settles (you'll see nonsense numbers); too small and it
  learns very slowly. `0.01`–`0.1` is a reasonable range *when features are
  standardized* (which is the default). If results look unstable, lower it.

There is also `WithL2(...)`, which gently discourages large coefficients to
reduce overfitting. Leave it at 0 unless your model fits the training data far
better than the test data.

### Auto (the default)

If you don't choose, goblas-ai uses `Auto`: closed-form when there are few
features, SGD when there are many. Either way the data is streamed, so dataset
size is not what decides it.

## Measuring how good the model is

After training, run the model on the **test set** — the rows it never saw — and
compare its predictions to the truth. goblas-ai gives you three scores, each
answering a different question.

```go
import "github.com/nakurai/goblas-ai/metrics"

r2   := metrics.R2(truth, preds)
rmse := metrics.RMSE(truth, preds)
mae  := metrics.MAE(truth, preds)
```

- **R² ("R squared")** — *How much of the variation does the model explain?* It
  ranges from 1.0 (perfect) down through 0.0 (no better than always guessing the
  average target) and can go negative (worse than guessing the average). It has
  no units, so it's good for a quick "is this any good?" check. 0.9 means the
  model explains 90% of the ups and downs in the target.
- **RMSE (root mean squared error)** — *How big is a typical error, in the
  target's own units?* If you predict prices in euros, an RMSE of 8000 means
  predictions are typically off by roughly €8000. It penalises big misses
  heavily.
- **MAE (mean absolute error)** — *What is the average error, treating all misses
  evenly?* Also in the target's units, but unlike RMSE it does not over-weight
  large errors, so it's a gentler "typical miss".

Use R² to judge overall fit, and RMSE/MAE to understand the error in real terms.

## Reading what the model learned

You can pull out the coefficients in the **original units** of your features
(goblas-ai undoes the internal scaling for you), which makes them interpretable:

```go
coef, intercept := lr.Coefficients()
for i, name := range lr.FeatureNames() {
	fmt.Printf("%+.2f per unit of %s\n", coef[i], name)
}
fmt.Printf("%+.2f base value\n", intercept)
```

This is one of linear regression's biggest advantages: the model is not a black
box. You can read, sanity-check, and explain it.

## Saving the result

```go
lr.SaveFile("price.gobl")
```

That file contains everything needed to make predictions later, including the
feature scaling. See [deployment.md](deployment.md) for how to reuse it.

## Next steps

- To update a model as new data arrives, see
  [online-learning.md](online-learning.md).
- To use a trained model in another program, see
  [deployment.md](deployment.md).
