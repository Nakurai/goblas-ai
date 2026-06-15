# Preparing your data

This guide explains how to get your data into a shape goblas-ai can learn from.
No prior machine-learning knowledge is assumed.

## The vocabulary, in plain terms

- A **dataset** is just a table: rows and columns, like a spreadsheet.
- Each **row** is one example you want to learn from — one house, one order, one
  day.
- A **feature** is an input column: something you know and will use to make a
  prediction (a house's size, its number of bedrooms).
- The **target** is the output column: the single value you want to predict (the
  house's price). You also hear it called the "label".
- **Training** is the process of looking at many rows where you know both the
  features and the target, and figuring out the relationship between them.

So your job in this step is to produce a table where every input you care about
is a numeric column, and the thing you want to predict is one more numeric
column.

## The input format: CSV

goblas-ai reads **CSV** files (comma-separated values), the format every
spreadsheet can export. The first line must be a **header** naming the columns.
For example, `housing.csv`:

```csv
size_sqft,bedrooms,age_years,price
2000,3,10,250000
1500,2,25,190000
3200,4,5,410000
```

Here `size_sqft`, `bedrooms`, and `age_years` are features, and `price` is the
target.

## Rule 1: every value must be a number

Version 1 of goblas-ai works with numbers only. If a column contains text, you
must convert it to numbers first. This is a manual step today, and here is how to
think about each case.

### A column with two options (yes/no, true/false)

Replace it with a single column of `0` and `1`. For example, a `has_garage`
column with `yes`/`no` becomes `1`/`0`.

### A column with a few categories (e.g. "city")

Do **not** number them `1, 2, 3`, because that would falsely tell the model that
"city 3" is somehow three times "city 1". Instead, create one yes/no column per
category. This is called **one-hot encoding**:

```
city          ->   city_paris  city_lyon  city_nice
"paris"            1           0          0
"lyon"             0           1          0
```

### A free-text column (a product description, a comment)

Text needs more involved handling than v1 covers; for now, leave such columns out
or reduce them to simple numeric signals you compute yourself (for example,
"length of the description").

## Rule 2: handle missing values before training

If a cell is empty or non-numeric, goblas-ai will stop with an error that tells
you the exact line, rather than guessing. You have two simple options:

1. **Drop the row** if only a few rows are affected.
2. **Fill the blank** with a sensible stand-in — commonly the column's average
   value. This is called *imputation*. Do the fill in your spreadsheet or a small
   script before training.

The library deliberately does not silently fill blanks for you, because the right
choice depends on your data and a silent guess could quietly distort the model.

## Loading the data in Go

Once your CSV is clean, there are two ways to load it.

### Stream it (works for files larger than memory)

`OpenCSV` reads the file a piece at a time. Nothing but the header is read up
front; the rows are read later, during training.

```go
data, err := dataset.OpenCSV("housing.csv", "price") // "price" is the target column
```

### Load it all into memory (convenient for smaller files)

`FrameFromCSV` reads the whole file into memory and gives you a `Frame`, which is
handy because you can ask it for its feature matrix and targets directly.

```go
frame, err := dataset.FrameFromCSV("housing.csv", "price")
fmt.Println(frame.Len(), "rows,", frame.NFeatures(), "features")
```

Both a streamed source and an in-memory `Frame` can be used anywhere goblas-ai
asks for data, so you can start in memory and switch to streaming later without
changing your training code.

## Splitting into training and test sets

To know whether a model is actually good, you must measure it on data it did
**not** learn from. If you judge a model only on the rows it trained on, it can
look great while having simply memorised them — a trap called *overfitting*.

So we set aside a slice of the data, the **test set**, and never train on it.
`SplitCSV` does this for you, and does it while still streaming:

```go
// Hold out 20% of the rows for testing. The seed (1) makes the split repeatable.
train, test, err := dataset.SplitCSV("housing.csv", "price", 0.2, 1)
```

A typical test fraction is 0.2 (20%). The split is deterministic: the same seed
always produces the same partition, so your results are reproducible.

## Why scaling matters (and why you usually don't have to do it yourself)

Features often live on wildly different scales: house size in the thousands,
number of bedrooms in single digits. Some training methods get slow or unstable
when scales differ that much.

The fix is **standardization**: shifting and stretching each column so it has an
average of 0 and a typical spread of 1. goblas-ai does this **automatically** by
default, and — importantly — stores the scaling inside the saved model so the
exact same adjustment is applied when you later make predictions. You feed raw
numbers in both cases; the library keeps the two consistent for you. See
[training.md](training.md) for more.

## Next step

With clean, loaded, and split data in hand, continue to
[training.md](training.md).
